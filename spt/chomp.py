import time
from typing import Optional, Sequence

import numpy as np
import torch

from .environment import EsdfEnvironment

DTYPE = torch.float64

# ---------------------------------------------------------------------------
# Matrix builders
# ---------------------------------------------------------------------------

def _build_finite_diff_matrix(n: int, dof: int = 3) -> torch.Tensor:
    K = torch.zeros((n + 1, n), dtype=DTYPE)
    for i in range(n):
        K[i, i]     =  1.0
        K[i + 1, i] = -1.0
    return torch.kron(K, torch.eye(dof, dtype=DTYPE))

def _build_smoothness_matrix(n: int, dof: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    K = _build_finite_diff_matrix(n, dof)
    A     = K.T @ K + 1e-6 * torch.eye(n * dof, dtype=DTYPE)
    A_inv = torch.linalg.inv(A)
    return A, A_inv

def _build_b_vector(start: torch.Tensor, goal: torch.Tensor, n: int, dof: int = 3) -> torch.Tensor:
    K = _build_finite_diff_matrix(n, dof)
    e = torch.zeros((n + 1) * dof, dtype=DTYPE)
    e[0:dof]  = -start
    e[-dof:] =  goal
    return K.T @ e

def _build_c_scalar(start: torch.Tensor, goal: torch.Tensor, n: int, dof: int = 3) -> torch.Tensor:
    e = torch.zeros((n + 1) * dof, dtype=DTYPE)
    e[0:dof]  = -start
    e[-dof:] =  goal
    return 0.5 * (e @ e)


# ---------------------------------------------------------------------------
# Initial path helpers
# ---------------------------------------------------------------------------

def resample_path(path: np.ndarray, num_waypoints: int) -> np.ndarray:
    """Resample a polyline to num_waypoints points spaced uniformly by arc length."""
    path    = np.asarray(path, dtype=np.float64)
    seg_len = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s       = np.concatenate([[0.0], np.cumsum(seg_len)])
    s_new   = np.linspace(0.0, s[-1], num_waypoints)
    return np.stack([np.interp(s_new, s, path[:, d]) for d in range(path.shape[1])], axis=1)


# ---------------------------------------------------------------------------
# Feasibility check
# ---------------------------------------------------------------------------

def find_collisions(env, trajectory: np.ndarray, step: float) -> np.ndarray:
    """
    Samples every segment of the trajectory every `step` metres (not only the waypoints, so thin
    walls between two waypoints are found) and returns the samples that are inside an (inflated)
    obstacle or outside the map. An empty result means the trajectory is feasible.
    """
    traj    = np.asarray(trajectory, dtype=np.float64)
    samples = [traj[:1]]
    for a, b in zip(traj[:-1], traj[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        samples.append(a + (b - a) * np.linspace(0.0, 1.0, n + 1)[1:, None])
    samples = np.vstack(samples)
    dist    = env.esdf_query(samples)
    return samples[~(dist >= 0.0)]      # NaN (outside the map) counts as infeasible


# ---------------------------------------------------------------------------
# Planner CHOMP
# ---------------------------------------------------------------------------

class CHOMPPlanner:
    def __init__(
        self,
        env: EsdfEnvironment,
        start: np.ndarray,
        goal: np.ndarray,
        params,
        init_path: Optional[Sequence[np.ndarray]] = None,
    ):
        """
        Args:
            env       : ESDF environment
            start/goal: (3,) positions [m]
            params    : PlannerParams class from config.py
            init_path : optional initial path (e.g. from RRT*); resampled to the
                        configured waypoint density. None → noisy straight line.
        """
        self.params     = params
        self.DOF        = params.DOF
        self.epsilon    = params.OPT_EPSILON

        self.env        = env
        # Corners of the ESDF grid: the potential clamps queries into it (see _esdf)
        self.map_lo     = np.array([ax[0]  for ax in env.esdf_interpolator.grid])
        self.map_hi     = np.array([ax[-1] for ax in env.esdf_interpolator.grid])

        self.start      = torch.as_tensor(np.asarray(start, dtype=np.float64), dtype=DTYPE)
        self.goal       = torch.as_tensor(np.asarray(goal, dtype=np.float64), dtype=DTYPE)
        self.target_z   = (self.start[2] + self.goal[2]) / 2

        if init_path is not None:
            self.init_xi = self._init_xi_from_path(init_path)
        else:
            self.init_xi = self._init_xi_straight_line()

        self.A, self.A_inv = _build_smoothness_matrix(self.n, self.DOF)
        self.b             = _build_b_vector(self.start, self.goal, self.n, self.DOF)
        self.c             = _build_c_scalar(self.start, self.goal, self.n, self.DOF)
        self.K             = _build_finite_diff_matrix(self.n, self.DOF)

    # ------------------------------------------------------------------
    # Trajectory initialisation
    # ------------------------------------------------------------------

    def _num_waypoints(self, length: float) -> int:
        return max(3, int(round(length * self.params.PAT_NUM_WAYPOINTS_PER_M)))

    def _init_xi_straight_line(self) -> torch.Tensor:
        """Straight-line init, inner waypoints only, shape (n, DOF). With some noise"""
        init_distance = torch.linalg.norm(self.start - self.goal).item()

        self.num_waypoints = self._num_waypoints(init_distance)
        self.n = self.num_waypoints - 2

        t = torch.linspace(0, 1, self.num_waypoints, dtype=DTYPE).reshape(-1, 1)
        xi = self.start + t * (self.goal - self.start)

        noise       = self.params.PAT_NOISE_INIT * torch.randn_like(xi)
        xi         += noise
        xi[0]       = self.start
        xi[-1]      = self.goal

        return xi[1:-1]

    def _init_xi_from_path(self, path: Sequence[np.ndarray]) -> torch.Tensor:
        """Initial path (start … near goal) → inner waypoints, shape (n, DOF)."""
        path = np.asarray(path, dtype=np.float64)
        path = np.vstack([self.start.numpy(), path[1:], self.goal.numpy()])   # pin exact start and goal

        length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        self.num_waypoints = self._num_waypoints(length)
        self.n = self.num_waypoints - 2

        xi = torch.as_tensor(resample_path(path, self.num_waypoints), dtype=DTYPE)
        return xi[1:-1]

    # ------------------------------------------------------------------
    # Kinematics helpers
    # ------------------------------------------------------------------

    def _full_traj(self, xi: torch.Tensor) -> torch.Tensor:
        """Prepend start and append goal to inner waypoints."""
        return torch.cat([self.start.unsqueeze(0), xi, self.goal.unsqueeze(0)], dim=0)

    def get_velocity(self, xi: torch.Tensor) -> torch.Tensor:
        """Central finite differences, shape (n, DOF)."""
        full = self._full_traj(xi)
        return (full[2:] - full[:-2]) / (2.0 * self.params.DYN_dT)

    def get_acceleration(self, xi: torch.Tensor) -> torch.Tensor:
        """Second-order finite differences, shape (n, DOF)."""
        full = self._full_traj(xi)
        return (full[2:] - 2.0 * full[1:-1] + full[:-2]) / (self.params.DYN_dT ** 2)

    # ------------------------------------------------------------------
    # Costs and gradients
    # ------------------------------------------------------------------

    def _esdf(self, pos: torch.Tensor) -> np.ndarray:
        """
        ESDF that continues outside the map: the query is clamped onto the map border and the
        distance to the border is subtracted, so leaving the map counts like entering an obstacle.
        (A plain query returns NaN there; that made the potential drop to 0 at the border and the
        finite-difference gradient explode, which threw waypoints several metres per step.)
        """
        pos     = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
        clamped = np.clip(pos, self.map_lo, self.map_hi)
        return self.env.esdf_query(clamped) - np.linalg.norm(pos - clamped, axis=1)

    def get_workspace_potential(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Smooth hinge potential c(d).
        Uses torch.where so the graph is preserved in every branch.
        """
        dist = torch.as_tensor(self._esdf(pos), dtype=DTYPE)

        eps  = self.epsilon

        inside   = -dist + 0.5 * eps                         # d < 0
        boundary = 0.5 * (dist - eps) ** 2 / eps             # 0 <= d <= eps
        outside  = torch.zeros(1, dtype=DTYPE)               # d > eps

        return torch.where(dist < 0, inside,
               torch.where(dist <= eps, boundary, outside))

    def get_grad_workspace_potential(self, pos: torch.Tensor, h: float = 1e-3) -> torch.Tensor:
        grad = torch.zeros_like(pos)
        for i in range(3):
            pos_plus  = pos.clone(); pos_plus[i]  += h
            pos_minus = pos.clone(); pos_minus[i] -= h
            grad[i]   = ((self.get_workspace_potential(pos_plus) -
                          self.get_workspace_potential(pos_minus)) / (2 * h)).squeeze()
        return grad

    def get_J_smooth(self, xi: torch.Tensor) -> torch.Tensor:
        """Smoothness cost, scalar."""
        q = xi.flatten()
        return 0.5 * q @ self.A @ q + q @ self.b + self.c

    def get_J_z(self, xi:torch.Tensor) -> torch.Tensor:
        """Height cost (only z value), averaged over the waypoints like its gradient"""
        q = xi[:, 2].flatten()

        return 0.5 * torch.sum((self.target_z - q) ** 2) / xi.shape[0]

    def get_J_obs(self, xi:torch.Tensor) -> torch.Tensor:
        """
        Obstacle cost (scalar):  Σ_k  c(x_k) · ‖ẋ_k‖
        Arc-length weighting (‖ẋ_k‖) makes the cost parametrisation-invariant.
        """
        vel   = self.get_velocity(xi)          # (n, DOF)
        J_obs = torch.zeros(1, dtype=DTYPE)

        for k in range(xi.shape[0]):
            c_k   = self.get_workspace_potential(xi[k])
            J_obs = J_obs + c_k * torch.norm(vel[k])

        return J_obs

    def get_cost(self, xi: torch.Tensor) -> float:
        """Weighted total cost that the optimisation minimises."""
        return (self.params.OPT_W_SMOOTH * self.get_J_smooth(xi) + self.params.OPT_W_OBS * self.get_J_obs(xi)
                + self.params.OPT_W_Z * self.get_J_z(xi)).item()

    def get_grad_J_smooth(self, xi: torch.Tensor) -> torch.Tensor:
        """Analytic gradient of J_smooth, shape (n*DOF,)."""
        return self.A @ xi.flatten() + self.b

    def get_grad_J_z(self, xi: torch.Tensor) -> torch.Tensor:
        """Gradient of the height cost, shape (n*DOF,)"""
        z = xi[:, 2]

        grad_z = (z - self.target_z)

        grad_z = grad_z / xi.shape[0]

        full_grad = torch.zeros_like(xi)
        full_grad[:, 2] = grad_z
        return full_grad.flatten()

    def get_grad_J_obstacle(self, xi: torch.Tensor) -> torch.Tensor:
        """
        Functional gradient of f_obs, eq. (4) in the CHOMP paper.
        Returns shape (n, DOF).
        """
        vel  = self.get_velocity(xi)
        acc  = self.get_acceleration(xi)
        grad = torch.zeros_like(xi)

        for k in range(xi.shape[0]):
            xd    = vel[k]
            xdd   = acc[k]
            speed = torch.norm(xd)

            xd_hat = xd / (speed + 1e-8)
            P      = torch.eye(self.DOF, dtype=DTYPE) - torch.outer(xd_hat, xd_hat)
            kappa  = P @ xdd / (speed ** 2 + 1e-8)

            c_k      = self.get_workspace_potential(xi[k])
            grad_c_k = self.get_grad_workspace_potential(xi[k])

            grad[k] = speed * (P @ grad_c_k - c_k * kappa)

        return grad

    # ------------------------------------------------------------------
    # Optimisation loop
    # ------------------------------------------------------------------

    def optimize(
        self,
        n_iter: int       = None,
        lam: float        = None,
        lam_decay: float  = None,
    ) -> torch.Tensor:
        """
        Standard CHOMP: covariant gradient descent on
        w_smooth * J_smooth + w_obs * J_obstacle + w_z * J_z.
        Stops after n_iter iterations, or earlier once the cost decreased by less than OPT_CONV_TOL
        (relative) over the last OPT_CONV_WINDOW iterations. The lowest-cost iterate is returned.
        Returns the full trajectory (start + inner waypoints + goal), shape (n+2, DOF).
        """
        n_iter       = n_iter       or self.params.OPT_N_ITER
        max_step     = self.params.OPT_MAX_STEP
        conv_tol     = self.params.OPT_CONV_TOL
        conv_window  = self.params.OPT_CONV_WINDOW
        lam          = lam          or self.params.OPT_LAM
        lam_decay    = lam_decay    or self.params.OPT_LAM_DECAY

        xi           = self.init_xi.clone()

        self.w_obs    = self.params.OPT_W_OBS
        self.w_z      = self.params.OPT_W_Z
        self.w_smooth = self.params.OPT_W_SMOOTH

        costs   = [self.get_cost(xi)]
        best_xi = xi
        tic     = time.time()

        for i in range(n_iter):
            grad_smooth   = self.get_grad_J_smooth(xi)
            grad_z        = self.get_grad_J_z(xi)
            grad_obstacle = self.get_grad_J_obstacle(xi).flatten()
            grad_U        = self.w_smooth * grad_smooth + self.w_obs * grad_obstacle + self.w_z * grad_z
            update        = (self.A_inv @ grad_U).reshape(self.n, self.DOF)

            step  = (lam_decay ** i) / lam * update
            # Trust region: A_inv spreads a local obstacle gradient over the whole trajectory and can
            # turn it into a jump of metres; no waypoint may move more than max_step per iteration
            step_len = step.norm(dim=1).max()
            if step_len > max_step:
                step = step * (max_step / step_len)
            xi = xi - step

            costs.append(self.get_cost(xi))
            if costs[-1] < min(costs[:-1]):
                best_xi = xi

            if (i+1) % 15 == 0:
                print(f"[CHOMP] iter={i}  J={costs[-1]:.4f}")

            # Convergence: (almost) no progress over the last conv_window iterations
            if i + 1 >= conv_window:
                old = costs[-1 - conv_window]
                if old - min(costs[-conv_window:]) < conv_tol * abs(old):
                    break

        n_done = i + 1
        print(f"[CHOMP] Optimization finished — {n_done} iterations"
              f"{' (converged)' if n_done < n_iter else ''}, J {costs[0]:.4f} -> {min(costs):.4f}, "
              f"{(time.time() - tic) / n_done * 1000:.1f} ms/iter")

        return self._full_traj(best_xi)
