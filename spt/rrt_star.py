"""
RRT* path planner for the initial path guess, using the GFM field as an additional cost.

Edge cost = Euclidean length
          + lambda_gfm * ReLU(GFM_MAX - mean GFM of the edge end points)

Performance optimisations vs. the naive version
------------------------------------------------
1. Persistent KDTree  — rebuilt every REBUILD_EVERY new nodes instead of
                         every iteration; nearest + radius queries are O(log n).
2. Pre-allocated position matrix  — no repeated list-to-array conversion.
3. Batched ESDF queries           — all sample points along an edge are
                                     evaluated in one interpolator call.
4. Lower-bound pruning            — candidates sorted by (node.cost + dist);
                                     loop breaks as soon as the lower bound
                                     exceeds the current best, skipping the
                                     expensive GFM evaluation entirely.
5. GFM cache                      — the GFM (a LiDAR raycast) is evaluated
                                     once per tree node instead of per edge.
6. Squared-distance goal check    — avoids one sqrt per iteration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.spatial import KDTree

from .environment import EsdfEnvironment


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class RRTStarConfig:
    start:         np.ndarray
    goal:          np.ndarray
    max_iter:      int   = 5000
    step_size:     float = 1.0    # max extension length [m]
    goal_radius:   float = 2.5    # goal acceptance radius [m]
    esdf_margin:   float = 0.0    # min clearance from obstacles [m]
    lambda_gfm:    float = 1.0    # GFM penalty weight
    edge_samples:  int   = 5      # collision-check samples per edge
    search_radius: float = 2.0    # rewire neighbourhood radius [m]
    seed:          Optional[int] = 42
    GFM_MAX:       float = 25.0

    @classmethod
    def from_params(cls, params, start: np.ndarray, goal: np.ndarray) -> "RRTStarConfig":
        """Build from the RRTParams class in config.py."""
        return cls(
            start         = np.asarray(start, dtype=np.float64),
            goal          = np.asarray(goal, dtype=np.float64),
            max_iter      = params.MAX_ITER,
            step_size     = params.STEP_SIZE,
            goal_radius   = params.GOAL_RADIUS,
            esdf_margin   = params.ESDF_MARGIN,
            lambda_gfm    = params.LAMBDA_GFM,
            edge_samples  = params.EDGE_SAMPLES,
            search_radius = params.SEARCH_RADIUS,
            seed          = params.SEED,
            GFM_MAX       = params.GFM_MAX,
        )


# ---------------------------------------------------------------------------
# Internal node
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ("pos", "cost", "parent", "gfm")

    def __init__(self, pos: np.ndarray, cost: float = 0.0,
                 parent: Optional["_Node"] = None):
        self.pos    = pos
        self.cost   = cost
        self.parent = parent
        self.gfm    = None      # cached GFM value at pos


# ---------------------------------------------------------------------------
# RRT* planner
# ---------------------------------------------------------------------------

class RRTStar:
    """
    RRT* with a composite edge cost:

        c(a, b) = ||b - a||
                + lambda_gfm * ReLU(GFM_MAX - (gfm(a) + gfm(b)) / 2)
    """

    # How many new nodes trigger a KDTree rebuild
    _REBUILD_EVERY = 50

    def __init__(self, env: EsdfEnvironment, cfg: RRTStarConfig):
        self.env = env
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)

        self._lo, self._hi = env.bounds

        # Pre-compute goal radius squared to avoid sqrt in the hot loop
        self._goal_r2 = cfg.goal_radius ** 2

        for name, pt in [("start", cfg.start), ("goal", cfg.goal)]:
            if not self._in_bounds(pt):
                raise ValueError(f"{name} {pt} is outside the map bounds "
                                 f"{self._lo.round(2)} – {self._hi.round(2)}.")
            if env.esdf_query(pt)[0] < cfg.esdf_margin:
                raise ValueError(f"{name} {pt} is inside an obstacle (esdf < margin).")

    # ------------------------------------------------------------------ public

    def plan(self) -> Optional[List[np.ndarray]]:
        cfg = self.cfg
        t0  = time.perf_counter()

        root  = _Node(cfg.start.copy(), cost=0.0)
        nodes: List[_Node] = [root]
        best_goal_node: Optional[_Node] = None

        # --- persistent position matrix (doubles when full) ---
        _cap    = max(512, cfg.max_iter // 4)
        _pts    = np.empty((_cap, 3), dtype=np.float64)
        _pts[0] = root.pos

        # Build initial tree and track when it was last built
        _tree      = KDTree(_pts[:1])
        _tree_size = 1          # number of rows _tree was built from

        for i in range(cfg.max_iter):
            n = len(nodes)

            # 0. Rebuild KDTree if enough new nodes have been added (amortised)
            if n >= _tree_size + self._REBUILD_EVERY:
                _tree      = KDTree(_pts[:n])
                _tree_size = n

            # 1. Sample
            q_rand = self._sample()

            # 2. Nearest — O(log n) via persistent tree
            #    (nodes added since the last rebuild are not in the tree yet)
            dists, nn_idx = _tree.query(q_rand)
            q_near = nodes[nn_idx]

            # 3. Steer
            q_new = _Node(self._steer(q_near.pos, q_rand))

            # 4. Collision check on the extend edge (batched, no Python loop)
            if not self._edge_free(q_near.pos, q_new.pos):
                continue

            # 5. Radius neighbours — O(log n) via persistent tree
            nb_idxs    = _tree.query_ball_point(q_new.pos, r=cfg.search_radius)
            neighbours = [nodes[k] for k in nb_idxs]

            # 6. Choose best parent with lower-bound pruning
            #    Sort by (node.cost + Euclidean dist) — a lower bound on the
            #    true edge cost.  Break as soon as lb >= best_cost so we skip
            #    the expensive GFM evaluation for all remaining candidates.
            best_parent = q_near
            best_cost   = q_near.cost + self._edge_cost(q_near, q_new)

            candidates = sorted(
                neighbours,
                key=lambda nb: nb.cost + np.linalg.norm(nb.pos - q_new.pos),
            )
            for nb in candidates:
                lb = nb.cost + np.linalg.norm(nb.pos - q_new.pos)
                if lb >= best_cost:
                    break                   # all remaining candidates are worse
                c = nb.cost + self._edge_cost(nb, q_new)
                if c < best_cost and self._edge_free(nb.pos, q_new.pos):
                    best_parent = nb
                    best_cost   = c

            q_new.cost   = best_cost
            q_new.parent = best_parent
            nodes.append(q_new)

            # Grow position matrix if needed, then append
            if n >= _cap:
                _cap *= 2
                _pts  = np.resize(_pts, (_cap, 3))
            _pts[n] = q_new.pos

            # 7. Rewire with same lower-bound gate
            for nb in neighbours:
                lb = q_new.cost + np.linalg.norm(q_new.pos - nb.pos)
                if lb >= nb.cost:
                    continue
                c = q_new.cost + self._edge_cost(q_new, nb)
                if c < nb.cost and self._edge_free(q_new.pos, nb.pos):
                    nb.parent = q_new
                    nb.cost   = c

            # 8. Goal check (squared distance — no sqrt); the last segment to the goal must be free too
            diff = q_new.pos - cfg.goal
            if np.dot(diff, diff) <= self._goal_r2 and self._edge_free(q_new.pos, cfg.goal):
                if best_goal_node is None or q_new.cost < best_goal_node.cost:
                    best_goal_node = q_new

            if i % 500 == 0:
                dist_to_goal = np.sqrt(min(np.dot(nd.pos - cfg.goal, nd.pos - cfg.goal) for nd in nodes))
                cost_str     = f"{best_goal_node.cost:.3f}" if best_goal_node is not None else "-"
                print(f"[RRT*] iter={i:5d}  nodes={n:5d}  closest_to_goal={dist_to_goal:.3f}  "
                      f"best_cost={cost_str}")

        elapsed = time.perf_counter() - t0
        print(f"[RRT*] {cfg.max_iter} iterations in {elapsed:.2f}s  |  "
              f"tree size: {len(nodes)}")

        if best_goal_node is None:
            print("[RRT*] No path found.")
            return None

        path = self._extract_path(best_goal_node)
        print(f"[RRT*] Path found — {len(path)} waypoints, "
              f"cost = {best_goal_node.cost:.4f}")

        return path

    # ----------------------------------------------------------------- helpers

    def _in_bounds(self, p: np.ndarray) -> bool:
        return bool(np.all(p >= self._lo) and np.all(p <= self._hi))

    def _sample(self) -> np.ndarray:
        if self._rng.random() < 0.1:
            return self.cfg.goal.copy()
        return self._rng.uniform(self._lo, self._hi)

    def _steer(self, from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
        delta = to_pos - from_pos
        dist  = np.linalg.norm(delta)
        if dist <= self.cfg.step_size:
            return to_pos.copy()
        return from_pos + (delta / dist) * self.cfg.step_size

    # -------------------------------------------------------------------- cost

    def _gfm(self, node: _Node) -> float:
        if node.gfm is None:
            node.gfm = self.env.get_gfm_value(node.pos)
        return node.gfm

    def _gfm_relu_penalty(self, a: _Node, b: _Node) -> float:
        avg     = (self._gfm(a) + self._gfm(b)) / 2.0
        penalty = self.cfg.GFM_MAX - avg
        return float(np.maximum(penalty, 0.0))

    def _edge_cost(self, a: _Node, b: _Node) -> float:
        return (float(np.linalg.norm(b.pos - a.pos))
                + self.cfg.lambda_gfm * self._gfm_relu_penalty(a, b))

    # ---------------------------------------------------------------- collision

    def _edge_free(self, a: np.ndarray, b: np.ndarray) -> bool:
        # At least edge_samples, but never coarser than half a voxel so thin walls are not skipped
        K    = max(self.cfg.edge_samples, 3, int(np.ceil(np.linalg.norm(b - a) / (0.5 * self.env.voxel_size))))
        ts   = np.linspace(0.0, 1.0, K + 2)
        pts  = a + np.outer(ts, b - a)

        if np.any(pts < self._lo) or np.any(pts > self._hi):
            return False

        esdf_vals = self.env.esdf_query(pts)
        valid     = ~np.isnan(esdf_vals)

        # Only check observed voxels; treat unobserved as free
        if not np.any(valid):
            return True
        return bool(np.all(esdf_vals[valid] >= self.cfg.esdf_margin))

    # -------------------------------------------------------------------- path

    @staticmethod
    def _extract_path(goal_node: _Node) -> List[np.ndarray]:
        path, node = [], goal_node
        while node is not None:
            path.append(node.pos)
            node = node.parent
        return path[::-1]
