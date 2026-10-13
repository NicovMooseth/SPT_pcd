"""Top-down (x-y) view of the optimized trajectory on top of the ESDF."""

import matplotlib

matplotlib.use("Agg")       # the figure is only saved, never shown

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

MAX_LAYERS = 40         # max ESDF z-layers sampled over the height range of the trajectory


def plot_top_view(env, trajectory, save_path, init_path=None, collisions=None, title=None):
    """
    Plots the trajectory from above. The background is the clearance at the heights the
    trajectory flies at: for every x/y cell the smallest ESDF value between the lowest and
    the highest waypoint (red = close to an obstacle). Obstacles at these heights are black.

    Args:
        env        : environment with bounds, voxel_size and esdf_query
        trajectory : (N, 3) optimized trajectory (start ... goal)
        save_path  : file the figure is saved to (e.g. .png)
        init_path  : optional (M, 3) initial path (RRT*), drawn dashed
        collisions : optional (K, 3) samples of the trajectory inside an (inflated) obstacle
        title      : optional first title line (e.g. test case and status)
    """
    traj = np.asarray(trajectory, dtype=np.float64)
    lo, hi = env.bounds
    vs = env.voxel_size

    # --- Clearance map: minimum ESDF over the height range of the trajectory ---
    xs = np.arange(lo[0], hi[0], vs) + 0.5 * vs
    ys = np.arange(lo[1], hi[1], vs) + 0.5 * vs
    n_layers = int(np.clip((traj[:, 2].max() - traj[:, 2].min()) / vs, 0, MAX_LAYERS - 1)) + 1
    X, Y = np.meshgrid(xs, ys)      # (ny, nx): rows are y, as imshow expects
    clearance = np.full(X.shape, np.inf)
    for z in np.linspace(traj[:, 2].min(), traj[:, 2].max(), n_layers):
        d = env.esdf_query(np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], axis=-1)).reshape(X.shape)
        clearance = np.fmin(clearance, d)       # fmin ignores NaN (outside the grid)
    clearance[np.isinf(clearance)] = np.nan
    extent = [lo[0], lo[0] + len(xs) * vs, lo[1], lo[1] + len(ys) * vs]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(clearance, origin="lower", extent=extent, cmap="jet_r", vmin=0.0)
    obstacles = clearance < 0.5 * vs
    ax.imshow(np.ma.masked_where(~obstacles, obstacles), origin="lower", extent=extent,
              cmap="gray_r", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Clearance at trajectory height [m]"
                 + (f" (obstacles inflated by {env.inflation:.2f} m)" if getattr(env, "inflation", 0) else ""))

    # --- Paths ---
    outline = [pe.Stroke(linewidth=6, foreground="black"), pe.Normal()]
    if init_path is not None:
        init = np.asarray(init_path, dtype=np.float64)
        ax.plot(init[:, 0], init[:, 1], "--", color="white", lw=2, label="Initial path (RRT*)",
                path_effects=[pe.Stroke(linewidth=4, foreground="black"), pe.Normal()])
    ax.plot(traj[:, 0], traj[:, 1], "-", color="magenta", lw=3, label="CHOMP trajectory",
            path_effects=outline, zorder=3)
    ax.plot(traj[1:-1, 0], traj[1:-1, 1], "o", color="magenta", ms=3, mec="black", mew=0.5, zorder=4)

    # Direction arrows along the trajectory
    for i in np.linspace(0, len(traj) - 2, 6, dtype=int)[1:-1]:
        ax.annotate("", xy=traj[i + 1, :2], xytext=traj[i, :2], zorder=5,
                    arrowprops=dict(arrowstyle="-|>", color="magenta", lw=0, mutation_scale=22,
                                    path_effects=[pe.Stroke(linewidth=2, foreground="black"), pe.Normal()]))

    # Start, goal and the point with the smallest clearance
    traj_clear = env.esdf_query(traj)
    k = np.nanargmin(traj_clear)
    ax.plot(*traj[0, :2], "o", color="white", ms=14, mec="black", mew=2, zorder=6, label="Start")
    ax.plot(*traj[-1, :2], "*", color="white", ms=22, mec="black", mew=2, zorder=6, label="Goal")
    ax.plot(*traj[k, :2], "X", color="yellow", ms=13, mec="black", mew=1.5, zorder=5,
            label=f"Min. clearance {traj_clear[k]:.2f} m")

    if collisions is not None and len(collisions) > 0:
        ax.plot(collisions[:, 0], collisions[:, 1], "o", color="red", ms=7, mec="white", mew=1.5,
                zorder=7, label=f"Collision ({len(collisions)} samples)")

    length = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
    status = "INFEASIBLE" if collisions is not None and len(collisions) > 0 else "collision-free"
    ax.set_title((f"{title}\n" if title else "")
                 + f"CHOMP trajectory (top view, {status}): {len(traj)} waypoints, length {length:.2f} m, "
                 f"z = {traj[:, 2].min():.2f} … {traj[:, 2].max():.2f} m")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()

    fig.savefig(save_path, dpi=200)
    print(f"Saved top view to {save_path}")
    plt.close(fig)
