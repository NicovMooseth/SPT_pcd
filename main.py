"""
SPT trajectory optimization pipeline.

    .pcd  ->  ESDF  ->  start / goal  ->  RRT* initial path  ->  CHOMP optimization
          ->  output/trajectory.csv + output/top_view.png

Usage:
    python main.py --pcd path/to/cloud.pcd                       # parameters from config.py
    python main.py --pcd path/to/cloud.pcd --start-goal random --seed 3
    python main.py --pcd path/to/cloud.pcd --start-goal pick     # click start and goal on an ESDF slice
    python main.py --pcd path/to/cloud.pcd --start-goal config   # MissionParams.START / GOAL
"""

import argparse
import os
import time

import numpy as np

import config
from spt.chomp import CHOMPPlanner, find_collisions
from spt.environment import EsdfEnvironment
from spt.plot_top_view import plot_top_view
from spt.rrt_star import RRTStar, RRTStarConfig
from spt.start_goal import get_start_goal

# Relative paths in config.py are resolved from the repository root
ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RRT* + CHOMP trajectory optimization on an ESDF computed from a .pcd")
    parser.add_argument("--pcd", default=os.path.join(ROOT, config.EnvParams.PCD_PATH),
                        help=f"point cloud (.pcd/.ply/...) (default: {config.EnvParams.PCD_PATH})")
    parser.add_argument("--start-goal", choices=["random", "pick", "config"],
                        default=config.MissionParams.START_GOAL_MODE,
                        help=f"how start and goal are chosen (default: {config.MissionParams.START_GOAL_MODE})")
    parser.add_argument("--seed", type=int, default=config.MissionParams.RANDOM_SEED,
                        help="random seed for --start-goal random (default: MissionParams.RANDOM_SEED)")
    parser.add_argument("--out", default=os.path.join(ROOT, config.OutputParams.OUTPUT_DIR),
                        help=f"output directory (default: {config.OutputParams.OUTPUT_DIR})")
    return parser.parse_args()


def load_env(pcd_path):
    """ESDF environment from a point cloud."""
    tic = time.time()
    print(f"Computing ESDF from {pcd_path} …")
    env = EsdfEnvironment(
        pcd_path     = pcd_path,
        env_params   = config.EnvParams,
        lidar_params = config.LidarParams,
        gfm_params   = config.GFMParams,
    )
    lo, hi = env.bounds
    print(f"Map loaded in {time.time() - tic:.1f}s — voxel size {env.voxel_size:.2f} m, "
          f"bounds {lo.round(1)} – {hi.round(1)}")
    return env


def plan(env, start, goal, rrt_retry=None):
    """
    RRT* (optional) -> CHOMP -> feasibility check, with one RRT* retry if enabled.

    Returns a dict with
        trajectory : (N, 3) final trajectory
        collisions : (K, 3) samples inside an (inflated) obstacle, empty if feasible
        init_path  : initial path of the final CHOMP run (None = straight line)
        status     : "chomp"      feasible after the first CHOMP run
                     "rrt_retry"  feasible only after the retry from an RRT* path
                     "infeasible" no collision-free trajectory found
    """
    rrt_retry = config.RRTParams.RRT_RETRY if rrt_retry is None else rrt_retry

    def plan_rrt():
        rrt_cfg = RRTStarConfig.from_params(config.RRTParams, start, goal)
        path    = RRTStar(env, rrt_cfg).plan()
        if path is None:
            print("RRT* found no path — falling back to a straight-line initialization.")
        return None if path is None else np.asarray(path)

    def plan_chomp(init_path):
        trajectory = CHOMPPlanner(env, start, goal, config.PlannerParams, init_path=init_path).optimize().numpy()
        # Check the segments between the waypoints too, at half the voxel size
        return trajectory, find_collisions(env, trajectory, step=0.5 * env.voxel_size)

    init_path = plan_rrt() if config.RRTParams.USE_RRT_INIT else None
    trajectory, collisions = plan_chomp(init_path)
    status = "chomp" if len(collisions) == 0 else "infeasible"

    if status == "infeasible" and init_path is None and rrt_retry:
        # CHOMP only deforms its initial path locally: a straight line through a wall usually stays there
        print(f"Trajectory crosses obstacles ({len(collisions)} samples) — retrying from an RRT* initial path.")
        rrt_path = plan_rrt()
        if rrt_path is not None:
            init_path = rrt_path
            trajectory, collisions = plan_chomp(init_path)
            status = "rrt_retry" if len(collisions) == 0 else "infeasible"

    clearance = env.esdf_query(trajectory)
    print(f"Trajectory: {len(trajectory)} waypoints, "
          f"length {np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)):.2f} m, "
          f"min clearance {np.nanmin(clearance):.2f} m (beyond the inflation of {env.inflation:.2f} m)")
    return dict(trajectory=trajectory, collisions=collisions, init_path=init_path, status=status)


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Environment: ESDF from the point cloud
    # ------------------------------------------------------------------
    env = load_env(args.pcd)

    # ------------------------------------------------------------------
    # 2. Start / goal
    # ------------------------------------------------------------------
    start, goal = get_start_goal(env, config.MissionParams, args.start_goal, seed=args.seed)
    print(f"Start {start.round(2)}  →  Goal {goal.round(2)}  ({args.start_goal})")
    for name, pos in (("Start", start), ("Goal", goal)):
        if not env.esdf_query(pos)[0] >= 0.0:
            print(f"Warning: {name} is inside an (inflated) obstacle or outside the map — "
                  f"no feasible trajectory can exist.")

    # ------------------------------------------------------------------
    # 3. Initial path (RRT*) + 4. trajectory optimization (CHOMP) + feasibility check
    # ------------------------------------------------------------------
    result     = plan(env, start, goal)
    trajectory = result["trajectory"]
    collisions = result["collisions"]

    # ------------------------------------------------------------------
    # 5. Output
    # ------------------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "trajectory.csv")
    if len(collisions) == 0:
        np.savetxt(out_path, trajectory, delimiter=",", header="x,y,z", comments="", fmt="%.4f")
        print(f"Trajectory is collision-free. Saved trajectory to {out_path}")
    else:
        # Never hand out an infeasible trajectory: remove an old result so it cannot be mistaken for this one
        if os.path.exists(out_path):
            os.remove(out_path)
        print(f"INFEASIBLE: {len(collisions)} samples along the trajectory are inside an (inflated) obstacle "
              f"or outside the map. No trajectory saved.")

    # The map lives in the aligned frame: the inverse of this maps the trajectory back to the .pcd frame
    np.savetxt(os.path.join(args.out, "T_global.txt"), env.T_global, fmt="%.6f")

    plot_top_view(env, trajectory, save_path=os.path.join(args.out, "top_view.png"),
                  init_path=result["init_path"], collisions=collisions)

    if len(collisions) > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
