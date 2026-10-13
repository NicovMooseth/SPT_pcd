# SPT — ESDF-based Trajectory Optimization from a Point Cloud

Plans a smooth, collision-free 3D trajectory through a point cloud map (`.pcd`). The ESDF is computed
directly from the cloud, so no ROS, voxblox or bag is needed.

```
.pcd  ──►  ESDF (EDT)  ──►  start / goal  ──►  RRT* initial path  ──►  CHOMP optimization  ──►  output/
                            (random, picked                            (smoothness + obstacle     trajectory.csv
                             by hand or fixed)                          + height cost)            top_view.png
```

1. **ESDF**: aligns the cloud to gravity, voxelizes it and runs a Euclidean distance transform
   (see [ESDF from the point cloud](#esdf-from-the-point-cloud)).
2. **Start / goal**: random points in free space, clicked on an ESDF slice, or fixed in `config.py`.
3. **RRT\***: finds a collision-free initial path. The edge cost is the path length plus an optional
   penalty on low GFM (geometric feature metric) values, computed by raycasting a simulated LiDAR into the ESDF.
4. **CHOMP**: covariant gradient descent on the trajectory, starting from the RRT\* path.
5. **Output**: the trajectory is checked for collisions and saved as CSV, together with a top-view plot.

Nothing is shown on screen while the pipeline runs. The only window is the ESDF slice for picking start and
goal by hand (`--start-goal pick`), and it opens before planning starts.

## Repository structure

```
├── config.py              # ALL pipeline parameters (pcd path, ESDF, start/goal, LiDAR, RRT*, CHOMP, output)
├── main.py                # pipeline entry point
├── requirements.txt
├── spt/
│   ├── environment.py     # ESDF from a .pcd, ESDF queries, GFM, start/goal picking window
│   ├── start_goal.py      # start/goal selection: random, picked by hand, or from config.py
│   ├── rrt_star.py        # RRT* initial path planner
│   ├── chomp.py           # CHOMP trajectory optimizer + feasibility check
│   └── plot_top_view.py   # top-view plot of the result (saved as .png)
├── data/                  # put point clouds here (ignored by git)
└── output/                # results (created at runtime, ignored by git)
```

## Installation

Requires **Python ≥ 3.10**.

With [uv](https://docs.astral.sh/uv/):

```bash
uv venv .venv --python 3.10
uv pip install --python .venv -r requirements.txt --index-strategy unsafe-best-match
source .venv/bin/activate
```

Or with an existing Python ≥ 3.10:

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs the CPU-only build of PyTorch.

## Usage

```bash
python main.py --pcd path/to/cloud.pcd                          # start/goal mode from config.py (default: random)
python main.py --pcd path/to/cloud.pcd --start-goal random --seed 3   # reproducible random start/goal
python main.py --pcd path/to/cloud.pcd --start-goal pick        # click start and goal on an ESDF slice
python main.py --pcd path/to/cloud.pcd --start-goal config      # MissionParams.START / GOAL
python main.py --pcd path/to/cloud.pcd --out results/run1       # other output directory
```

Without `--pcd`, the cloud is read from `EnvParams.PCD_PATH` (default `data/map.pcd`).

### Start and goal

| Mode     | How                                                                                           |
|----------|-----------------------------------------------------------------------------------------------|
| `random` | Uniform random points with at least `RANDOM_CLEARANCE` from obstacles, at least `RANDOM_MIN_DISTANCE` apart. With `RANDOM_INSIDE_ONLY`, only points between floor and ceiling. `--seed` makes them reproducible. |
| `pick`   | An ESDF slice opens. Move the slider to the height, left-click a point, press **Save point**; repeat for the goal, then close the window. The first saved point is the start, the second the goal. |
| `config` | `MissionParams.START` / `GOAL`. Coordinates are in the aligned map frame (floor at z = 0), so every map needs its own values. |

Start and goal must be inside the map bounds and in free space. The bounds are printed when the map loads.

### Output

| File                    | Contents                                                                         |
|-------------------------|----------------------------------------------------------------------------------|
| `output/trajectory.csv` | `x,y,z` per waypoint, start to goal, in the aligned map frame                     |
| `output/top_view.png`   | top view of the trajectory and the RRT\* initial path on the clearance map         |
| `output/T_global.txt`   | 4×4 transform from the `.pcd` frame into the aligned map frame                    |

Before saving, every segment of the trajectory is sampled at half the voxel size and checked against the
(inflated) ESDF. An infeasible trajectory is never saved: `main.py` exits with code 1, removes an old
`trajectory.csv`, and marks the colliding samples red in `top_view.png`.

To get the trajectory in the original `.pcd` frame, apply the inverse of `T_global.txt`:

```python
import numpy as np
T   = np.loadtxt("output/T_global.txt")
xyz = np.loadtxt("output/trajectory.csv", delimiter=",", skiprows=1)
xyz_pcd = (np.linalg.inv(T) @ np.c_[xyz, np.ones(len(xyz))].T).T[:, :3]
```

`EnvParams.OBSTACLE_INFLATION` (hard, robot radius + margin) and `PlannerParams.OPT_EPSILON`
(soft cost margin of CHOMP) add up: the CHOMP cost starts at inflation + epsilon from a surface,
the feasibility check only enforces the inflation.

With `RRTParams.USE_RRT_INIT = False`, CHOMP starts from a straight line instead. With `RRT_RETRY = True`,
a colliding result is then retried once from an RRT\* initial path.

## ESDF from the point cloud

`spt/environment.py`:

1. aligns the cloud to gravity (RANSAC ground plane, refined so walls and floors are vertical/horizontal),
   puts the floor at z = 0 and optionally rotates the dominant wall direction onto the x-axis,
2. removes height layers with too few points (outliers above the ceiling or below the floor),
3. voxelizes the points (`EnvParams.VOXEL_SIZE`),
4. marks unexplored regions as obstacles: x/y regions of `EnvParams.UNKNOWN_REGION_SIZE` without a
   single point over their whole height (no floor, so most likely not scanned yet),
5. runs a signed Euclidean distance transform on occupied + unexplored voxels.

Steps 1, 2 and 4 can be switched off in `config.py` with `EnvParams.ROTATION_ALIGNMENT`,
`EnvParams.POINT_REMOVAL` and `EnvParams.UNKNOWN_AS_OBSTACLE`.
Outside the grid there is no ESDF value: RRT\* and the feasibility check treat it as blocked, and the
CHOMP cost rises past the border as if it were an obstacle.
Unexplored columns are not seen by the simulated LiDAR (GFM).

## Parameters

Every parameter lives in `config.py`:

| Class           | Contents                                                                         |
|-----------------|----------------------------------------------------------------------------------|
| `EnvParams`     | pcd path, obstacle inflation, ESDF voxel size, alignment, height filter, unexplored region size |
| `MissionParams` | start/goal mode, fixed start and goal, picking axis, random sampling settings     |
| `LidarParams`   | simulated LiDAR (FOV, number of rays, range) used for the GFM                     |
| `GFMParams`     | surface threshold for the GFM                                                     |
| `RRTParams`     | RRT\* iterations, step size, goal radius, GFM weight, seed, …                     |
| `PlannerParams` | CHOMP waypoint density, step size, convergence, safety distance, cost weights     |
| `OutputParams`  | output directory                                                                  |
