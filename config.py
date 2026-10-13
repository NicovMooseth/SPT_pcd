"""
Central parameter file for the SPT trajectory optimization pipeline.

Pipeline:  .pcd  ->  ESDF (EDT)  ->  start / goal  ->  RRT* initial path  ->  CHOMP optimization

Every tunable parameter used anywhere in the pipeline lives in this file.
"""

import numpy as np


class EnvParams:
    # --- Input --------------------------------------------------------
    PCD_PATH = "data/map.pcd"           # point cloud

    # --- Obstacle inflation -------------------------------------------
    OBSTACLE_INFLATION = 0.15           # [m] robot radius + safety margin: everything closer than this to an
                                        # obstacle counts as obstacle (ESDF is shifted by this for planning,
                                        # RRT* collision checks and the final feasibility check)

    # --- Preprocessing steps (on/off) ---------------------------------
    POINT_REMOVAL       = True          # remove points in sparse height layers (HEIGHT_BIN, MIN_LAYER_FRACTION)
    ROTATION_ALIGNMENT  = True          # rotate the cloud so the floor is horizontal and at z = 0
    UNKNOWN_AS_OBSTACLE = True          # treat unexplored columns as obstacles (UNKNOWN_REGION_SIZE)

    # --- ESDF ---------------------------------------------------------
    VOXEL_SIZE         = 0.05           # ESDF voxel size [m]
    ALIGN_YAW          = True           # with ROTATION_ALIGNMENT: also rotate the dominant wall direction onto the x-axis
    HEIGHT_BIN         = 0.5            # height layer size for the outlier filter [m]
    MIN_LAYER_FRACTION = 0.01           # height layers with < this fraction of all points are removed
    GRID_PADDING       = 2              # free voxels added around the cloud on every side
    UNKNOWN_REGION_SIZE = 0.25          # x/y regions of this size [m] without any point in their whole column
                                        # (no floor -> unexplored) are treated as obstacles


class MissionParams:
    # How start and goal are chosen; override with `python main.py --start-goal <mode>`
    #   "random"  random points in free space (RANDOM_* below)
    #   "pick"    click them on an interactive ESDF slice (PICK_AXIS)
    #   "config"  START / GOAL below
    START_GOAL_MODE = "random"

    # "config": start and goal position in the map frame (aligned frame, floor at z = 0) [m]
    # (free space in office.pcd; every map needs its own values)
    START = np.array([-0.06, 0.93, 2.41])
    GOAL  = np.array([8.11, -2.32, 1.85])

    # "pick": slice axis
    PICK_AXIS = "z"

    # "random"
    RANDOM_SEED         = None          # None = new points every run; override with `--seed <int>`
    RANDOM_CLEARANCE    = 0.5           # [m] start and goal need ESDF > this (beyond OBSTACLE_INFLATION)
    RANDOM_MIN_DISTANCE = 2.0           # [m] minimum straight-line distance between start and goal
    RANDOM_Z_RANGE      = None          # (z_min, z_max) [m] for start and goal; None = full map height
    RANDOM_INSIDE_ONLY  = True          # start and goal need an obstacle below (floor) and above (ceiling)
                                        # in their x/y column: no points under the floor or above the ceiling


class LidarParams:
    # Simulated LiDAR used to evaluate the GFM (geometric feature metric) at a position
    HORIZONTAL_FOV_MIN:    float = 0
    HORIZONTAL_FOV_MAX:    float = 360
    HORIZONTAL_RESOLUTION: int   = 6    # number of rays
    VERTICAL_FOV_MIN:      float = -7
    VERTICAL_FOV_MAX:      float = 52
    VERTICAL_RESOLUTION:   int   = 3    # number of rays

    MAX_RAY_LENGTH: float = 30.0        # [m]
    MAX_RAY_STEPS:  int   = 20          # sphere-tracing steps per ray


class GFMParams:
    SURFACE_THRESHOLD = 1.5             # |esdf| < SURFACE_THRESHOLD * voxel_size counts as surface


class RRTParams:
    USE_RRT_INIT:  bool  = True         # False: CHOMP starts from a (noisy) straight line
    RRT_RETRY:     bool  = True         # True: if the CHOMP result collides, retry once from an RRT* path
                                        # (only matters with USE_RRT_INIT = False)
    MAX_ITER:      int   = 2000
    STEP_SIZE:     float = 1.0          # max extension length [m] (0.3 was too short: no path through doors in 5000 iterations)
    GOAL_RADIUS:   float = 0.1          # goal acceptance radius [m]
    ESDF_MARGIN:   float = 0.1          # min clearance from obstacles [m]
    LAMBDA_GFM:    float = 0.0          # GFM penalty weight
    GFM_MAX:       float = 25.0         # GFM values below this are penalised
    EDGE_SAMPLES:  int   = 5            # min collision-check samples per edge (at least every half voxel)
    SEARCH_RADIUS: float = 1.5          # rewire neighbourhood radius [m] (should be > STEP_SIZE)
    SEED                 = 42


class PlannerParams:
    # --- Fixed (architectural) ----------------------------------------
    DOF: int = 3                        # spatial dimensions (x, y, z)

    # --- Initial Path -------------------------------------------------
    PAT_NUM_WAYPOINTS_PER_M = 3         # waypoint density of the initial trajectory
    PAT_NOISE_INIT = 0.05               # noise on each straight-line initial waypoint

    # --- Velocity and acceleration ------------------------------------
    DYN_dT = 0.1

    # --- Optimisation -------------------------------------------------
    OPT_N_ITER: int        = 100        # max gradient steps
    OPT_LAM: float         = 15.0       # step size denominator (λ); ↑ = smaller steps
    OPT_LAM_DECAY: float   = 0.98       # λ multiplied by this each step → step shrinks
    OPT_MAX_STEP: float    = 0.1        # [m] max waypoint displacement per iteration (trust region)
    OPT_CONV_TOL: float    = 1e-3       # stop early when the cost dropped by less than this (relative) ...
    OPT_CONV_WINDOW: int   = 5          # ... over this many iterations (OPT_N_ITER stays the maximum)

    OPT_EPSILON: float     = 0.5        # [m] soft margin: CHOMP obstacle cost starts this far outside the
                                        # (inflated) obstacles; not enforced, only penalised

    OPT_W_OBS: float       = 0.35       # Optimization Weights
    OPT_W_Z: float         = 0.2
    OPT_W_SMOOTH: float    = 0.25


class OutputParams:
    OUTPUT_DIR = "output"               # trajectory.csv, top_view.png and T_global.txt are saved here
