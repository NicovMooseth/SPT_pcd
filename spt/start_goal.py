"""Start and goal selection: fixed from config.py, random in free space, or picked by hand on an ESDF slice."""

import numpy as np

MAX_TRIES = 100_000     # random draws per point before giving up

_floor_ceiling_cache = {}


def between_floor_and_ceiling(env, pts) -> np.ndarray:
    """
    Mask of the points that have an obstacle voxel (ESDF <= 0) somewhere below and somewhere above
    them in the same x/y column, i.e. that lie between floor and ceiling and not under the floor or
    above the ceiling. Works on any map (no fixed heights); the column masks are computed once per map.
    """
    if id(env) not in _floor_ceiling_cache:
        blocked = np.asarray(env.esdf_map) <= 0.0
        below   = np.logical_or.accumulate(blocked, axis=2)                  # obstacle at or below this voxel
        above   = np.logical_or.accumulate(blocked[:, :, ::-1], axis=2)[:, :, ::-1]
        _floor_ceiling_cache[id(env)] = below & above
    inside = _floor_ceiling_cache[id(env)]
    axes   = env.esdf_interpolator.grid
    idx    = [np.clip(np.round((pts[:, d] - axes[d][0]) / env.voxel_size).astype(int), 0, len(axes[d]) - 1)
              for d in range(3)]
    return inside[idx[0], idx[1], idx[2]]


def sample_free_point(env, rng, lo, hi, clearance, inside_only) -> np.ndarray:
    """Uniform random point in [lo, hi] with ESDF > clearance (inflation already included),
    between floor and ceiling if inside_only."""
    for _ in range(MAX_TRIES // 1000):
        pts  = lo + rng.random((1000, 3)) * (hi - lo)
        free = env.esdf_query(pts) > clearance                              # NaN (outside) -> False
        if inside_only:
            free &= between_floor_and_ceiling(env, pts)
        ok   = np.flatnonzero(free)
        if len(ok) > 0:
            return pts[ok[0]]
    raise RuntimeError(f"No free point with ESDF > {clearance} m"
                       f"{' between floor and ceiling' if inside_only else ''} found in {MAX_TRIES} draws.")


def sample_start_goal(env, params, seed=None) -> tuple[np.ndarray, np.ndarray]:
    """Random start and goal in free space, at least params.RANDOM_MIN_DISTANCE apart (MissionParams)."""
    rng    = np.random.default_rng(seed)
    lo, hi = (np.array(b, dtype=np.float64) for b in env.bounds)
    if params.RANDOM_Z_RANGE is not None:
        lo[2], hi[2] = max(lo[2], params.RANDOM_Z_RANGE[0]), min(hi[2], params.RANDOM_Z_RANGE[1])

    def sample():
        return sample_free_point(env, rng, lo, hi, params.RANDOM_CLEARANCE, params.RANDOM_INSIDE_ONLY)

    start = sample()
    for _ in range(1000):
        goal = sample()
        if np.linalg.norm(goal - start) >= params.RANDOM_MIN_DISTANCE:
            return start, goal
    raise RuntimeError(f"No goal at least {params.RANDOM_MIN_DISTANCE} m away from the start found.")


def pick_start_goal(env, axis="z") -> tuple[np.ndarray, np.ndarray]:
    """Start and goal clicked on an interactive ESDF slice (the first two saved points)."""
    print("Pick start and goal: left-click the slice, press 'Save point' twice, then close the window.")
    points = env.plot_slice(axis=axis)
    if len(points) < 2:
        raise SystemExit("Need two saved points (start, goal).")
    return points[0], points[1]


def get_start_goal(env, params, mode, seed=None) -> tuple[np.ndarray, np.ndarray]:
    """
    mode : "config"  MissionParams.START / GOAL
           "random"  random free points (MissionParams.RANDOM_*), reproducible with seed
           "pick"    clicked on an ESDF slice (MissionParams.PICK_AXIS)
    """
    if mode == "config":
        return (np.asarray(params.START, dtype=np.float64), np.asarray(params.GOAL, dtype=np.float64))
    if mode == "random":
        return sample_start_goal(env, params, seed)
    if mode == "pick":
        return pick_start_goal(env, params.PICK_AXIS)
    raise ValueError(f"Unknown start/goal mode {mode!r} (config, random or pick)")
