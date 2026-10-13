"""ESDF environment computed from a point cloud (.pcd) via a Euclidean distance transform."""

from pathlib import Path

import numpy as np
import open3d as o3d
import pyvista as pv
import torch
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree


def _rotation_from_to(a, b) -> np.ndarray:
    """Rotation matrix that rotates unit vector a onto unit vector b."""
    v = np.cross(a, b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + a @ b)


def _floor_height(z, horizontal, layer=0.1, min_share=0.3):
    """
    Height of the floor: the lowest large horizontal plane. Horizontal points are binned into
    height layers; layers with at least min_share of the fullest layer count as large planes
    (floor, ceilings, tables, ...). The largest plane is often the ceiling, so the lowest one is used.
    Returns the floor height (median of its points) and the number of large layers.
    """
    z_h    = z[horizontal]
    counts, edges = np.histogram(z_h, bins=np.arange(z_h.min(), z_h.max() + 2 * layer, layer))
    large  = np.flatnonzero(counts >= min_share * counts.max())
    lowest = large[0]
    # Points of the lowest large layer and its neighbours (a plane can straddle a layer edge)
    floor  = (z_h >= edges[lowest] - layer) & (z_h < edges[lowest + 1] + layer)
    return np.median(z_h[floor]), len(large)


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class EsdfEnvironment:
    """ESDF environment computed directly from a point cloud (.pcd) via a Euclidean distance transform."""

    def __init__(self, pcd_path: str, env_params, lidar_params, gfm_params):
        self.pcd_path     = pcd_path
        self.env_params   = env_params
        self.lidar_params = lidar_params
        self.gfm_params   = gfm_params
        self.inflation    = getattr(env_params, "OBSTACLE_INFLATION", 0.0)

        self._load_from_pcd()

        # Axes of the voxel grid: ESDF values at the voxel corners, normals at the voxel centres
        self.voxel_axes  = [self.grid_origin[i] + np.arange(n) * self.voxel_size
                            for i, n in enumerate(self.grid_size)]
        self.center_axes = [self.grid_origin[i] + (np.arange(n) + 0.5) * self.voxel_size
                            for i, n in enumerate(self.grid_size)]
        self.esdf_interpolator = RegularGridInterpolator(self.voxel_axes, self.esdf_map, method="linear",
                                                         bounds_error=False, fill_value=np.nan)

        self._load_lidar_directions()
        self._build_surface_pts()
        self._build_normal_interpolator()

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper) corner of the ESDF grid in metres."""
        lo = np.asarray(self.grid_origin, dtype=np.float64)
        return lo, lo + np.asarray(self.grid_size) * self.voxel_size

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_from_pcd(self):
        """
        Reads the point cloud, aligns it to a gravity-aligned frame (floor at z = 0),
        removes sparse height outliers, voxelizes it and computes the ESDF with an EDT.
        Each preprocessing step can be switched off in EnvParams
        (ROTATION_ALIGNMENT, POINT_REMOVAL, UNKNOWN_AS_OBSTACLE).

        The resulting map lives in the aligned frame; self.T_global (4x4) maps
        points from the original .pcd frame into it.
        """
        p   = self.env_params
        pcd = o3d.io.read_point_cloud(str(Path(self.pcd_path).expanduser()))
        if pcd.is_empty():
            raise RuntimeError(f"No points read from {self.pcd_path}")

        self.T_global = np.eye(4)
        if p.ROTATION_ALIGNMENT:
            pcd, self.T_global = self._align_to_global_frame(pcd, p.ALIGN_YAW)

        points = np.asarray(pcd.points)
        if p.POINT_REMOVAL:
            points = points[self._remove_height_outliers(points, p.HEIGHT_BIN, p.MIN_LAYER_FRACTION)]

        self._compute_esdf(points, p.VOXEL_SIZE)

    @staticmethod
    def _align_to_global_frame(pcd, align_yaw=True):
        """
        Aligns the point cloud to a global frame from geometry only (no IMU/GPS):
          z: gravity. Normal of the ground plane (RANSAC), refined so that as many
             surfaces as possible are horizontal (floor, ceilings) or vertical (walls).
          z = 0: floor height (lowest large horizontal plane; the largest one is often the ceiling).
          x/y: dominant wall direction (optional). North cannot be determined without GPS.
        Returns the aligned point cloud and the 4x4 transformation.
        """
        z_axis = np.array([0.0, 0.0, 1.0])
        ds = pcd.voxel_down_sample(0.2)
        ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.6, max_nn=30))
        normals = np.asarray(ds.normals)

        def score(up):
            d = np.abs(normals @ up)
            return np.mean(d > 0.95) + np.mean(d < 0.1)   # share of horizontal + vertical surfaces

        # 1. Ground plane as initial "up"
        plane, _ = ds.segment_plane(distance_threshold=0.15, ransac_n=3, num_iterations=3000)
        up = np.asarray(plane[:3]) / np.linalg.norm(plane[:3])
        up *= np.sign(up[2])

        # 2. Refine within a 5° cone around the initial guess
        R_cone = _rotation_from_to(z_axis, up)
        candidates = [up] + [R_cone @ [np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), np.cos(t)]
                             for t in np.radians(np.arange(0.5, 5.01, 0.5))
                             for a in np.radians(np.arange(0, 360, 10))]
        best_up = max(candidates, key=score)
        R = _rotation_from_to(best_up, z_axis)

        # 3. Yaw: wall normals (horizontal after step 2), angle modulo 90°
        if align_yaw:
            n_rot  = normals @ R.T
            walls  = np.abs(n_rot[:, 2]) < 0.1
            angles = np.degrees(np.arctan2(n_rot[walls, 1], n_rot[walls, 0])) % 90
            hist, edges = np.histogram(angles, bins=90, range=(0, 90))
            yaw = edges[hist.argmax()] + 0.5
            yaw = yaw - 90 if yaw > 45 else yaw
            c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
            R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) @ R

        # 4. Floor at z = 0: lowest large horizontal plane
        ground, n_planes = _floor_height(np.asarray(ds.points) @ R[2], np.abs(normals @ R[2]) > 0.95)

        T = np.eye(4)
        T[:3, :3] = R
        T[2, 3]   = -ground
        return o3d.geometry.PointCloud(pcd).transform(T), T

    @staticmethod
    def _remove_height_outliers(points, bin_size, min_fraction) -> np.ndarray:
        """Mask that drops all points in height layers holding < min_fraction of all points."""
        layers, layer_of_pt, counts = np.unique(np.floor(points[:, 2] / bin_size),
                                                return_inverse=True, return_counts=True)
        kept_layers = layers[counts >= min_fraction * len(points)]
        keep = counts[layer_of_pt.ravel()] >= min_fraction * len(points)
        return keep

    def _compute_esdf(self, points, voxel_size):
        """
        Voxelizes the points (occupied = at least one point in the voxel) and computes,
        for every voxel, the signed Euclidean distance to the nearest obstacle boundary in metres.
        Sets voxel_size, grid_origin (corner of voxel 0), grid_size, esdf_map and unknown_columns.
        """
        pad              = self.env_params.GRID_PADDING
        self.voxel_size  = voxel_size
        self.grid_origin = points.min(axis=0) - pad * voxel_size

        coords = ((points - self.grid_origin) / voxel_size).astype(int)
        self.grid_size = list(coords.max(axis=0) + 1 + pad)

        occupied = np.zeros(self.grid_size, dtype=bool)
        occupied[tuple(coords.T)] = True

        # Unexplored columns (no point over the whole height) count as obstacles in the ESDF
        if self.env_params.UNKNOWN_AS_OBSTACLE:
            self.unknown_columns = self._find_unknown_columns(occupied, voxel_size,
                                                              self.env_params.UNKNOWN_REGION_SIZE)
        else:
            self.unknown_columns = np.zeros(occupied.shape[:2], dtype=bool)

        # Signed EDT: distance of every free voxel to the nearest occupied or unexplored one (> 0) and
        # minus the distance of every blocked voxel to the nearest free one (< 0). The sign matters for
        # CHOMP: an unsigned map is flat (0) inside obstacles, so there is no gradient that pushes out.
        blocked       = occupied | self.unknown_columns[:, :, None]
        self.esdf_map = ((distance_transform_edt(~blocked) - distance_transform_edt(blocked))
                         * voxel_size).astype(np.float32)
        print(f"ESDF grid {tuple(self.grid_size)} ({occupied.size:,} voxels, "
              f"{occupied.sum():,} occupied) at {voxel_size:.2f} m")

    @staticmethod
    def _find_unknown_columns(occupied, voxel_size, region_size) -> np.ndarray:
        """
        2D mask (nx, ny) of unexplored columns: x/y regions of region_size x region_size
        without a single point over their whole height. There is no floor there, so the
        region has most likely not been scanned yet. region_size <= 0 disables the check.
        """
        nx, ny = occupied.shape[:2]
        if not region_size or region_size <= 0:
            return np.zeros((nx, ny), dtype=bool)

        f = max(1, round(region_size / voxel_size))     # voxels per region and axis
        # Pad to a multiple of f so the grid splits into whole regions
        col = np.pad(occupied.any(axis=2), ((0, -nx % f), (0, -ny % f)))
        region_has_points = col.reshape(col.shape[0] // f, f, col.shape[1] // f, f).any(axis=(1, 3))
        return np.repeat(np.repeat(~region_has_points, f, axis=0), f, axis=1)[:nx, :ny]

    # ------------------------------------------------------------------
    # ESDF
    # ------------------------------------------------------------------

    def esdf_query(self, pos) -> np.ndarray:
        """
        Args:
            pos : (3,) for single point → returns (1,) array
                (N, 3) for batch     → returns (N,) array
            Positions outside the grid return NaN.
        Obstacles are inflated by EnvParams.OBSTACLE_INFLATION (negative = inside the inflated obstacle).
        """
        pos = _to_numpy(pos).astype(np.float64).reshape(-1, 3)
        return np.atleast_1d(self.esdf_interpolator(pos)) - self.inflation

    # ------------------------------------------------------------------
    # Lidar
    # ------------------------------------------------------------------

    def _load_lidar_directions(self):
        """Precompute unit vectors for LiDAR"""
        lidar    = self.lidar_params
        h_angles = np.linspace(np.deg2rad(lidar.HORIZONTAL_FOV_MIN), np.deg2rad(lidar.HORIZONTAL_FOV_MAX),
                               int(lidar.HORIZONTAL_RESOLUTION), endpoint=False)
        v_angles = np.linspace(np.deg2rad(lidar.VERTICAL_FOV_MIN), np.deg2rad(lidar.VERTICAL_FOV_MAX),
                               int(lidar.VERTICAL_RESOLUTION), endpoint=False)
        v, h = np.meshgrid(v_angles, h_angles, indexing="ij")
        self.lidar_directions = np.stack([np.cos(v) * np.cos(h), np.cos(v) * np.sin(h), np.sin(v)],
                                         axis=-1).reshape(-1, 3)

    def _build_surface_pts(self):
        """Extract surface points (voxel centres) from the ESDF zero-crossing."""
        surface_mask = np.abs(self.esdf_map) < self.voxel_size * self.gfm_params.SURFACE_THRESHOLD
        # Unexplored columns are obstacles but have no real geometry the LiDAR could see
        surface_mask &= ~self.unknown_columns[:, :, None]
        self.surface_pts  = self.grid_origin + (np.argwhere(surface_mask) + 0.5) * self.voxel_size
        self.surface_tree = cKDTree(self.surface_pts)

    def _get_lidar_hits(self, pos):
        """Sphere-traces all LiDAR rays from pos and returns the surface points they hit."""
        directions = self.lidar_directions
        threshold  = self.voxel_size * 0.5

        curr_pos   = np.tile(pos, (len(directions), 1)).astype(np.float64)
        total_dist = np.zeros(len(directions))
        active     = np.ones(len(directions), dtype=bool)
        hit        = np.zeros(len(directions), dtype=bool)

        for _ in range(self.lidar_params.MAX_RAY_STEPS):
            idx = np.flatnonzero(active)
            if len(idx) == 0:
                break
            dist = self.esdf_interpolator(curr_pos[idx])     # real surfaces, without inflation
            hit[idx[dist < threshold]] = True
            # Stop rays that hit, left the grid (NaN) or are too long
            stop = (dist < threshold) | np.isnan(dist) | (total_dist[idx] > self.lidar_params.MAX_RAY_LENGTH)
            active[idx[stop]] = False

            idx, step = idx[~stop], np.maximum(dist[~stop], threshold)
            curr_pos[idx]   += directions[idx] * step[:, None]
            total_dist[idx] += step

        if not hit.any():
            return np.empty((0, 3))
        dist, nearest = self.surface_tree.query(curr_pos[hit], k=1)
        # Rays stopped by an unexplored column are far from any real surface: no hit
        return self.surface_pts[nearest[dist < 2 * self.voxel_size]]

    def _build_normal_interpolator(self):
        """Interpolators for the surface normals (normalised ESDF gradient) at the voxel centres."""
        grad = np.gradient(self.esdf_map, self.voxel_size)
        mag  = np.sqrt(sum(g ** 2 for g in grad)) + 1e-12
        self._normal_interps = [RegularGridInterpolator(self.center_axes, g / mag, method="linear",
                                                        bounds_error=False, fill_value=0.0) for g in grad]

    def _get_surface_normals(self, positions: np.ndarray) -> np.ndarray:
        """
        Interpolate surface normals at given positions from ESDF gradient.

        Args:
            positions : (N, 3) world-frame positions
        Returns:
            normals   : (N, 3) unit normals at each position
        """
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        normals   = np.stack([interp(positions) for interp in self._normal_interps], axis=-1)
        # Re-normalise after interpolation
        return normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12)

    # ------------------------------------------------------------------
    # GFM
    # ------------------------------------------------------------------

    def get_gfm_value(self, pos):
        normals = self._get_surface_normals(self._get_lidar_hits(pos))
        ev = np.linalg.eigvalsh(normals.T @ normals)
        return np.sum(np.log(np.maximum(ev, 1e-6)))

    # ------------------------------------------------------------------
    # Picking start / goal
    # ------------------------------------------------------------------

    def plot_slice(self, axis: str = 'z') -> np.ndarray:
        """
        Interactive ESDF slice: left-click to pick a point, "Save point" to keep it.
        Returns the saved points as an (N, 3) array once the window is closed.
        """
        plotter       = pv.Plotter()
        picked_points = []
        axis_i        = {"x": 0, "y": 1, "z": 2}[axis]

        # --- full ESDF volume in world frame ---
        grid            = pv.ImageData()
        grid.dimensions = np.array(self.grid_size) + 1
        grid.spacing    = (self.voxel_size,) * 3
        grid.origin     = self.grid_origin - 0.5 * self.voxel_size
        grid.cell_data["esdf"] = self.esdf_map.flatten(order="F")

        axis_min, axis_max = grid.bounds[2 * axis_i], grid.bounds[2 * axis_i + 1]
        current_slice_val  = [(axis_min + axis_max) / 2.0]

        def update_slice(value):
            current_slice_val[0] = value
            origin         = np.array(grid.center)
            origin[axis_i] = value
            sliced = grid.slice(normal=axis, origin=origin)
            if sliced.n_points > 0:
                plotter.add_mesh(sliced, scalars="esdf", cmap="jet", name="esdf_slice")

        update_slice(current_slice_val[0])
        plotter.add_slider_widget(
            callback=update_slice,
            rng=[axis_min, axis_max],
            value=current_slice_val[0],
            title=f"{axis.upper()} (m)",
            pointa=(0.1, 0.05), pointb=(0.9, 0.05),
            style="modern",
        )

        # --- picking ---
        last_picked = [None]

        def on_pick(point):
            pt         = np.array(point, dtype=float)
            pt[axis_i] = current_slice_val[0]
            last_picked[0] = pt
            plotter.add_mesh(pv.Sphere(radius=0.5, center=pt), color="yellow", name="pick_marker")

        plotter.enable_surface_point_picking(
            callback=on_pick, show_message=False, show_point=False, left_clicking=True,
        )

        # --- points list display ---
        def update_points_text():
            text = "\n".join(f"[{i + 1}] ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})"
                             for i, p in enumerate(picked_points)) or "No points saved"
            plotter.add_text(text, position="upper_right", font_size=8, name="points_list")

        update_points_text()

        # --- save / delete buttons ---
        def on_save(_):
            if last_picked[0] is None:
                print("No point picked yet — left-click the map first.")
                return
            picked_points.append(last_picked[0].copy())
            print(f"[{len(picked_points)}] Saved: {last_picked[0]}")
            update_points_text()

        def on_delete(_):
            if not picked_points:
                print("No points to delete.")
                return
            removed = picked_points.pop()
            print(f"Deleted: {removed}  ({len(picked_points)} remaining)")
            update_points_text()

        for callback, label, color, y in [(on_save, "Save point", "green", 80),
                                          (on_delete, "Delete last", "red", 40)]:
            plotter.add_checkbox_button_widget(callback, value=False, position=(10, y), size=30,
                                               color_on=color, color_off=color)
            plotter.add_text(label, position=(50, y + 3), font_size=10)

        # --- camera: look along the slice axis ---
        view_up = (0, 0, 1) if axis != "z" else (0, 1, 0)
        focal   = np.array(grid.center)
        plotter.camera.position    = focal + np.eye(3)[axis_i] * (axis_max - axis_min) * 3
        plotter.camera.focal_point = focal
        plotter.camera.up          = view_up
        plotter.camera.zoom(1.2)

        plotter.show_bounds(
            grid='back', location='outer', ticks='both',
            xtitle='X (m)', ytitle='Y (m)', ztitle='Z (m)', fmt="%.1f",
        )
        plotter.show()

        return np.array(picked_points)
