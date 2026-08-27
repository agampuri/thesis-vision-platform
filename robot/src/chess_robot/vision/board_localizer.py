"""
chess_robot.vision.board_localizer — ArUco markers on the board frame -> 64 square
centres in the ROBOT BASE frame, at any board position/rotation on the table.

Board frame convention: origin = centre of square a1, +x toward the h-file,
+y toward rank 8, units metres. Marker centres are configured in this frame
(vision.board.markers in board_config.yaml) after gluing + measuring them.
"""
import time
import numpy as np

from .calibration import detect_aruco_centers, median_depth, project_base_to_pixel


def solve_rigid_2d(P_src, P_dst):
    """2D rigid transform (rotation + translation, no scale) src->dst.
    P_src, P_dst: Nx2 arrays, N >= 2. Returns (R 2x2, t (2,), rms)."""
    A = np.asarray(P_src, dtype=float)
    B = np.asarray(P_dst, dtype=float)
    assert A.shape == B.shape and A.shape[0] >= 2 and A.shape[1] == 2
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    t = cb - R @ ca
    res = np.linalg.norm((A @ R.T + t) - B, axis=1)
    return R, t, float(np.sqrt(np.mean(res ** 2)))


def square_name(file_idx, rank_idx):
    return chr(ord('a') + file_idx) + str(rank_idx + 1)


class BoardPose:
    """Result of one successful localization."""

    def __init__(self, squares, squares_px, rms, n_markers, z_plane, pitch_px):
        self.squares = squares          # {'a1': (x, y, z), ...} robot base frame
        self.squares_px = squares_px    # {'a1': (u, v), ...} pixel positions
        self.rms = rms                  # 2D fit residual (m)
        self.n_markers = n_markers
        self.z_plane = z_plane          # board surface height in base frame (m)
        self.pitch_px = pitch_px        # approx square pitch in pixels (snap radius base)
        self.timestamp = time.time()


class BoardLocalizer:
    def __init__(self, board_cfg, logger=None):
        """board_cfg = config['vision']['board'] dict."""
        self.pitch_x = float(board_cfg['pitch_x'])
        self.pitch_y = float(board_cfg['pitch_y'])
        self.markers = {int(k): (float(v[0]), float(v[1]))
                        for k, v in board_cfg['markers'].items()}
        self.aruco_dict = board_cfg.get('aruco_dict', 'DICT_4X4_50')
        self.min_markers = int(board_cfg.get('min_markers', 2))
        self.logger = logger

    # ---- pure geometry (unit-testable without OpenCV) ----
    def pose_from_marker_base_points(self, marker_base, z_plane, T_base_cam=None, K=None):
        """marker_base: {id: (x, y)} measured marker centres in base frame.
        Returns BoardPose or None."""
        ids = [i for i in marker_base if i in self.markers]
        if len(ids) < self.min_markers:
            return None
        P_board = np.array([self.markers[i] for i in ids])
        P_base = np.array([marker_base[i][:2] for i in ids])
        R, t, rms = solve_rigid_2d(P_board, P_base)
        squares, squares_px = {}, {}
        for f in range(8):
            for r in range(8):
                pb = R @ np.array([f * self.pitch_x, r * self.pitch_y]) + t
                name = square_name(f, r)
                squares[name] = (float(pb[0]), float(pb[1]), float(z_plane))
                if T_base_cam is not None and K is not None:
                    u, v, _ = project_base_to_pixel(squares[name], T_base_cam, K)
                    squares_px[name] = (u, v)
        pitch_px = None
        if 'a1' in squares_px and 'b1' in squares_px and squares_px['a1'][0] is not None:
            pa, pb_ = np.array(squares_px['a1']), np.array(squares_px['b1'])
            pitch_px = float(np.linalg.norm(pa - pb_))
        return BoardPose(squares, squares_px, rms, len(ids), z_plane, pitch_px)

    # ---- full pipeline from a frame ----
    def locate(self, bgr, depth_m, K, extrinsics):
        """Detect markers in the frame, lift them to the base frame, fit the board.
        extrinsics: CameraExtrinsics. Returns BoardPose or None."""
        centers_px = detect_aruco_centers(bgr, self.aruco_dict)
        marker_base, zs = {}, []
        for mid, (u, v) in centers_px.items():
            if mid not in self.markers:
                continue
            z = median_depth(depth_m, u, v, k=5)
            if z is None or z <= 0:
                continue
            p = extrinsics.pixel_to_base(u, v, z, K)
            marker_base[mid] = (p[0], p[1])
            zs.append(p[2])
        if len(marker_base) < self.min_markers:
            if self.logger:
                self.logger.debug(f"Board markers found: {sorted(marker_base)} "
                                  f"(need {self.min_markers})")
            return None
        z_plane = float(np.median(zs))
        pose = self.pose_from_marker_base_points(marker_base, z_plane, extrinsics.T, K)
        if pose is not None and pose.rms > 0.004 and self.logger:
            self.logger.warning(f"Board fit residual high: {pose.rms*1000:.1f} mm")
        return pose
