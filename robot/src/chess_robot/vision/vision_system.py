"""
chess_robot.vision.vision_system — the facade the rest of the app talks to.

Owns: camera, extrinsics, board localizer, (optional) piece detector, timing.
Coordinates are valid only while fresh; consumers fall back to config geometry
when get_square_xyz() returns None — the system never degrades below the
predecessor's behaviour.
"""
import os
import time

from .calibration import CameraExtrinsics
from .board_localizer import BoardLocalizer


class VisionSystem:
    def __init__(self, config, logger=None, camera=None):
        """config = the 'vision:' block of board_config.yaml (dict).
        camera: injectable for tests; default = RealSenseDirect from config."""
        self.config = config
        self.logger = logger
        self.freshness_s = float(config.get('freshness_s', 10.0))
        ext_path = os.path.expanduser(config['extrinsics_file'])
        if not os.path.exists(ext_path):
            raise FileNotFoundError(
                f"Extrinsics not found: {ext_path} — run tools/calibrate_workspace.py first")
        self.extrinsics = CameraExtrinsics.load(ext_path)
        self.localizer = BoardLocalizer(config['board'], logger=logger)

        if camera is not None:
            self.camera = camera
        else:
            from .camera_factory import create_camera
            self.camera = create_camera(config.get('camera', {}), logger=logger)

        self.detector = None
        model_path = config.get('model_occupancy')
        if model_path:
            model_path = os.path.expanduser(model_path)
            if os.path.exists(model_path):
                from .piece_detector import PieceDetector
                self.detector = PieceDetector(model_path,
                                              conf=float(config.get('conf_threshold', 0.40)),
                                              logger=logger)
            elif logger:
                logger.warning(f"Occupancy model not found at {model_path} — "
                               "coordinates will work, move detection will not")

        self._pose = None
        self._last_raw = None
        self._occupancy = None
        self._unsnapped = 0
        self._last_update = 0.0
        self.timings = {}

    # ---------------- lifecycle ----------------
    def start(self):
        self.camera.start()

    def stop(self):
        try:
            self.camera.stop()
        except Exception:
            pass

    # ---------------- main entry ----------------
    def update(self):
        """Grab one frame, localize the board, (optionally) detect pieces.
        Returns True if a board pose was obtained."""
        t0 = time.time()
        try:
            bgr, depth, K = self.camera.get_frame()
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Camera frame failed: {e}")
            return False
        t1 = time.time()
        self._last_raw = (bgr, depth, K)
        pose = self.localizer.locate(bgr, depth, K, self.extrinsics)
        t2 = time.time()
        if pose is None:
            return False
        self._pose = pose

        self._occupancy, self._unsnapped = None, 0
        if self.detector is not None and pose.pitch_px:
            from .piece_detector import occupancy_from_detections
            dets = self.detector.detect(bgr)
            self._occupancy, self._unsnapped = occupancy_from_detections(
                dets, pose.squares_px, snap_radius_px=0.45 * pose.pitch_px)
        t3 = time.time()

        self._last_update = time.time()
        self.timings = {'capture_ms': (t1 - t0) * 1000,
                        'localize_ms': (t2 - t1) * 1000,
                        'detect_ms': (t3 - t2) * 1000}
        return True

    # ---------------- consumers ----------------
    def is_fresh(self, max_age=None):
        age = time.time() - self._last_update
        return self._pose is not None and age <= (max_age or self.freshness_s)

    def get_square_xyz(self, square):
        """(x, y, z) in robot base frame, or None if stale/unknown -> caller falls back."""
        if not self.is_fresh():
            return None
        return self._pose.squares.get(square)

    def get_occupancy(self):
        """(occupancy dict or None, unsnapped count) from the most recent update."""
        if not self.is_fresh():
            return None, 0
        return self._occupancy, self._unsnapped

    def get_raw_frame(self):
        """(bgr, depth_m, K) from the most recent successful update, or None."""
        if not self.is_fresh() or self._last_raw is None:
            return None
        return self._last_raw

    def health(self):
        return {'fresh': self.is_fresh(),
                'age_s': round(time.time() - self._last_update, 1) if self._last_update else None,
                'markers': self._pose.n_markers if self._pose else 0,
                'fit_rms_mm': round(self._pose.rms * 1000, 2) if self._pose else None,
                'detector': self.detector is not None,
                'timings_ms': {k: round(v, 1) for k, v in self.timings.items()}}
