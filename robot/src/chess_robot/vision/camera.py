"""
chess_robot.vision.camera — direct RealSense D435 access (no ROS dependency).

The app and all tools open the camera directly via pyrealsense2. Do NOT run the
realsense2_camera ROS node at the same time (the device can only be opened once).
"""
import time
import numpy as np


class RealSenseDirect:
    """Color + aligned-depth frames from a RealSense D435.

    get_frame() -> (bgr uint8 HxWx3, depth float32 HxW in meters, K 3x3 list)
    """

    def __init__(self, width=1280, height=720, fps=30, warmup=10, logger=None,
                 serial=None):
        self.serial = serial
        self.width, self.height, self.fps = width, height, fps
        self.warmup = warmup
        self.logger = logger
        self._pipe = None
        self._align = None
        self._scale = None
        self._K = None

    def start(self):
        import pyrealsense2 as rs  # lazy: tools/tests can import this module without the lib
        cfg = rs.config()
        if self.serial:
            cfg.enable_device(str(self.serial))
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self._pipe = rs.pipeline()
        profile = self._pipe.start(cfg)
        self._align = rs.align(rs.stream.color)
        self._scale = profile.get_device().first_depth_sensor().get_depth_scale()
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self._K = [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]]
        for _ in range(self.warmup):  # let auto-exposure settle
            self._pipe.wait_for_frames()
        if self.logger:
            self.logger.info(f"RealSense started {self.width}x{self.height}@{self.fps} "
                             f"fx={intr.fx:.1f} fy={intr.fy:.1f}")

    def get_frame(self, timeout_ms=2000):
        frames = self._pipe.wait_for_frames(timeout_ms)
        frames = self._align.process(frames)
        c, d = frames.get_color_frame(), frames.get_depth_frame()
        if not c or not d:
            raise RuntimeError("RealSense returned incomplete frame")
        bgr = np.asanyarray(c.get_data()).copy()
        depth = np.asanyarray(d.get_data()).astype(np.float32) * self._scale
        return bgr, depth, self._K

    @staticmethod
    def list_devices():
        """[(serial, name)] of connected RealSense devices."""
        import pyrealsense2 as rs
        return [(d.get_info(rs.camera_info.serial_number),
                 d.get_info(rs.camera_info.name))
                for d in rs.context().devices]

    def stop(self):
        if self._pipe is not None:
            try:
                self._pipe.stop()
            except Exception:
                pass
            self._pipe = None
