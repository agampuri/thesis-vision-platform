"""Stereolabs ZED / ZED Mini adapter with the SAME interface as RealSenseDirect:
    start() / stop() / get_frame() -> (bgr, depth_m, K)
so everything downstream (calibration, localizer, grasp planner) is unchanged.

Depth is computed on the GPU by the ZED SDK (pyzed). Heavy import is lazy.
Invalid depth (NaN/inf) is converted to 0.0, matching the RealSense convention.

Self-test (no robot needed):
    cd ~/chess_remote/robot/src && python3 -m chess_robot.vision.camera_zed --preview
"""
import time

import numpy as np


class ZEDCamera:
    def __init__(self, width=1280, height=720, fps=30, warmup=10, logger=None,
                 serial=None, depth_mode=None, min_depth_m=0.15):
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.warmup = int(warmup)
        self.logger = logger
        self.serial = serial
        self.depth_mode = depth_mode          # e.g. 'NEURAL' | 'ULTRA'; None = best available
        self.min_depth_m = float(min_depth_m)
        self._cam = None
        self._runtime = None
        self._mats = None
        self._K = None

    # ---------------- lifecycle ----------------
    def start(self):
        import pyzed.sl as sl  # lazy: rest of the code imports this module fine without the SDK
        init = sl.InitParameters()
        # resolution: HD720 matches the pipeline's 1280x720 default
        res_map = {(1280, 720): 'HD720', (1920, 1080): 'HD1080', (2208, 1242): 'HD2K'}
        res_name = res_map.get((self.width, self.height), 'HD720')
        init.camera_resolution = getattr(sl.RESOLUTION, res_name)
        init.camera_fps = self.fps
        init.coordinate_units = sl.UNIT.METER
        init.depth_minimum_distance = self.min_depth_m
        # depth mode: explicit if configured, else best available in this SDK build
        wanted = ([self.depth_mode] if self.depth_mode else []) + ['NEURAL', 'ULTRA', 'PERFORMANCE']
        for name in wanted:
            if name and hasattr(sl.DEPTH_MODE, name):
                init.depth_mode = getattr(sl.DEPTH_MODE, name)
                self._depth_mode_name = name
                break
        if self.serial:
            init.set_from_serial_number(int(self.serial))

        self._cam = sl.Camera()
        err = self._cam.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {err} "
                               "(USB3 port? another app using the camera? SDK installed?)")
        self._runtime = sl.RuntimeParameters()
        self._mats = (sl.Mat(), sl.Mat())

        # intrinsics of the LEFT rectified camera (depth is registered to it)
        info = self._cam.get_camera_information()
        try:    # SDK 4.x layout
            lc = info.camera_configuration.calibration_parameters.left_cam
        except AttributeError:  # SDK 3.x layout
            lc = info.calibration_parameters.left_cam
        self._K = [[float(lc.fx), 0.0, float(lc.cx)],
                   [0.0, float(lc.fy), float(lc.cy)],
                   [0.0, 0.0, 1.0]]

        for _ in range(self.warmup):
            self._grab()
        if self.logger:
            self.logger.info(f"ZED started {self.width}x{self.height}@{self.fps} "
                             f"depth={getattr(self, '_depth_mode_name', '?')} "
                             f"fx={self._K[0][0]:.1f}")

    def stop(self):
        if self._cam is not None:
            self._cam.close()
            self._cam = None

    # ---------------- frames ----------------
    def _grab(self, tries=10):
        import pyzed.sl as sl
        for _ in range(tries):
            if self._cam.grab(self._runtime) == sl.ERROR_CODE.SUCCESS:
                return True
            time.sleep(0.02)
        return False

    def get_frame(self):
        """-> (bgr uint8 HxWx3, depth float32 metres HxW aligned to bgr, K 3x3)."""
        import pyzed.sl as sl
        if self._cam is None:
            raise RuntimeError("ZEDCamera.get_frame() before start()")
        if not self._grab():
            raise RuntimeError("ZED grab failed repeatedly")
        m_img, m_depth = self._mats
        self._cam.retrieve_image(m_img, sl.VIEW.LEFT)          # BGRA, rectified left
        self._cam.retrieve_measure(m_depth, sl.MEASURE.DEPTH)  # float32 m, aligned to left
        bgr = np.ascontiguousarray(m_img.get_data()[:, :, :3]).copy()
        depth = np.nan_to_num(m_depth.get_data().astype(np.float32),
                              nan=0.0, posinf=0.0, neginf=0.0)
        return bgr, depth, self._K

    @staticmethod
    def list_devices():
        import pyzed.sl as sl
        return [(str(d.serial_number), str(d.camera_model)) for d in sl.Camera.get_device_list()]


def _preview():
    cam = ZEDCamera()
    print("Devices:", ZEDCamera.list_devices())
    print("Starting ZED ...")
    cam.start()
    try:
        for i in range(5):
            bgr, depth, K = cam.get_frame()
            h, w = depth.shape[:2]
            centre = depth[h // 2 - 2:h // 2 + 3, w // 2 - 2:w // 2 + 3]
            valid = depth[depth > 0]
            print(f"frame {i}: {w}x{h}  fx={K[0][0]:.1f} cx={K[0][2]:.1f}  "
                  f"centre depth={float(np.median(centre)):.3f} m  "
                  f"valid px={100.0 * valid.size / depth.size:.1f}%")
        try:
            import cv2
            cv2.imwrite('/tmp/zed_preview.jpg', bgr)
            print("saved /tmp/zed_preview.jpg")
        except Exception:
            pass
    finally:
        cam.stop()
    print("ZED PREVIEW OK")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--preview', action='store_true')
    a = ap.parse_args()
    if a.preview:
        _preview()
