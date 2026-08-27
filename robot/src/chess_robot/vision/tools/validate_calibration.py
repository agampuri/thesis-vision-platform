"""60-second session check: re-detect the calibration markers and compare against
the saved reference. Run at the START of every lab session; if it fails, the
camera moved -> recalibrate before trusting any vision coordinate.

Usage: cd ~/chess_remote/robot/src && python3 -m chess_robot.vision.tools.validate_calibration"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from chess_robot.vision.camera_factory import create_camera, load_camera_cfg_from_board_config
from chess_robot.vision.calibration import (detect_aruco_centers, median_depth,
                                            CameraExtrinsics)

EXT_PATH = os.path.expanduser('~/chess_remote/config/camera_extrinsics.yaml')
REF_PATH = os.path.expanduser('~/chess_remote/config/calibration_reference.yaml')
THRESH_MM = 4.0

def main():
    import yaml
    ext = CameraExtrinsics.load(EXT_PATH)
    ref = yaml.safe_load(open(REF_PATH))['markers']
    cam = create_camera(load_camera_cfg_from_board_config())
    cam.start()
    try:
        bgr, depth, K = cam.get_frame()
    finally:
        cam.stop()
    seen = detect_aruco_centers(bgr)
    errs = []
    print(f"{'id':>4} {'offset_mm':>10}")
    for mid, p_ref in sorted(ref.items()):
        if int(mid) not in seen:
            print(f"{mid:>4} {'not visible':>10}")
            continue
        u, v = seen[int(mid)]
        z = median_depth(depth, u, v, k=7)
        if z is None:
            print(f"{mid:>4} {'no depth':>10}")
            continue
        p = ext.pixel_to_base(u, v, z, K)
        e = np.linalg.norm(p - np.array(p_ref)) * 1000
        errs.append(e)
        print(f"{mid:>4} {e:>10.2f}")
    if not errs:
        print("No reference markers measurable — cannot validate.")
        sys.exit(2)
    print(f"\nmean {np.mean(errs):.2f} mm   max {np.max(errs):.2f} mm   threshold {THRESH_MM} mm")
    if np.max(errs) > THRESH_MM:
        print("FAIL — camera likely moved. Recalibrate (tools/calibrate_workspace.py).")
        sys.exit(1)
    print("PASS — calibration still valid.")

if __name__ == '__main__':
    main()
