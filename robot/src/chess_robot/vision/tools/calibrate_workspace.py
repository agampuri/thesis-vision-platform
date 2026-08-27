"""Workspace registration (Method A): jog the robot TCP to ArUco markers, enter the
robot coordinates from UFACTORY Studio, solve T_base<-cam, save extrinsics.

Usage:
  cd ~/chess_remote/robot/src && python3 -m chess_robot.vision.tools.calibrate_workspace

Flow per marker: place/keep marker flat -> jog TCP tip onto its CENTRE -> read
X Y Z (mm) in UFACTORY Studio -> type:  <id> <X> <Y> <Z>     e.g.:  0 312.4 -88.1 106.9
Commands:  list  = show currently visible marker ids    done = solve & save (need >= 4)
Aim for 6-8 markers spread over the workspace, 1-2 raised ~20 mm on a block."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from chess_robot.vision.camera_factory import create_camera, load_camera_cfg_from_board_config
from chess_robot.vision.calibration import (detect_aruco_centers, median_depth,
                                            deproject, solve_rigid_transform,
                                            CameraExtrinsics)

EXT_PATH = os.path.expanduser('~/chess_remote/config/camera_extrinsics.yaml')
REF_PATH = os.path.expanduser('~/chess_remote/config/calibration_reference.yaml')


def main():
    cam = create_camera(load_camera_cfg_from_board_config())
    print("Starting camera...")
    cam.start()
    pairs = {}  # id -> (p_cam, p_base)
    try:
        while True:
            line = input("\n<id> <X_mm> <Y_mm> <Z_mm>  |  list  |  done  > ").strip()
            if line == 'done':
                break
            bgr, depth, K = cam.get_frame()
            seen = detect_aruco_centers(bgr)
            if line == 'list':
                print(f"  visible markers: {sorted(seen)}   collected: {sorted(pairs)}")
                continue
            try:
                mid, x, y, z = line.split()
                mid = int(mid)
                p_base = np.array([float(x), float(y), float(z)]) / 1000.0
            except ValueError:
                print("  could not parse — format: 0 312.4 -88.1 106.9")
                continue
            if mid not in seen:
                print(f"  marker {mid} NOT visible (visible: {sorted(seen)}) — not saved")
                continue
            u, v = seen[mid]
            zc = median_depth(depth, u, v, k=7)
            if zc is None:
                print("  no valid depth at marker centre — fix glare/distance, retry")
                continue
            p_cam = deproject(u, v, zc, K)
            pairs[mid] = (p_cam, p_base)
            print(f"  saved marker {mid}: cam={np.round(p_cam,4).tolist()} "
                  f"base={np.round(p_base,4).tolist()}   ({len(pairs)} total)")
    finally:
        cam.stop()

    if len(pairs) < 4:
        print(f"Only {len(pairs)} pairs — need >= 4 (6-8 recommended). Aborting.")
        sys.exit(1)

    ids = sorted(pairs)
    P_cam = np.array([pairs[i][0] for i in ids])
    P_base = np.array([pairs[i][1] for i in ids])
    R, t, rms, res = solve_rigid_transform(P_cam, P_base)
    print(f"\nFit over {len(ids)} markers: RMS = {rms*1000:.2f} mm")
    for i, mid in enumerate(ids):
        print(f"  marker {mid}: residual {res[i]*1000:.2f} mm")
    if len(ids) >= 5:
        loo = []
        for k in range(len(ids)):
            m = [j for j in range(len(ids)) if j != k]
            Rk, tk, _, _ = solve_rigid_transform(P_cam[m], P_base[m])
            loo.append(np.linalg.norm(Rk @ P_cam[k] + tk - P_base[k]))
        print(f"  leave-one-out: mean {np.mean(loo)*1000:.2f} mm, max {np.max(loo)*1000:.2f} mm")

    ext = CameraExtrinsics.from_Rt(R, t, fit_rms_mm=rms * 1000,
                                   n_points=len(ids), method='workspace_correspondence')
    ext.save(EXT_PATH)
    import yaml
    with open(REF_PATH, 'w') as f:
        yaml.safe_dump({'markers': {int(i): [float(v) for v in pairs[i][1]] for i in ids}}, f)
    print(f"\nSaved extrinsics -> {EXT_PATH}\nSaved reference  -> {REF_PATH}")
    gate = "PASS" if rms <= 0.002 else "ABOVE TARGET (<= 2 mm) — add markers / check depth glare"
    print(f"Acceptance gate: {gate}")

if __name__ == '__main__':
    main()
