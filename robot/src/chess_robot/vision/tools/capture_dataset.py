"""Dataset capture: saves RGB png + depth npy pairs for labeling/training.
Usage: cd ~/chess_remote/robot/src && python3 -m chess_robot.vision.tools.capture_dataset --session s01
Press ENTER to save a frame, type q + ENTER to quit. Vary board pose, pieces, lighting."""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from chess_robot.vision.camera_factory import create_camera, load_camera_cfg_from_board_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', required=True, help='session name, e.g. s01 (split datasets BY session)')
    ap.add_argument('--out', default=os.path.expanduser('~/chess_remote/dataset/raw'))
    a = ap.parse_args()
    import cv2
    out = os.path.join(a.out, a.session)
    os.makedirs(out, exist_ok=True)
    cam = create_camera(load_camera_cfg_from_board_config())
    cam.start()
    n = len([f for f in os.listdir(out) if f.endswith('.png')])
    print(f"Saving to {out} (starting at frame {n}). ENTER=save, q=quit.")
    try:
        while True:
            cmd = input(f"[{n} saved] > ").strip().lower()
            if cmd == 'q':
                break
            bgr, depth, K = cam.get_frame()
            cv2.imwrite(os.path.join(out, f"frame_{n:04d}.png"), bgr)
            np.save(os.path.join(out, f"frame_{n:04d}_depth.npy"), depth)
            if n == 0:
                json.dump({'K': K}, open(os.path.join(out, 'intrinsics.json'), 'w'))
            n += 1
    finally:
        cam.stop()
    print(f"Done — {n} frames in {out}")

if __name__ == '__main__':
    main()
