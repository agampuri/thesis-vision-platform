"""Generate printable ArUco markers (DICT_4X4_50).
Usage:  python3 make_markers.py --ids 0 1 2 3 4 5 6 7 10 11 12 13 --size-mm 40 --out ~/chess_remote/markers
Print the PNGs at 100% scale / 300 DPI; verify the black square with a ruler."""
import argparse, os
import numpy as np
import cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids', type=int, nargs='+', required=True)
    ap.add_argument('--size-mm', type=float, default=40.0)
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--out', default=os.path.expanduser('~/chess_remote/markers'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    px = int(round(a.size_mm / 25.4 * a.dpi))
    border = px // 5
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for mid in a.ids:
        if hasattr(cv2.aruco, 'generateImageMarker'):
            img = cv2.aruco.generateImageMarker(d, mid, px)
        else:
            img = cv2.aruco.drawMarker(d, mid, px)
        canvas = np.full((px + 2 * border, px + 2 * border), 255, np.uint8)
        canvas[border:border + px, border:border + px] = img
        cv2.putText(canvas, f"id {mid}  {a.size_mm:.0f}mm", (border, canvas.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
        path = os.path.join(a.out, f"aruco_{mid}.png")
        cv2.imwrite(path, canvas)
        print(f"wrote {path}  (print at {a.dpi} DPI -> black square = {a.size_mm} mm)")

if __name__ == '__main__':
    main()
