"""Point-to-select: resolve a pointing hand to a table location (S7).
Fingertip + knuckle define a 3D ray; its intersection with the table plane is
the pointed spot. MediaPipe is imported lazily (pip3 install mediapipe)."""
import numpy as np

from ..vision.calibration import deproject, median_depth


def resolve_pointing(bgr, depth_m, K, extrinsics, table_z, logger=None):
    """-> (x, y) in robot base frame, or None if no pointing hand is found."""
    import mediapipe as mp  # lazy
    import cv2
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                  min_detection_confidence=0.6) as hands:
        res = hands.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        return None
    lm = res.multi_hand_landmarks[0].landmark
    h, w = depth_m.shape[:2]
    pts = []
    for idx in (5, 8):  # index knuckle -> index fingertip
        u, v = lm[idx].x * w, lm[idx].y * h
        d = median_depth(depth_m, u, v, k=9)
        if d is None:
            return None
        pts.append(extrinsics.cam_to_base(deproject(u, v, d, K)))
    p0, p1 = np.asarray(pts[0]), np.asarray(pts[1])
    ray = p1 - p0
    if abs(ray[2]) < 1e-6:
        return None
    t = (table_z - p1[2]) / ray[2]
    if t < 0:
        return None  # pointing upward
    hit = p1 + t * ray
    if logger:
        logger.info(f"Pointing resolved to ({hit[0]:.3f}, {hit[1]:.3f})")
    return float(hit[0]), float(hit[1])
