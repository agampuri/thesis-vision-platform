"""Top-down grasp synthesis from a depth-derived object mask.

Pipeline: bbox (from the open-vocabulary detector) -> depth segmentation against
the table plane -> centroid + principal axis -> grasp pose (x, y, z, yaw, width).
Pure geometry + OpenCV; offline-testable with synthetic depth images."""
from dataclasses import dataclass
import numpy as np

from ..vision.calibration import deproject


@dataclass
class GraspPose:
    x: float
    y: float
    z: float            # commanded TCP z for the grip
    yaw: float          # object long-axis angle in the robot base frame (rad)
    width_m: float      # object width across the SHORT axis (finger gap needed)
    top_z: float        # object top surface in base frame
    n_pixels: int


def segment_object(depth_m, bbox, d_table, min_height_m=0.005):
    """Binary mask of pixels inside bbox that stand above the table plane.
    bbox = (x1, y1, x2, y2) pixels; d_table = camera depth of the table (m)."""
    import cv2
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    h, w = depth_m.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    mask = np.zeros((h, w), np.uint8)
    crop = depth_m[y1:y2, x1:x2]
    m = (crop > 0.05) & (crop < d_table - min_height_m)
    mask[y1:y2, x1:x2] = m.astype(np.uint8) * 255
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask


class GraspPlanner:
    def __init__(self, extrinsics, zones_cfg, logger=None):
        self.ext = extrinsics
        self.table_z = float(zones_cfg.get('table_z', 0.0))
        g = zones_cfg.get('gripper', {})
        self.default_grasp_depth = float(g.get('default_grasp_depth_m', 0.015))
        self.logger = logger
        # camera height above base origin, for the near-nadir table-depth estimate
        self.cam_z = float(np.asarray(self.ext.T)[2, 3])

    def synthesize(self, depth_m, K, bbox, object_info=None):
        """-> (GraspPose | None, reason)."""
        import cv2
        d_table = self.cam_z - self.table_z
        mask = segment_object(depth_m, bbox, d_table)
        ys, xs = np.nonzero(mask)
        if len(xs) < 30:
            return None, "object mask too small (depth segmentation failed)"
        depths = depth_m[ys, xs]
        d_med = float(np.median(depths))
        d_top = float(np.percentile(depths, 10))
        cx, cy = float(xs.mean()), float(ys.mean())

        p_c = self.ext.pixel_to_base(cx, cy, d_med, K)
        top_z = float(self.ext.pixel_to_base(cx, cy, d_top, K)[2])

        pts = np.column_stack([xs, ys]).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        e1, e2 = box[1] - box[0], box[2] - box[1]
        long_e = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
        short_px = min(np.linalg.norm(e1), np.linalg.norm(e2))
        fx = float(np.asarray(K, dtype=float).reshape(3, 3)[0, 0])
        width_m = float(short_px) * d_med / fx

        lu = long_e / (np.linalg.norm(long_e) + 1e-9)
        b0 = self.ext.pixel_to_base(cx, cy, d_med, K)
        b1 = self.ext.pixel_to_base(cx + lu[0] * 30.0, cy + lu[1] * 30.0, d_med, K)
        yaw = float(np.arctan2(b1[1] - b0[1], b1[0] - b0[0]))
        # gripper is symmetric: fold into (-pi/2, pi/2]
        while yaw <= -np.pi / 2:
            yaw += np.pi
        while yaw > np.pi / 2:
            yaw -= np.pi

        gd = float((object_info or {}).get('grasp_depth_m', self.default_grasp_depth))
        z = max(top_z - gd, self.table_z + 0.003)
        return GraspPose(float(p_c[0]), float(p_c[1]), z, yaw, width_m,
                         top_z, int(len(xs))), ""


def bbox_center_base(depth_m, K, bbox, extrinsics):
    """Centre of a detection box lifted to the robot base frame, or None."""
    from ..vision.calibration import median_depth
    x1, y1, x2, y2 = bbox
    u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    d = median_depth(depth_m, u, v, k=9)
    if d is None:
        return None
    return tuple(extrinsics.pixel_to_base(u, v, d, K))
