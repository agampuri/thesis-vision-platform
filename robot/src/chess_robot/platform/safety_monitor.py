"""Platform-wide safety: workspace polygon, feasibility gates, motion-in-frame check.
Gates run BEFORE any motion is planned; infeasible intents are rejected with a reason."""
import math


def point_in_polygon(x, y, poly):
    """Ray-casting point-in-polygon. poly: [[x, y], ...]"""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


class SafetyMonitor:
    def __init__(self, zones_cfg, logger=None):
        self.logger = logger
        self.poly = zones_cfg.get('workspace_polygon') or []
        s = zones_cfg.get('safety', {})
        g = zones_cfg.get('gripper', {})
        self.max_reach = float(s.get('max_reach_m', 0.42))
        self.payload_net_g = float(s.get('payload_net_g', 200))
        self.max_opening = float(g.get('max_opening_m', 0.016))
        self.width_clearance = float(s.get('width_clearance_m', 0.004))
        self.motion_thresh = float(s.get('motion_diff_threshold', 12.0))
        self.table_z = float(zones_cfg.get('table_z', 0.0))

    def check_point(self, x, y):
        if self.poly and not point_in_polygon(x, y, self.poly):
            return False, f"({x:.3f},{y:.3f}) outside workspace polygon"
        if math.hypot(x, y) > self.max_reach:
            return False, f"({x:.3f},{y:.3f}) beyond reach {self.max_reach} m"
        return True, ""

    def check_grasp(self, grasp, object_info=None):
        """grasp: GraspPose. Returns (ok, reason)."""
        ok, why = self.check_point(grasp.x, grasp.y)
        if not ok:
            return False, why
        if grasp.width_m + self.width_clearance > self.max_opening:
            return False, (f"object width {grasp.width_m*1000:.0f} mm + clearance "
                           f"exceeds gripper opening {self.max_opening*1000:.0f} mm")
        if grasp.z < self.table_z + 0.002:
            return False, f"grasp z {grasp.z:.3f} would hit the table ({self.table_z:.3f})"
        if object_info and float(object_info.get('mass_g', 0)) > self.payload_net_g:
            return False, "object exceeds net payload"
        return True, ""

    def motion_in_frame(self, prev_gray, cur_gray):
        """True if significant motion between two grayscale frames (hand entered)."""
        import cv2
        import numpy as np
        if prev_gray is None or cur_gray is None:
            return False
        d = cv2.absdiff(prev_gray, cur_gray)
        return float(np.mean(d)) > self.motion_thresh
