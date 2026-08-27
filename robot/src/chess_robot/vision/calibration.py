"""
chess_robot.vision.calibration — math core of the vision platform.

Pure numpy + yaml. Runs and self-tests WITHOUT any camera or robot attached:
    python3 calibration.py
"""
import os
import datetime
import numpy as np
import yaml


# ---------------- pinhole geometry ----------------

def deproject(u, v, depth_m, K):
    """Pixel (u, v) + depth (m) + 3x3 intrinsics K  ->  3D point in CAMERA frame (m)."""
    K = np.asarray(K, dtype=float).reshape(3, 3)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = float(depth_m)
    return np.array([(float(u) - cx) * z / fx,
                     (float(v) - cy) * z / fy,
                     z])


def median_depth(depth_m, u, v, k=5):
    """Median of valid (>0) depths in a k x k window around pixel (u, v).
    depth_m: HxW float array in meters. Returns None if the window has no valid depth."""
    h, w = depth_m.shape[:2]
    r = k // 2
    u0, u1 = max(0, int(u) - r), min(w, int(u) + r + 1)
    v0, v1 = max(0, int(v) - r), min(h, int(v) + r + 1)
    win = depth_m[v0:v1, u0:u1]
    vals = win[win > 0]
    return float(np.median(vals)) if vals.size else None


# ---------------- rigid transform (Kabsch / Umeyama, no scale) ----------------

def solve_rigid_transform(P_src, P_dst):
    """R (3x3), t (3,) minimizing ||R @ p_src + t - p_dst||².
    P_src, P_dst: Nx3 corresponding points, N >= 3 non-collinear.
    Returns (R, t, rms, residuals)."""
    A = np.asarray(P_src, dtype=float)
    B = np.asarray(P_dst, dtype=float)
    assert A.shape == B.shape and A.ndim == 2 and A.shape[1] == 3 and A.shape[0] >= 3
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cb - R @ ca
    residuals = np.linalg.norm((A @ R.T + t) - B, axis=1)
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    return R, t, rms, residuals


def axis_angle_to_R(axis, angle):
    """Rotation matrix from axis-angle (numpy only; used by the self-test)."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    x, y, z = a
    c, s, C = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


# ---------------- extrinsics container ----------------

class CameraExtrinsics:
    """T_base<-cam as a 4x4 homogeneous matrix, with YAML persistence."""

    def __init__(self, T=None):
        self.T = np.eye(4) if T is None else np.asarray(T, dtype=float).reshape(4, 4)
        self.meta = {}

    @classmethod
    def from_Rt(cls, R, t, **meta):
        T = np.eye(4)
        T[:3, :3] = np.asarray(R, dtype=float)
        T[:3, 3] = np.asarray(t, dtype=float).ravel()
        e = cls(T)
        e.meta = dict(meta)
        return e

    def cam_to_base(self, p_cam):
        p = np.asarray(p_cam, dtype=float).ravel()
        return self.T[:3, :3] @ p + self.T[:3, 3]

    def pixel_to_base(self, u, v, depth_m, K):
        return self.cam_to_base(deproject(u, v, depth_m, K))

    def save(self, path):
        data = {"T_base_cam": [[float(x) for x in row] for row in self.T],
                "created": datetime.datetime.now().isoformat(timespec="seconds")}
        for k, v in self.meta.items():
            data[k] = float(v) if isinstance(v, (int, float, np.floating)) else v
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = yaml.safe_load(f)
        e = cls(np.array(data["T_base_cam"], dtype=float))
        e.meta = {k: v for k, v in data.items() if k != "T_base_cam"}
        return e


def project_base_to_pixel(p_base, T_base_cam, K):
    """Robot-base 3D point -> pixel (u, v) and camera-axis depth z.
    Inverse of pixel_to_base. Returns (u, v, z_cam); z_cam <= 0 means behind camera."""
    T = np.asarray(T_base_cam, dtype=float).reshape(4, 4)
    R, t = T[:3, :3], T[:3, 3]
    p_cam = R.T @ (np.asarray(p_base, dtype=float).ravel() - t)
    K = np.asarray(K, dtype=float).reshape(3, 3)
    z = p_cam[2]
    if z <= 1e-9:
        return None, None, z
    u = K[0, 0] * p_cam[0] / z + K[0, 2]
    v = K[1, 1] * p_cam[1] / z + K[1, 2]
    return float(u), float(v), float(z)


# ---------------- ArUco (works on old AND new OpenCV APIs) ----------------

def detect_aruco_centers(image, dict_name="DICT_4X4_50"):
    """Detect ArUco markers; returns {marker_id: (cx, cy)} in pixel coordinates.
    Handles both OpenCV <=4.6 (detectMarkers) and >=4.7 (ArucoDetector) APIs."""
    import cv2
    gray = image
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    if hasattr(cv2.aruco, "ArucoDetector"):
        det = cv2.aruco.ArucoDetector(adict, cv2.aruco.DetectorParameters())
        corners, ids, _ = det.detectMarkers(gray)
    else:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, adict, parameters=params)
    out = {}
    if ids is not None:
        for mid, c in zip(ids.ravel().tolist(), corners):
            cc = c.reshape(-1, 2).mean(axis=0)
            out[int(mid)] = (float(cc[0]), float(cc[1]))
    return out


# ---------------- self-test ----------------

def _self_test():
    rng = np.random.default_rng(42)
    print("== chess_robot.vision.calibration self-test (no hardware needed) ==")

    # 1) rigid-transform recovery under 1 mm noise
    R_true = axis_angle_to_R(rng.normal(size=3), 0.7)
    t_true = np.array([0.40, -0.15, 0.90])
    P_cam = rng.uniform([-0.3, -0.2, 0.5], [0.3, 0.2, 0.9], size=(8, 3))
    P_base = P_cam @ R_true.T + t_true
    R, t, rms, _ = solve_rigid_transform(P_cam, P_base + rng.normal(scale=0.001, size=P_base.shape))
    rot_err = np.degrees(np.arccos(np.clip((np.trace(R.T @ R_true) - 1) / 2, -1, 1)))
    t_err = np.linalg.norm(t - t_true)
    print(f"  rigid fit : rms={rms*1000:.2f} mm  rot_err={rot_err:.3f} deg  t_err={t_err*1000:.2f} mm")
    assert rms < 0.003 and rot_err < 0.5 and t_err < 0.005, "rigid transform fit FAILED"

    # 2) deprojection sanity: principal point at 0.75 m -> (0, 0, 0.75)
    K = [[910.0, 0, 640.0], [0, 910.0, 360.0], [0, 0, 1]]
    p = deproject(640, 360, 0.75, K)
    assert np.allclose(p, [0, 0, 0.75]), p
    print(f"  deproject : principal point -> {np.round(p, 4).tolist()}")

    # 3) pixel->base roundtrip + YAML save/load
    ext = CameraExtrinsics.from_Rt(R, t, fit_rms_mm=rms * 1000, n_points=8, method="kabsch")
    path = "/tmp/_extrinsics_selftest.yaml"
    ext.save(path)
    ext2 = CameraExtrinsics.load(path)
    assert np.allclose(ext.T, ext2.T), "YAML roundtrip mismatch"
    pb = ext2.pixel_to_base(800, 500, 0.78, K)
    print(f"  yaml      : roundtrip OK ({path}); pixel(800,500)@0.78m -> base {np.round(pb, 4).tolist()}")

    # 4) median depth must survive a hole at the centre pixel
    d = np.zeros((10, 10)); d[4:7, 4:7] = 0.75; d[5, 5] = 0.0
    md = median_depth(d, 5, 5, k=3)
    assert md is not None and abs(md - 0.75) < 1e-9
    print(f"  depth     : median with centre hole -> {md} m")

    print("== ALL SELF-TESTS PASSED ==")


if __name__ == "__main__":
    _self_test()
