"""Pure quaternion helpers (x, y, z, w). No ROS imports — unit-testable anywhere."""
import math


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def downward_quat_with_yaw(yaw):
    """Gripper-down orientation (0,1,0,0) rotated by `yaw` around the world Z axis."""
    q_down = (0.0, 1.0, 0.0, 0.0)
    s, c = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
    return quat_mul((0.0, 0.0, s, c), q_down)


def quat_to_matrix(q):
    x, y, z, w = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
