"""MotionService: one place for arm + gripper, with selectable gripper backend.
backend 'ufactory' = stock two-finger gripper (no feedback);
backend 'dh'       = DH PGE over Modbus (position/force + held/not-held feedback)."""


class MotionService:
    def __init__(self, node, zones_cfg, logger=None):
        self.node = node
        self.robot = node.movement.robot
        self.logger = logger
        self.zones = zones_cfg
        g = zones_cfg.get('gripper', {})
        self.backend = g.get('backend', 'ufactory')
        self.axis_offset = float(g.get('axis_offset_rad', 1.5708))
        self._dh = None
        self._dh_cfg = g.get('dh', {})

    # ---------------- arm ----------------
    async def move_to(self, x, y, z, yaw=None):
        from ..movement.robot_hardware import MoveResult  # lazy: ROS only at runtime
        r = await self.robot.move_to_pose(float(x), float(y), float(z), yaw)
        return r == MoveResult.SUCCESS

    async def park(self):
        p = self.zones.get('park')
        if not p:
            return True
        return await self.move_to(p['x'], p['y'], p['z'])

    # ---------------- gripper ----------------
    def _dh_gripper(self):
        if self._dh is None:
            from ..hardware.dh_gripper import DHGripper
            self._dh = DHGripper(port=self._dh_cfg.get('port', '/dev/ttyUSB0'),
                                 slave=int(self._dh_cfg.get('slave', 1)),
                                 baudrate=int(self._dh_cfg.get('baud', 115200)),
                                 stroke_mm=float(self._dh_cfg.get('stroke_mm', 26.0)),
                                 logger=self.logger)
            self._dh.initialize()
        return self._dh

    async def gripper_open(self):
        if self.backend == 'dh':
            self._dh_gripper().open()
            return True
        return await self.robot.control_gripper(close=False)

    async def gripper_close(self, force_pct=None):
        """-> (ok, holding). holding is None when the backend has no feedback."""
        if self.backend == 'dh':
            from ..hardware.dh_gripper import GripState
            st = self._dh_gripper().close(
                force_pct=force_pct or self._dh_cfg.get('default_force_pct', 30))
            return True, st == GripState.GRIPPED
        ok = await self.robot.control_gripper(close=True)
        return ok, None

    async def gripper_idle(self):
        if self.backend == 'dh':
            return True
        return await self.robot.stop_gripper()

    def grasp_yaw(self, object_yaw):
        """Object long-axis angle -> commanded gripper yaw (fingers across the short axis)."""
        import math
        y = object_yaw + self.axis_offset
        while y > math.pi:
            y -= 2 * math.pi
        while y < -math.pi:
            y += 2 * math.pi
        return y
