"""S2 — language-directed pick & place: 'put the red cube in the blue bin'."""
from .base import Skill, SkillStatus
from ..platform.grasp_planner import bbox_center_base


class PickPlaceSkill(Skill):
    name = "pick_place"

    def can_handle(self, intent):
        return 0.9 if intent.action in ('pick', 'pick_place') else 0.0

    async def start(self, intent, services):
        await super().start(intent, services)
        self.stage = 'look'
        self.grasp = None
        self.obj_info = None
        self.target = None

    def _resolve_target(self):
        z = self.s.zones
        name = self.intent.target_zone
        if name and name in (z.get('zones') or {}):
            return z['zones'][name]
        return z.get('drop_default')

    async def step(self):
        log = self.s.logger
        if self.stage == 'look':
            frame = await self.s.look()
            if frame is None:
                log.error("pick_place: no vision frame/board pose")
                return SkillStatus.FAILED
            bgr, depth, K = frame
            query, self.obj_info = self.s.router.resolve_object(self.intent.object_query)
            self.s.open_vocab.set_classes([query])
            dets = self.s.open_vocab.detect(bgr)
            dets = [d for d in dets if self._in_workspace(depth, K, d[2])]
            if not dets:
                log.error(f"pick_place: '{query}' not found on the table")
                return SkillStatus.FAILED
            label, conf, bbox = dets[0]
            log.info(f"pick_place: found '{label}' conf={conf:.2f}")
            self.grasp, why = self.s.grasp.synthesize(depth, K, bbox, self.obj_info)
            if self.grasp is None:
                log.error(f"pick_place: grasp synthesis failed: {why}")
                return SkillStatus.FAILED
            ok, why = self.s.safety.check_grasp(self.grasp, self.obj_info)
            if not ok:
                log.error(f"pick_place: safety rejected grasp: {why}")
                return SkillStatus.FAILED
            self.target = self._resolve_target()
            if self.target is None:
                log.error("pick_place: no target zone configured")
                return SkillStatus.FAILED
            ok, why = self.s.safety.check_point(self.target['x'], self.target['y'])
            if not ok:
                log.error(f"pick_place: target rejected: {why}")
                return SkillStatus.FAILED
            self.stage = 'pick'
            return SkillStatus.RUNNING

        if self.stage == 'pick':
            g = self.grasp
            yaw = self.s.motion.grasp_yaw(g.yaw)
            hover = g.top_z + 0.10
            if not await self.s.motion.gripper_open():
                return SkillStatus.FAILED
            if not await self.s.motion.move_to(g.x, g.y, hover, yaw):
                return SkillStatus.FAILED
            if not await self.s.motion.move_to(g.x, g.y, g.z, yaw):
                return SkillStatus.FAILED
            force = (self.obj_info or {}).get('force_pct')
            ok, holding = await self.s.motion.gripper_close(force_pct=force)
            if not ok:
                return SkillStatus.FAILED
            if holding is False:
                log.error("pick_place: gripper reports NO object held — aborting")
                await self.s.motion.gripper_open()
                await self.s.motion.move_to(g.x, g.y, hover)
                return SkillStatus.FAILED
            if not await self.s.motion.move_to(g.x, g.y, hover, yaw):
                return SkillStatus.FAILED
            self.stage = 'place'
            return SkillStatus.RUNNING

        if self.stage == 'place':
            t = self.target
            travel_z = max(self.grasp.top_z + 0.12, 0.18)
            if not await self.s.motion.move_to(t['x'], t['y'], travel_z):
                return SkillStatus.FAILED
            rel_z = self.s.safety.table_z + float(t.get('z_release', 0.06))
            if not await self.s.motion.move_to(t['x'], t['y'], rel_z):
                return SkillStatus.FAILED
            await self.s.motion.gripper_open()
            await self.s.motion.move_to(t['x'], t['y'], travel_z)
            await self.s.motion.gripper_idle()
            log.info("pick_place: done")
            return SkillStatus.DONE

        return SkillStatus.FAILED

    def _in_workspace(self, depth, K, bbox):
        p = bbox_center_base(depth, K, bbox, self.s.vision.extrinsics)
        if p is None:
            return False
        return self.s.safety.check_point(p[0], p[1])[0]
