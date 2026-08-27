"""S3 — sort & tidy: detect every catalogue object, route each to its bin."""
from .base import Skill, SkillStatus
from .pick_place_skill import PickPlaceSkill
from ..platform.intent_router import Intent
from ..platform.grasp_planner import bbox_center_base


class SortSkill(Skill):
    name = "sort"

    def can_handle(self, intent):
        return 0.9 if intent.action == 'sort' else 0.0

    async def start(self, intent, services):
        await super().start(intent, services)
        self.queue = None
        self.done_n, self.fail_n = 0, 0

    def _zone_for(self, label):
        rules = self.s.zones.get('sort_rules', {})
        for key, zone in rules.items():
            if key != 'default' and key in label.lower():
                return zone
        return rules.get('default', '')

    async def _scan(self):
        frame = await self.s.look()
        if frame is None:
            return None
        bgr, depth, K = frame
        queries = sorted({o.get('query', n.replace('_', ' '))
                          for n, o in (self.s.objects.get('objects') or {}).items()})
        if not queries:
            self.s.logger.error("sort: objects.yaml has no catalogue entries")
            return None
        self.s.open_vocab.set_classes(queries)
        dets = self.s.open_vocab.detect(bgr)
        queue = []
        for label, conf, bbox in dets:
            p = bbox_center_base(depth, K, bbox, self.s.vision.extrinsics)
            if p is None or not self.s.safety.check_point(p[0], p[1])[0]:
                continue
            if self._inside_a_bin(p):
                continue
            queue.append(label)
        return queue

    def _inside_a_bin(self, p):
        for z in (self.s.zones.get('zones') or {}).values():
            if abs(p[0] - z['x']) < 0.06 and abs(p[1] - z['y']) < 0.06:
                return True
        return False

    async def step(self):
        if self.queue is None:
            self.queue = await self._scan()
            if self.queue is None:
                return SkillStatus.FAILED
            self.s.logger.info(f"sort: {len(self.queue)} objects to sort: {self.queue}")
            return SkillStatus.RUNNING
        if not self.queue:
            self.s.logger.info(f"sort: finished — {self.done_n} sorted, {self.fail_n} failed")
            return SkillStatus.DONE
        label = self.queue.pop(0)
        sub = PickPlaceSkill()
        sub_intent = Intent('pick_place', object_query=label,
                            target_zone=self._zone_for(label), raw=f"sort:{label}")
        await sub.start(sub_intent, self.s)
        status = SkillStatus.RUNNING
        for _ in range(20):
            status = await sub.step()
            if status != SkillStatus.RUNNING:
                break
        if status == SkillStatus.DONE:
            self.done_n += 1
        else:
            self.fail_n += 1
            self.s.logger.warning(f"sort: '{label}' failed; continuing with the rest")
        return SkillStatus.RUNNING
