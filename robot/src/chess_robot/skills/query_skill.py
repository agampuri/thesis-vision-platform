"""S6 — perception Q&A: 'how many red cubes?', 'where is the marker?' (+ pointing gesture)."""
from .base import Skill, SkillStatus
from ..platform.grasp_planner import bbox_center_base


class QuerySkill(Skill):
    name = "query"

    def can_handle(self, intent):
        return 0.9 if intent.action in ('query_count', 'query_find') else 0.0

    async def step(self):
        frame = await self.s.look()
        if frame is None:
            return SkillStatus.FAILED
        bgr, depth, K = frame
        query, _ = self.s.router.resolve_object(self.intent.object_query)
        self.s.open_vocab.set_classes([query])
        dets = self.s.open_vocab.detect(bgr)
        located = []
        for label, conf, bbox in dets:
            p = bbox_center_base(depth, K, bbox, self.s.vision.extrinsics)
            if p is not None and self.s.safety.check_point(p[0], p[1])[0]:
                located.append((label, conf, p))
        if self.intent.action == 'query_count':
            answer = f"I can see {len(located)} x '{query}' on the table."
            print(f"\n  {answer}\n")
            self.s.logger.info(answer)
            return SkillStatus.DONE
        if not located:
            print(f"\n  I cannot find '{query}' on the table.\n")
            return SkillStatus.DONE
        label, conf, p = located[0]
        answer = f"'{query}' is at x={p[0]*100:.1f} cm, y={p[1]*100:.1f} cm — pointing at it."
        print(f"\n  {answer}\n")
        await self.s.motion.move_to(p[0], p[1], p[2] + 0.12)
        import asyncio
        await asyncio.sleep(1.2)
        return SkillStatus.DONE
