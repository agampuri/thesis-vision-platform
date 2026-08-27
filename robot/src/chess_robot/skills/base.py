"""The skill contract (general_platform_plan.md §4.1). Deliberately minimal."""
from enum import Enum


class SkillStatus(Enum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Skill:
    name = "base"

    def can_handle(self, intent) -> float:
        """0..1 confidence that this skill should handle the intent."""
        return 0.0

    async def start(self, intent, services):
        self.intent = intent
        self.s = services

    async def step(self) -> SkillStatus:
        raise NotImplementedError

    async def stop(self):
        """ALWAYS-SAFE: open gripper, retreat to park. Called on any exit path."""
        try:
            await self.s.motion.gripper_open()
        except Exception:
            pass
        try:
            await self.s.motion.park()
        except Exception:
            pass
