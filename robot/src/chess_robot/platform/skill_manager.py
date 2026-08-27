"""Routes intents to skills and drives the skill life cycle with always-safe stop."""
from ..skills.base import SkillStatus


class SkillManager:
    def __init__(self, logger=None):
        self.skills = []
        self.logger = logger

    def register(self, skill):
        self.skills.append(skill)

    def select(self, intent):
        scored = sorted(((s.can_handle(intent), s) for s in self.skills),
                        key=lambda t: -t[0])
        if not scored or scored[0][0] < 0.5:
            return None
        return scored[0][1]

    async def run(self, skill, intent, services, max_steps=200):
        status = SkillStatus.FAILED
        try:
            await skill.start(intent, services)
            for _ in range(max_steps):
                status = await skill.step()
                if status != SkillStatus.RUNNING:
                    break
        except KeyboardInterrupt:
            if self.logger:
                self.logger.warning(f"{skill.name}: interrupted by user")
            status = SkillStatus.FAILED
        except Exception as e:
            if self.logger:
                self.logger.error(f"{skill.name} crashed: {e}")
            status = SkillStatus.FAILED
        finally:
            try:
                await skill.stop()
            except Exception:
                pass
        if self.logger:
            self.logger.info(f"{skill.name} -> {status.value}")
        return status
