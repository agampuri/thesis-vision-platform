"""Shared services handed to every skill."""


class Services:
    def __init__(self, node=None, vision=None, motion=None, grasp=None,
                 safety=None, router=None, open_vocab=None,
                 zones=None, objects=None, logger=None):
        self.node = node
        self.vision = vision
        self.motion = motion
        self.grasp = grasp
        self.safety = safety
        self.router = router
        self.open_vocab = open_vocab
        self.zones = zones or {}
        self.objects = objects or {}
        self.logger = logger

    async def look(self):
        """Park the arm and take a fresh, unoccluded look. -> (bgr, depth, K) | None."""
        await self.motion.park()
        if not self.vision.update():
            return None
        return self.vision.get_raw_frame()
