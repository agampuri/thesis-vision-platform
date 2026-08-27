"""YOLO-World wrapper: detect arbitrary objects from a text description.
Heavy imports are lazy; the model downloads on first use (needs internet once)."""


class OpenVocabDetector:
    def __init__(self, model_name='yolov8s-world.pt', conf=0.25, logger=None):
        self.model_name = model_name
        self.conf = float(conf)
        self.logger = logger
        self._model = None
        self._classes = None

    def _ensure(self):
        if self._model is None:
            from ultralytics import YOLOWorld  # lazy
            self._model = YOLOWorld(self.model_name)
            if self.logger:
                self.logger.info(f"YOLO-World loaded: {self.model_name}")

    def set_classes(self, names):
        self._ensure()
        names = [str(n) for n in names]
        if names != self._classes:
            self._model.set_classes(names)
            self._classes = names

    def detect(self, bgr):
        """-> [(label, conf, (x1, y1, x2, y2))] sorted by confidence desc."""
        self._ensure()
        res = self._model.predict(bgr, conf=self.conf, verbose=False)[0]
        out = []
        for b in res.boxes:
            out.append((res.names[int(b.cls[0])], float(b.conf[0]),
                        tuple(b.xyxy[0].tolist())))
        return sorted(out, key=lambda d: -d[1])
