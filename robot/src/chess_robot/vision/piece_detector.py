"""
chess_robot.vision.piece_detector — YOLO inference + snapping detections to squares.

The live system uses the 2-class occupancy model (white_piece / black_piece);
the 12-class model is for evaluation. Color is derived from the class name
prefix, so both models work here.
"""
import numpy as np


class PieceDetector:
    def __init__(self, model_path, conf=0.40, logger=None):
        self.model_path = model_path
        self.conf = float(conf)
        self.logger = logger
        self._model = None  # lazy

    def _ensure_model(self):
        if self._model is None:
            from ultralytics import YOLO  # lazy: heavy import
            self._model = YOLO(self.model_path)
            if self.logger:
                self.logger.info(f"Loaded detector {self.model_path} "
                                 f"classes={list(self._model.names.values())}")

    def detect(self, bgr):
        """-> list of (class_name, conf, cx_px, cy_px)."""
        self._ensure_model()
        res = self._model.predict(bgr, conf=self.conf, verbose=False)[0]
        out = []
        names = res.names
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append((names[int(b.cls[0])], float(b.conf[0]),
                        (x1 + x2) / 2.0, (y1 + y2) / 2.0))
        return out


def color_of(class_name):
    n = class_name.lower()
    if n.startswith('white'):
        return 'w'
    if n.startswith('black'):
        return 'b'
    return None


def occupancy_from_detections(detections, squares_px, snap_radius_px):
    """Snap pixel-space detections to the nearest square centre.

    detections: [(class_name, conf, cx, cy)]
    squares_px: {'a1': (u, v), ...} from BoardPose
    snap_radius_px: max distance for a valid snap (≈ 0.45 * pitch_px)

    Returns (occupancy {'e4': 'w'|'b'}, unsnapped_count).
    unsnapped_count > 0 usually means a hand or a piece in mid-air is in frame.
    """
    names = list(squares_px.keys())
    centers = np.array([squares_px[n] for n in names], dtype=float)
    occupancy, unsnapped = {}, 0
    best_conf = {}
    for cname, conf, cx, cy in detections:
        col = color_of(cname)
        if col is None:
            continue
        d = np.linalg.norm(centers - np.array([cx, cy]), axis=1)
        i = int(np.argmin(d))
        if d[i] > snap_radius_px:
            unsnapped += 1
            continue
        sq = names[i]
        if conf >= best_conf.get(sq, -1.0):  # keep highest-confidence per square
            best_conf[sq] = conf
            occupancy[sq] = col
    return occupancy, unsnapped
