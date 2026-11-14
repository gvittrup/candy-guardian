from dataclasses import dataclass

import time
import numpy as np
from ultralytics import YOLO

from .config import resolve

# COCO keypoint indices
L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 7, 8, 9, 10


@dataclass
class Person:
    track_id: int
    box: tuple   # x1, y1, x2, y2 in frame pixels
    hands: list  # [left, right], each (x, y) or None


class Detector:
    def __init__(self, cfg, grab_cfg):
        self.model = YOLO(str(resolve(cfg.model)), task="pose")
        self.imgsz = cfg.imgsz
        self.conf = cfg.conf
        self.kp_conf = grab_cfg.keypoint_conf
        self.extend = grab_cfg.hand_extend

    def __call__(self, frame):

        start = time.perf_counter()
        r = self.model.track(frame, persist=True, tracker="bytetrack.yaml",
                             imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        print(f"inference {(time.perf_counter()-start)*1000:.1f} ms")
        if r.boxes is None or r.boxes.id is None:
            return []
        ids = r.boxes.id.int().tolist()
        boxes = r.boxes.xyxy.cpu().numpy()
        kxy = r.keypoints.xy.cpu().numpy()
        kconf = r.keypoints.conf
        kconf = kconf.cpu().numpy() if kconf is not None else np.ones(kxy.shape[:2])

        people = []
        for i, tid in enumerate(ids):
            hands = [self._hand(kxy[i], kconf[i], w, e)
                     for w, e in ((L_WRIST, L_ELBOW), (R_WRIST, R_ELBOW))]
            people.append(Person(tid, tuple(boxes[i]), hands))
            
        return people

    def _hand(self, xy, conf, wrist, elbow):
        if conf[wrist] < self.kp_conf:
            return None
        w = xy[wrist]
        if conf[elbow] >= self.kp_conf:
            w = w + self.extend * (w - xy[elbow])
        return float(w[0]), float(w[1])