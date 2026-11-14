import logging
import queue
import threading

import cv2
import open_clip
import torch
from PIL import Image

log = logging.getLogger(__name__)

NONE = "none"


class CostumeClassifier:
    def __init__(self, cfg, labels):
        self.cfg = cfg
        torch.set_num_threads(2)  # leave the other cores for YOLO
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
        self.model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")

        self.labels = list(labels) + [NONE]
        prompts = [f"a photo of a person wearing a {l} costume" for l in labels]
        prompts.append("a photo of a person wearing everyday clothes")
        with torch.no_grad():
            t = self.model.encode_text(tokenizer(prompts))
            self.text = t / t.norm(dim=-1, keepdim=True)

        self.results = {}    # track id -> (label, confidence)
        self._attempts = {}  # track id -> crops submitted
        self._last_submit = {}
        self._q = queue.Queue(maxsize=4)
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, tid, frame, box, now):
        if self._attempts.get(tid, 0) >= self.cfg.attempts:
            return
        if now - self._last_submit.get(tid, float("-inf")) < self.cfg.submit_every_s:
            return
        x1, y1, x2, y2 = (int(v) for v in box)
        if y2 - y1 < self.cfg.min_box_height:
            return
        crop = frame[max(0, y1):y2, max(0, x1):x2].copy()
        try:
            self._q.put_nowait((tid, crop))
        except queue.Full:
            return
        self._attempts[tid] = self._attempts.get(tid, 0) + 1
        self._last_submit[tid] = now

    def _loop(self):
        while True:
            tid, crop = self._q.get()
            img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            with torch.no_grad():
                f = self.model.encode_image(self.preprocess(img).unsqueeze(0))
                f = f / f.norm(dim=-1, keepdim=True)
                probs = (100 * f @ self.text.T).softmax(dim=-1)[0]
            i = int(probs.argmax())
            label, conf = self.labels[i], float(probs[i])
            best = self.results.get(tid)
            if best is None or conf > best[1]:
                self.results[tid] = (label, conf)
                log.info("track %d looks like: %s (%.2f)", tid, label, conf)

    def get(self, tid):
        """Costume label if confident, else None."""
        r = self.results.get(tid)
        if r and r[0] != NONE and r[1] >= self.cfg.min_confidence:
            return r[0]
        return None

    def forget(self, tid):
        for d in (self.results, self._attempts, self._last_submit):
            d.pop(tid, None)