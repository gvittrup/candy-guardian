import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


class BowlWatch:
    """Detects the bowl being removed by comparing its area to a photo of the empty spot."""

    def __init__(self, cfg, bowl, ref_path):
        self.cfg = cfg
        self.bowl = bowl
        self.ref_path = ref_path
        self.ref = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE) if ref_path.exists() else None
        self.diff = None  # latest score, shown on the debug stream
        self.missing = False
        self._empty_since = None
        self._present_since = None
        self._last_alarm = float("-inf")
        self._alarms = 0

    def capture_empty(self, frame):
        self.ref = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(self.ref_path), self.ref)
        log.info("empty-spot photo saved")

    def _score(self, frame):
        """Mean pixel difference between the bowl area now and the empty-spot photo."""
        poly = self.bowl.poly
        x, y, w, h = cv2.boundingRect(poly)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [poly - (x, y)], 255)
        m = mask > 0
        cur = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        cur = cv2.GaussianBlur(cur, (9, 9), 0).astype(np.float32)[m]
        ref = cv2.GaussianBlur(self.ref[y:y + h, x:x + w], (9, 9), 0).astype(np.float32)[m]
        # subtract each mean so porch-light or exposure changes don't count as a difference
        return float(np.mean(np.abs((cur - cur.mean()) - (ref - ref.mean()))))

    def update(self, frame, now, alarm_playing=False):
        """Returns "stolen" when an alarm should play, "returned" once when the bowl is back, else None."""
        if not self.cfg.enabled or self.ref is None or self.bowl.poly is None:
            return None
        if self.ref.shape != frame.shape[:2]:
            log.warning("empty-spot photo size doesn't match the camera; capture it again")
            self.ref = None
            return None

        self.diff = self._score(frame)
        empty = self.diff < self.cfg.empty_diff_max
        self._empty_since = (self._empty_since or now) if empty else None
        self._present_since = None if empty else (self._present_since or now)

        if not self.missing:
            if not (empty and now - self._empty_since >= self.cfg.missing_after_s):
                return None
            self.missing, self._alarms, self._last_alarm = True, 0, float("-inf")
            log.warning("bowl missing (diff %.1f)", self.diff)

        if not empty and now - self._present_since >= self.cfg.return_after_s:
            self.missing = False
            log.info("bowl returned (diff %.1f)", self.diff)
            return "returned"
        if alarm_playing:
            self._last_alarm = now  # the repeat gap starts when the current line ends
            return None
        if empty and now - self._last_alarm >= self.cfg.repeat_s and self._alarms < self.cfg.max_alarms:
            self._last_alarm = now
            self._alarms += 1
            return "stolen"
        return None