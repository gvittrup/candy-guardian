import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


class Bowl:
    """Polygon around the candy bowl, in frame pixels."""

    def __init__(self, path, margin_px):
        self.path = Path(path)
        self.margin = margin_px
        self.poly = None
        if self.path.exists():
            self.poly = np.array(json.loads(self.path.read_text()), np.int32)

    def set(self, points):
        self.path.write_text(json.dumps(points))
        self.poly = np.array(points, np.int32)
        log.info("bowl outline saved: %s", points)

    def contains(self, pt, was_inside=False):
        poly = self.poly
        if poly is None or pt is None:
            return False
        d = cv2.pointPolygonTest(poly, pt, True)  # signed distance, > 0 inside
        return d >= (-self.margin if was_inside else 0)


@dataclass
class TrackState:
    first_seen: float
    last_seen: float
    grabs: int = 0
    dwell: float = 0.0
    dip_start: float | None = None
    last_inside: float = 0.0
    dip_hands: set = field(default_factory=set)
    flagged: bool = False

    def in_dip(self):
        return self.dip_start is not None


class GrabJudge:
    def __init__(self, cfg, bowl):
        self.cfg = cfg
        self.bowl = bowl
        self.tracks = {}

    def update(self, people, now):
        """Feed one frame of detections. Returns (newly flagged ids, expired ids)."""
        for p in people:
            st = self.tracks.get(p.track_id)
            if st is None:
                st = self.tracks[p.track_id] = TrackState(now, now)
            st.last_seen = now
            inside = {i for i, h in enumerate(p.hands)
                      if self.bowl.contains(h, was_inside=st.in_dip())}
            if inside:
                if not st.in_dip():
                    st.dip_start, st.dip_hands = now, set()
                st.dip_hands |= inside
                st.last_inside = now

        flagged, expired = [], []
        for tid, st in list(self.tracks.items()):
            self._maybe_close_dip(st, now)
            if not st.flagged and self._too_much(st, now):
                st.flagged = True
                flagged.append(tid)
            if now - st.last_seen > self.cfg.track_timeout_s:
                del self.tracks[tid]
                expired.append(tid)
        return flagged, expired

    def _maybe_close_dip(self, st, now):
        if st.in_dip() and now - st.last_inside > self.cfg.merge_gap_s:
            duration = st.last_inside - st.dip_start
            if duration >= self.cfg.min_dip_s:
                st.grabs += len(st.dip_hands)
                st.dwell += duration
            st.dip_start = None

    def _too_much(self, st, now):
        open_dwell = now - st.dip_start if st.in_dip() else 0.0
        return st.grabs > self.cfg.max_grabs or st.dwell + open_dwell > self.cfg.max_dwell_s