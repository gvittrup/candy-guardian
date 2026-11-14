import logging
import queue
import threading
import time
import subprocess
from collections import deque
from datetime import datetime

import cv2

log = logging.getLogger(__name__)


class Recorder:
    """Saves a clip from just before someone appears until they've been gone a few seconds."""

    def __init__(self, cfg, out_dir, fps, audio=None):
        self.cfg = cfg
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.audio = audio
        self._pre = deque()  # (time, frame) from the last pre_roll_s
        self._q = queue.Queue(maxsize=int(fps * 5))
        self._start = None
        self._last_person = None
        self._cleanup()
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()

        if cfg.convert:
            for p in self.out_dir.glob(f"*{cfg.ext}"):
                threading.Thread(target=self._convert, args=(p,), daemon=True).start()

    def update(self, frame, now, present):
        if self._start is None:
            self._pre.append((now, frame))
            while now - self._pre[0][0] > self.cfg.pre_roll_s:
                self._pre.popleft()
            if not present:
                return
            self._start = self._last_person = now
            self._q.put(("open", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), frame.shape))
            for t, f in self._pre:
                self._send(("frame", t, f))
            self._pre.clear()
            return
        if present:
            self._last_person = now
        self._send(("frame", now, frame))
        if now - self._last_person > self.cfg.post_roll_s or now - self._start > self.cfg.max_clip_s:
            self._q.put(("close",))
            self._start = None

    def close(self):
        self._q.put(("close",))
        self._q.put(("stop",))
        self._thread.join(timeout=5)

    def _send(self, item):
        try:
            self._q.put_nowait(item)
        except queue.Full:
            pass  # disk can't keep up; drop the frame rather than slow detection

    def _writer(self):
        out, path, t0, t_last, written = None, None, None, None, 0
        while True:
            item = self._q.get()
            if item[0] == "open":
                h, w = item[2][:2]
                path = self.out_dir / f"{item[1]}{self.cfg.ext}"
                out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*self.cfg.codec), self.fps, (w, h))
                t0, written = None, 0
                log.info("recording %s", path.name)
            elif item[0] == "frame" and out:
                _, t, frame = item
                t0 = t if t0 is None else t0
                t_last = t
                # repeat frames so the clip plays at real speed even when the loop runs slow
                while written < int((t - t0) * self.fps) + 1:
                    out.write(frame)
                    written += 1
            elif item[0] == "close" and out:
                out.release()
                out = None
                if self.audio and t0 is not None:
                    self.audio.save(path.with_suffix(".wav"), t0, t_last)
                log.info("saved %s (%.0f s)", path.name, written / self.fps)
                if self.cfg.convert:
                    threading.Thread(target=self._convert, args=(path,), daemon=True).start()
            elif item[0] == "stop":
                return

    def _cleanup(self):
        cutoff = time.time() - self.cfg.keep_days * 86400
        for p in self.out_dir.iterdir():
            if p.name.endswith(".tmp") or (p.suffix in (".avi", ".mp4", ".wav") and p.stat().st_mtime < cutoff):
                p.unlink()

    def _convert(self, src):
        """Re-encode to H.264 MP4 (with audio if there is any) so browsers can play it."""
        tmp = src.parent / (src.stem + ".mp4.tmp")
        wav = src.with_suffix(".wav")
        has_audio = wav.exists()
        cmd = ["nice", "-n", "19", "ffmpeg", "-loglevel", "error", "-y", "-i", str(src)]
        if has_audio:
            cmd += ["-i", str(wav), "-c:a", "aac", "-b:a", "64k"]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-threads", "2", "-f", "mp4", str(tmp)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(src.with_suffix(".mp4"))
            src.unlink()
            wav.unlink(missing_ok=True)
            log.info("converted %s%s", src.with_suffix(".mp4").name, " with audio" if has_audio else "")
        else:
            tmp.unlink(missing_ok=True)
            log.warning("couldn't convert %s: %s", src.name, r.stderr.strip()[-300:])