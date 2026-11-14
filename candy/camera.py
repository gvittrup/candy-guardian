import logging
import threading
import time

import cv2

log = logging.getLogger(__name__)


class Camera:
    """Reads frames on a background thread and keeps only the newest one."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._frame = None
        self._ts = 0.0
        self._lock = threading.Lock()
        self._running = True
        self.cap = self._open()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _open(self):
        cap = cv2.VideoCapture(self.cfg.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera {self.cfg.device}")
        log.info("camera open: %dx%d @ %.0f fps",
                 cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                 cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                 cap.get(cv2.CAP_PROP_FPS))
        return cap

    def _loop(self):
        failures = 0
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                failures += 1
                if failures == 30:  # about a second of failed reads: reopen
                    log.warning("camera read failing, reopening")
                    self.cap.release()
                    time.sleep(1)
                    try:
                        self.cap = self._open()
                    except RuntimeError as e:
                        log.error("%s", e)
                    failures = 0
                time.sleep(0.03)
                continue
            failures = 0
            with self._lock:
                self._frame, self._ts = frame, time.monotonic()

    def read(self):
        """Return (newest frame, its timestamp); frame is None until the first read."""
        with self._lock:
            return self._frame, self._ts

    def close(self):
        self._running = False
        self._thread.join(timeout=1)
        self.cap.release()