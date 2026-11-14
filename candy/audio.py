import logging
import subprocess
import threading
import time
import wave
from collections import deque

log = logging.getLogger(__name__)


class AudioTap:
    """Keeps the last few minutes of microphone audio in memory so clips can include it."""

    RATE = 16000
    CHUNK = 1600  # samples, 0.1 s

    def __init__(self, cfg, keep_s):
        self.keep_s = keep_s
        self._chunks = deque()  # (start time, raw 16-bit mono bytes)
        self._lock = threading.Lock()
        self._proc = subprocess.Popen(
            ["arecord", "-q", "-D", cfg.device, "-f", "S16_LE", "-r", str(self.RATE), "-c", "1", "-t", "raw"],
            stdout=subprocess.PIPE)
        threading.Thread(target=self._read, daemon=True).start()
        log.info("recording audio from %s", cfg.device)

    def _read(self):
        while True:
            data = self._proc.stdout.read(self.CHUNK * 2)
            if not data:
                log.warning("microphone stopped (arecord exit code %s); clips continue without audio",
                            self._proc.poll())
                return
            now = time.monotonic()
            with self._lock:
                self._chunks.append((now - len(data) / 2 / self.RATE, data))
                while now - self._chunks[0][0] > self.keep_s:
                    self._chunks.popleft()

    def save(self, path, t0, t1):
        """Write the audio between two monotonic times to a WAV file. Returns False if there's none."""
        dur = self.CHUNK / self.RATE
        with self._lock:
            data = b"".join(d for t, d in self._chunks if t + dur >= t0 and t <= t1)
        if not data:
            return False
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.RATE)
            w.writeframes(data)
        return True

    def close(self):
        self._proc.terminate()