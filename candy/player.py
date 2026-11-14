import logging
import subprocess

log = logging.getLogger(__name__)


class Player:
    def __init__(self, cfg):
        self.backend = cfg.backend
        self.device = cfg.device
        self.cooldown = cfg.global_cooldown_s
        self._proc = None
        self._last = float("-inf")
        self.kind = None  # what's playing: "kid", "greet", "stolen" or "returned"

    def busy(self):
        return self._proc is not None and self._proc.poll() is None

    def playing(self, kind):
        return self.busy() and self.kind == kind

    def ready(self, now, over_greeting=False):
        """Free to speak: nothing playing (or only a greeting, if allowed) and the cooldown is over."""
        free = not self.busy() or (over_greeting and self.kind == "greet")
        return free and now - self._last >= self.cooldown

    def play(self, wav, now, interrupt=False, kind="kid"):
        if interrupt and self.busy():
            self._proc.terminate()
        if kind != "greet":
            self._last = now  # greetings don't start the cooldown, so a scold can follow right away
        self.kind = kind
        if self.backend == "none":
            log.info("[no speaker] would play %s", wav.name)
            return
        self._proc = subprocess.Popen(["aplay", "-q", "-D", self.device, str(wav)])