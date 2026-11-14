import logging
import threading
import time
import signal

import cv2

from . import config
from .bowlwatch import BowlWatch
from .camera import Camera
from .detector import Detector
from .grabs import Bowl, GrabJudge
from .lines import COSTUMES, GREETINGS, RETURNED, STOLEN, LineBook
from .player import Player
from .stream import DebugStream
from .voice import VoiceBank
from .recorder import Recorder

log = logging.getLogger("candy")

ORANGE, GREEN, RED, CYAN = (0, 165, 255), (0, 200, 0), (0, 0, 255), (255, 255, 0)


def annotate(frame, bowl, watch, people, judge, costumes, fps):
    img = frame.copy()
    if bowl.poly is not None:
        cv2.polylines(img, [bowl.poly], True, RED if watch.missing else ORANGE, 2)
    for p in people:
        st = judge.tracks.get(p.track_id)
        color = RED if st and st.flagged else GREEN
        x1, y1, x2, y2 = (int(v) for v in p.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"#{p.track_id}"
        if st:
            label += f" grabs {st.grabs} dwell {st.dwell:.1f}s"
        costume = costumes.get(p.track_id) if costumes else None
        if costume:
            label += f" {costume}"
        cv2.putText(img, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        for h in p.hands:
            if h:
                c = RED if bowl.contains(h) else CYAN
                cv2.circle(img, (int(h[0]), int(h[1])), 6, c, -1)

    status = f"{fps:.1f} fps"
    if watch.ref is None:
        status += "  no empty-spot photo"
    elif watch.diff is not None:
        status += f"  bowl diff {watch.diff:.1f}"
    cv2.putText(img, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ORANGE, 2)
    if watch.missing:
        cv2.putText(img, "BOWL MISSING", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, RED, 3)
    return img


def _stop(*_):
    raise KeyboardInterrupt  # lets the finally block close the open clip


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = config.load()

    book = LineBook(cfg.costume.enabled)
    voices = VoiceBank(cfg.voice)
    texts = book.texts()
    log.info("pre-rendering %d lines (cached after the first run)", len(texts))
    voices.prerender(texts)

    bowl = Bowl(config.resolve(cfg.bowl.roi_file), cfg.bowl.margin_px)
    if bowl.poly is None:
        log.warning("no bowl outline yet: open the debug page and draw one")
    watch = BowlWatch(cfg.bowl_watch, bowl, config.resolve(cfg.bowl_watch.ref_file))
    if cfg.bowl_watch.enabled and watch.ref is None:
        log.warning("no empty-spot photo yet: remove the bowl and press Capture empty spot")
    judge = GrabJudge(cfg.grabs, bowl)
    detector = Detector(cfg.detector, cfg.grabs)
    player = Player(cfg.player)
    costumes = None
    if cfg.costume.enabled:
        from .costume import CostumeClassifier  # heavy import; skip when disabled
        costumes = CostumeClassifier(cfg.costume, list(COSTUMES))
    capture_empty = threading.Event()  # set by the web page, handled on the next frame
    clips_dir = config.resolve(cfg.recording.dir) if cfg.recording.enabled else None
    stream = DebugStream(cfg.stream, bowl.set, capture_empty.set, clips_dir) if cfg.stream.enabled else None    
    
    cam = Camera(cfg.camera)
    audio = None
    recorder = None
    if cfg.recording.enabled:
        if cfg.audio.enabled:
            from .audio import AudioTap
            rc = cfg.recording
            audio = AudioTap(cfg.audio, rc.max_clip_s + rc.pre_roll_s + rc.post_roll_s + 10)
        recorder = Recorder(cfg.recording, config.resolve(cfg.recording.dir), cfg.detector.target_fps, audio)

    period = 1.0 / cfg.detector.target_fps
    pending = {}  # track id -> time flagged
    seen = {}  # track id -> time first seen
    greeted = set()  # track ids already covered by a greeting
    last_greet = float("-inf")
    last_ts, fps = 0.0, 0.0
    log.info("running")
    try:
        while True:
            t0 = time.monotonic()
            frame, ts = cam.read()
            if frame is None or ts == last_ts:
                time.sleep(0.005)
                continue
            last_ts = ts

            if capture_empty.is_set():
                watch.capture_empty(frame)
                capture_empty.clear()

            people = detector(frame)
            now = time.monotonic()
            if costumes:
                for p in people:
                    costumes.submit(p.track_id, frame, p.box, now)

            flagged, expired = judge.update(people, now)
            for tid in flagged:
                st = judge.tracks[tid]
                log.info("track %d: too much candy (grabs=%d, dwell=%.1fs)", tid, st.grabs, st.dwell)
                pending[tid] = now
            for tid in expired:
                seen.pop(tid, None)
                greeted.discard(tid)
                if costumes:
                    costumes.forget(tid)

            # bowl theft beats everything, but an alarm never cuts off another alarm
            event = watch.update(frame, now, alarm_playing=player.playing("stolen"))
            if event:
                text = book.pick_from(STOLEN if event == "stolen" else RETURNED)
                log.info("bowl %s: %s", event, text)
                player.play(voices.render(text), now, interrupt=True, kind=event)
                pending.clear()

            # a scold may cut off a greeting, nothing else
            for tid, at in list(pending.items()):
                if now - at > cfg.player.pending_s:
                    del pending[tid]
                elif not watch.missing and player.ready(now, over_greeting=True):
                    costume = costumes.get(tid) if costumes else None
                    text = book.pick(costume)
                    log.info("speaking to track %d (%s): %s", tid, costume or "generic", text)
                    player.play(voices.render(text), now, interrupt=True, kind="kid")
                    del pending[tid]
                    break

            # greet new arrivals after a short delay, once per group
            if cfg.greeting.enabled:
                g = cfg.greeting
                for p in people:
                    seen.setdefault(p.track_id, now)
                fresh = [tid for tid, t in seen.items() if tid not in greeted and now - t >= g.delay_s]
                if fresh and now - last_greet < g.cooldown_s:
                    greeted.update(fresh)  # arrived with the group that was just greeted
                elif fresh and not watch.missing and not pending and player.ready(now):
                    text = book.pick_from(GREETINGS)
                    log.info("greeting tracks %s: %s", fresh, text)
                    player.play(voices.render(text), now, kind="greet")
                    last_greet = now
                    greeted.update(seen)
                else:
                    greeted.update(t for t in fresh if now - seen[t] > g.delay_s + g.max_wait_s)

            view = None
            if stream or (recorder and cfg.recording.annotated):
                view = annotate(frame, bowl, watch, people, judge, costumes, fps)
            if recorder:
                present = any(p.box[3] - p.box[1] >= cfg.recording.min_box_height for p in people)
                recorder.update(view if cfg.recording.annotated else frame, now, present)
            if stream:
                stream.publish(view)

            elapsed = time.monotonic() - t0
            fps = 0.9 * fps + 0.1 / max(elapsed, 1e-3) if fps else 1 / max(elapsed, 1e-3)
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        if recorder:
            recorder.close()
        if audio:
            audio.close()
        cam.close()


if __name__ == "__main__":
    main()