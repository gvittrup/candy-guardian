import json
import logging
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

log = logging.getLogger(__name__)

PAGE = """<!doctype html>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Candy Guardian</title>
<style>
  body { font-family: system-ui; background: #111; color: #eee; margin: 16px; }
  img { max-width: 100%; cursor: crosshair; display: block; }
  button { margin: 8px 8px 0 0; padding: 6px 12px; }
  a { color: #f90; }
</style>
<h1>Candy Guardian</h1>
<p><a href="/clips">Recorded clips</a></p>
<p>Click the corners of the bowl, then Save. Points: <span id="n">0</span></p>
<img id="v" src="/stream">
<button id="save">Save bowl</button><button id="clear">Clear</button><button id="empty">Capture empty spot</button>
<script>
  const v = document.getElementById('v'), n = document.getElementById('n');
  let pts = [];
  v.onclick = e => {
    const r = v.getBoundingClientRect();
    pts.push([Math.round((e.clientX - r.left) * v.naturalWidth / r.width),
              Math.round((e.clientY - r.top) * v.naturalHeight / r.height)]);
    n.textContent = pts.length;
  };
  document.getElementById('clear').onclick = () => { pts = []; n.textContent = 0; };
  document.getElementById('save').onclick = async () => {
    if (pts.length < 3) return alert('Click at least 3 points');
    await fetch('/roi', { method: 'POST', body: JSON.stringify(pts) });
    pts = []; n.textContent = 0;
  };
  document.getElementById('empty').onclick = async () => {
    if (!confirm('Bowl removed and nobody in view?')) return;
    await fetch('/empty', { method: 'POST' });
  };
</script>
"""

CLIP_NAME = re.compile(r"^[\w-]+\.(mp4|avi)$")  # only our own timestamped files, no paths

CLIPS_PAGE = """<!doctype html>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Candy Guardian clips</title>
<style>
  body { font-family: system-ui; background: #111; color: #eee; margin: 16px; }
  video { width: 100%; max-width: 960px; display: block; background: #000; }
  a { color: #f90; }
  li { margin: 6px 0; }
  button { margin-left: 8px; }
</style>
<h1>Clips</h1>
<p><a href="/">Back to live view</a></p>
<video id="v" controls playsinline></video>
<ul>{rows}</ul>
<script>
  const v = document.getElementById('v');
  document.querySelectorAll('[data-src]').forEach(a => a.onclick = e => {
    e.preventDefault(); v.src = a.dataset.src; v.play();
  });
  document.querySelectorAll('[data-del]').forEach(b => b.onclick = async () => {
    if (!confirm('Delete this clip?')) return;
    await fetch('/delete/' + b.dataset.del, { method: 'POST' });
    b.parentElement.remove();
  });
</script>
"""


class DebugStream:
    def __init__(self, cfg, on_roi, on_empty, clips_dir=None):
        self.quality = cfg.jpeg_quality
        self.on_roi = on_roi
        self.on_empty = on_empty
        self.clips_dir = clips_dir
        self._jpeg = None
        self._cond = threading.Condition()
        server = ThreadingHTTPServer((cfg.host, cfg.port), self._handler())
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        log.info("debug stream on http://%s:%d", cfg.host, cfg.port)

    def publish(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            with self._cond:
                self._jpeg = buf.tobytes()
                self._cond.notify_all()

    def _clip_path(self, name):
        if not self.clips_dir or not CLIP_NAME.match(name):
            return None
        p = self.clips_dir / name
        return p if p.is_file() else None

    def _clips_page(self):
        rows = []
        files = sorted(self.clips_dir.iterdir(), reverse=True) if self.clips_dir and self.clips_dir.exists() else []
        for p in files:
            if not CLIP_NAME.match(p.name):
                continue
            day, _, clock = p.stem.partition("_")
            label = f"{day} {clock.replace('-', ':')}"
            mb = p.stat().st_size / 1e6
            if p.suffix == ".mp4":
                rows.append(f'<li><a href="#" data-src="/clips/{p.name}">{label}</a> {mb:.1f} MB'
                            f'<button data-del="{p.name}">Delete</button></li>')
            else:
                rows.append(f'<li>{label} (recording or converting) '
                            f'<a href="/clips/{p.name}" download>download</a></li>')
        return CLIPS_PAGE.replace("{rows}", "\n".join(rows) or "<li>No clips yet</li>")

    def _handler(self):
        stream = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == "/":
                    self._send_html(PAGE)
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        while True:
                            with stream._cond:
                                stream._cond.wait(timeout=2)
                                jpg = stream._jpeg
                            if jpg is None:
                                continue
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                elif self.path == "/clips":
                    self._send_html(stream._clips_page())
                elif self.path.startswith("/clips/"):
                    self._send_clip(self.path[len("/clips/"):])
                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path.startswith("/delete/"):
                    path = stream._clip_path(self.path[len("/delete/"):])
                    if path is None:
                        return self.send_error(404)
                    path.unlink()
                    self.send_response(204)
                    self.end_headers()
                    return
                if self.path == "/empty":
                    stream.on_empty()
                    self.send_response(204)
                    self.end_headers()
                    return
                if self.path != "/roi":
                    return self.send_error(404)
                try:
                    n = int(self.headers.get("Content-Length", 0))
                    pts = json.loads(self.rfile.read(n))
                    pts = [[int(x), int(y)] for x, y in pts]
                    if len(pts) < 3:
                        raise ValueError("need 3+ points")
                except (ValueError, TypeError) as e:
                    return self.send_error(400, str(e))
                stream.on_roi(pts)
                self.send_response(204)
                self.end_headers()

            def _send_html(self, text):
                body = text.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_clip(self, name):
                path = stream._clip_path(name)
                if path is None:
                    return self.send_error(404)
                size = path.stat().st_size
                start, end = 0, size - 1
                m = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range", ""))
                if m and (m[1] or m[2]):
                    if m[1]:
                        start = int(m[1])
                        end = min(int(m[2]), size - 1) if m[2] else size - 1
                    else:
                        start = max(0, size - int(m[2]))
                    if start > end:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.end_headers()
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                else:
                    self.send_response(200)
                self.send_header("Content-Type", "video/mp4" if name.endswith(".mp4") else "video/x-msvideo")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(end - start + 1))
                self.end_headers()
                try:
                    with open(path, "rb") as f:
                        f.seek(start)
                        left = end - start + 1
                        while left > 0:
                            chunk = f.read(min(65536, left))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            left -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # browser closed or seeked elsewhere

        return Handler