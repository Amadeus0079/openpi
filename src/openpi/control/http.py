"""Loopback HTTP transport for the console and its incremental telemetry."""

import contextlib
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs
from urllib.parse import urlparse
from uuid import uuid4

STATIC = Path(__file__).with_name("static")


def make_server(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, code, data, mime="application/json"):
            if isinstance(data, (dict, list)):
                data = json.dumps(data, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(data)

        def local_host(self):
            host = self.headers.get("Host", "").split(":")[0]
            if host not in {"127.0.0.1", "localhost"}:
                self.reply(403, {"error": "Loopback Host required"})
                return False
            return True

        def do_GET(self):
            if not self.local_host():
                return
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    viewer = uuid4().hex
                    app.viewer_seen(viewer)
                    html = (STATIC / "workbench.html").read_text()
                    html = html.replace("TOKEN_VALUE", app.token).replace("CLIENT_VALUE", viewer)
                    self.reply(200, html.encode(), "text/html; charset=utf-8")
                elif parsed.path == "/static/workbench.js":
                    self.reply(200, (STATIC / "workbench.js").read_bytes(), "application/javascript; charset=utf-8")
                elif parsed.path == "/api/state":
                    if "viewer" in query:
                        app.viewer_seen(query["viewer"][0])
                    self.reply(200, app.snapshot())
                elif parsed.path == "/api/catalog":
                    with app.lock:
                        app.catalog.scan(force=True)
                        self.reply(200, app.catalog.snapshot())
                elif parsed.path == "/api/telemetry":
                    self.reply(200, app.telemetry_snapshot(int(query.get("after", [0])[0]), query.get("run", [""])[0]))
                elif parsed.path.startswith("/camera/"):
                    with app.lock:
                        data = app.camera_frame(parsed.path.split("/")[-1])
                    self.reply(200 if data else 404, data or b"", "image/jpeg")
                elif parsed.path.startswith("/reference/"):
                    data = app.reference_frame(parsed.path.split("/")[-1])
                    self.reply(200 if data else 404, data or b"", "image/jpeg")
                else:
                    self.reply(404, {"error": "Not found"})
            except (ValueError, OSError) as error:
                self.reply(400, {"error": str(error)})

        def do_POST(self):
            if not self.local_host():
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 4096:
                    raise ValueError("Request too large")
                payload = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("Expected JSON object")
                path = urlparse(self.path).path
                leaving = path == "/api/viewer_leave"
                token = payload.get("token") if leaving else self.headers.get("X-Replay-Token")
                if token != app.token:
                    self.reply(403, {"error": "Invalid session token"})
                    return
                if leaving:
                    app.viewer_leave(payload.get("viewer", ""))
                elif path.startswith("/api/"):
                    app.act(path.removeprefix("/api/"), payload)
                else:
                    self.reply(404, {"error": "Not found"})
                    return
                self.reply(200, {"accepted": True})
            except (ValueError, RuntimeError) as error:
                self.reply(409, {"error": str(error)})

    server = ThreadingHTTPServer((app.args.host, app.args.port), Handler)
    server.daemon_threads = True
    return server


def serve(app):
    server = make_server(app)
    print(f"OpenPI control: http://{app.args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
