"""IPC boundary to the separate NumPy-2 YAM camera/IK environment."""

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from uuid import uuid4


class LiveClient:
    def __init__(self, args, checkpoint, event_callback):
        self.directory = args.output / "live"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock, self.pending = threading.Lock(), {}
        self.event_callback = event_callback
        self.closed = False
        self.created = time.monotonic()
        self.error = None
        self.log = (self.directory / "worker.log").open("a")
        python = str(args.live_python or args.ik_python or sys.executable)
        command = [
            python,
            "-m",
            "openpi.control.child",
            str(os.getpid()),
            python,
            "-m",
            "openpi.control.yam.live_worker",
            "--checkpoint",
            str(checkpoint),
            "--profile",
            str(args.live_profile),
            "--client-root",
            str(args.client_root),
            "--recordings",
            str(args.recordings_root),
            "--output",
            str(self.directory),
            "--owner",
            str(args.yam_owner or args.client_root),
        ]
        env = {
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]) + os.pathsep + str(args.client_root),
            "PYTHONUNBUFFERED": "1",
        }
        ready = threading.Event()

        def spawn_and_read():
            try:
                self.process = subprocess.Popen(
                    command,
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=self.log,
                    text=True,
                    bufsize=1,
                )
            except Exception as error:
                self.error = str(error)
                ready.set()
                return
            ready.set()
            self._read()

        threading.Thread(target=spawn_and_read, daemon=True, name="yam-live-parent").start()
        if not ready.wait(30) or self.error:
            raise RuntimeError(self.error or "Live worker spawn timeout")

    def _read(self):
        try:
            for line in self.process.stdout:
                message = json.loads(line)
                if "event" in message:
                    self.event_callback(message["event"], message.get("data", {}))
                elif message.get("id") in self.pending:
                    result, event = self.pending[message["id"]]
                    result.update(message)
                    event.set()
        except Exception as error:
            self.error = str(error)
        finally:
            self.error = self.error or "Live worker exited; see live/worker.log"
            for result, event in list(self.pending.values()):
                result.setdefault("error", self.error)
                event.set()

    def call(self, action, payload=None, timeout=900):
        request_id, result, event = uuid4().hex, {}, threading.Event()
        with self.lock:
            if self.closed or self.process.poll() is not None:
                raise RuntimeError(self.error or "Live worker is not running")
            self.pending[request_id] = (result, event)
            self.process.stdin.write(json.dumps({"id": request_id, "action": action, "payload": payload or {}}) + "\n")
            self.process.stdin.flush()
        try:
            if not event.wait(timeout):
                self.notify("stop")
                raise TimeoutError("Live operation timed out")
            if "error" in result:
                raise RuntimeError(result["error"])
            return result.get("result")
        finally:
            self.pending.pop(request_id, None)

    def notify(self, action):
        if not self.closed and self.process.poll() is None:
            with self.lock:
                try:
                    self.process.stdin.write(json.dumps({"action": action, "payload": {}}) + "\n")
                    self.process.stdin.flush()
                except (OSError, ValueError):
                    pass

    def snapshot(self):
        try:
            data = json.loads((self.directory / "status.json").read_text())
            data["status_age_s"] = time.monotonic() - data["updated_monotonic"]
            if data["status_age_s"] > 2 or data["updated_monotonic"] < self.created:
                data["ready"] = False
            if self.process.poll() is not None:
                data.update(ready=False, error=self.error or "Live worker exited")
            return data
        except (OSError, ValueError):
            return {"ready": False, "error": self.error, "phase": "connecting"}

    def image(self, name, *, reference=False):
        if name not in {"left", "main", "right"}:
            return None
        path = self.directory / (("reference-" if reference else "camera-") + name + ".jpg")
        try:
            return path.read_bytes()
        except OSError:
            return None

    def close(self):
        self.notify("stop")
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.closed = True
        self.log.close()
