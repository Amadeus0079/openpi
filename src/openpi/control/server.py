"""Own one model process and its GPU lease; never signal PIDs loaded from disk."""

import fcntl
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import threading
import time
import urllib.request


class ViewerLeases:
    def __init__(self, timeout=10.0):
        self.timeout = timeout
        self.viewers = {}
        self.lock = threading.Lock()

    def seen(self, viewer):
        if not isinstance(viewer, str) or len(viewer) != 32 or any(c not in "0123456789abcdef" for c in viewer):
            raise ValueError("Invalid viewer")
        with self.lock:
            self.viewers[viewer] = time.monotonic()

    def leave(self, viewer):
        with self.lock:
            self.viewers.pop(viewer, None)

    def expired(self):
        with self.lock:
            now = time.monotonic()
            self.viewers = {k: v for k, v in self.viewers.items() if now - v <= self.timeout}
            return not self.viewers

    def recent(self, seconds=3):
        with self.lock:
            return any(time.monotonic() - value < seconds for value in self.viewers.values())


class ModelServer:
    def __init__(self, directory, *, config, device, memory_fraction, model_lock=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config, self.device, self.memory_fraction = config, device, memory_fraction
        self.lock_path = Path(model_lock) if model_lock else self.directory / "model.lock"
        self.process = None
        self.lease = None
        self.entry = None
        self.metadata = None
        self.port = None
        self.phase = "stopped"
        self.message = "Server 未启动"
        self.spawn_queue = queue.Queue()
        threading.Thread(target=self._spawn_loop, name="openpi-model-parent", daemon=True).start()

    def _spawn_loop(self):
        while True:
            args, kwargs, result = self.spawn_queue.get()
            try:
                result.put(subprocess.Popen(args, **kwargs))
            except Exception as error:
                result.put(error)

    def snapshot(self, selected):
        if self.process is not None and self.process.poll() is not None and self.phase == "ready":
            self.stop()
            self.message = "模型进程已退出，请查看日志后重新加载"
        return {
            "phase": self.phase,
            "message": self.message,
            "pid": self.process.pid if self.process else None,
            "dataset": self.entry.id if self.entry else None,
            "checkpoint": str(self.entry.path) if self.entry else None,
            "model_sha256": (self.metadata or {}).get("model_sha256"),
            "selected_match": bool(self.phase == "ready" and self.entry and selected == self.entry.id),
        }

    def stop(self):
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if self.lease is not None:
            self.lease.close()
            self.lease = None
        self.phase, self.message, self.metadata = "stopped", "Server 已关闭", None

    def start(self, entry, guard):
        if self.snapshot(entry.id)["selected_match"] and self.entry.signature == entry.signature:
            return self.metadata
        self.stop()
        self.entry, self.phase, self.message = entry, "loading", "等待模型锁并校验 checkpoint…"
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lease = self.lock_path.open("a")
        try:
            deadline = time.monotonic() + 900
            while True:
                guard()
                try:
                    fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as error:
                    if time.monotonic() > deadline:
                        raise TimeoutError("等待其他模型释放 GPU 超时") from error
                    time.sleep(0.2)
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                self.port = probe.getsockname()[1]
            evidence = self.directory / "models" / entry.id
            evidence.mkdir(parents=True, exist_ok=True)
            argv = [
                sys.executable,
                "-m",
                "openpi.control.child",
                str(os.getpid()),
                sys.executable,
                "-m",
                "openpi.control.serve",
                "--checkpoint",
                str(entry.path),
                "--config",
                self.config,
                "--device",
                self.device,
                "--port",
                str(self.port),
                "--memory-fraction",
                str(self.memory_fraction),
                "--evidence",
                str(evidence),
            ]
            env = {
                **os.environ,
                "JAX_PLATFORMS": "cpu",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "OMP_NUM_THREADS": "4",
                "PYTHONUNBUFFERED": "1",
            }
            with (evidence / "server.log").open("a") as log:
                result = queue.Queue()
                self.spawn_queue.put(
                    (
                        argv,
                        {
                            "env": env,
                            "stdin": subprocess.DEVNULL,
                            "stdout": log,
                            "stderr": subprocess.STDOUT,
                            "start_new_session": True,
                        },
                        result,
                    )
                )
                process = result.get(timeout=30)
            if isinstance(process, Exception):
                raise process
            self.process = process
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                guard()
                if self.process.poll() is not None:
                    raise RuntimeError(f"模型加载失败，日志：{evidence / 'server.log'}")
                try:
                    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
                        f"http://127.0.0.1:{self.port}/healthz", timeout=1
                    ) as health:
                        if health.status == 200:
                            break
                except OSError:
                    time.sleep(0.3)
            else:
                raise TimeoutError("模型加载超时")
            from openpi_client import msgpack_numpy
            from websockets.sync.client import connect

            with connect(f"ws://127.0.0.1:{self.port}", compression=None, max_size=None) as client:
                metadata = msgpack_numpy.unpackb(client.recv(timeout=20))
            if Path(metadata["checkpoint"]).resolve() != entry.path.resolve() or metadata["step"] != entry.step:
                raise ValueError("模型握手 checkpoint/step 不一致")
            if entry.model_sha256 and metadata.get("model_sha256") != entry.model_sha256:
                raise ValueError("模型握手 SHA 不一致")
            self.metadata, self.phase, self.message = metadata, "ready", "Server 已加载所选 checkpoint"
            return metadata
        except BaseException:
            self.stop()
            self.phase, self.message = "error", "模型加载中止，请查看日志"
            raise

    def infer(self, observation):
        if self.phase != "ready" or self.process is None or self.process.poll() is not None:
            raise ValueError("请先加载所选模型")
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect

        packer = msgpack_numpy.Packer()
        with connect(f"ws://127.0.0.1:{self.port}", compression=None, max_size=None) as client:
            metadata = msgpack_numpy.unpackb(client.recv(timeout=20))
            if metadata.get("model_sha256") != self.metadata.get("model_sha256"):
                raise ValueError("模型身份在运行期间变化")
            client.send(packer.pack({"type": "reset"}))
            if msgpack_numpy.unpackb(client.recv(timeout=20)) != {"status": "ok"}:
                raise ValueError("模型 reset 未确认")
            client.send(packer.pack(observation))
            return msgpack_numpy.unpackb(client.recv(timeout=180))
