"""OpenPI control console: checkpoint inventory, owned serving and saved-replay diagnostics."""

import argparse
from collections import deque
import io
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
import traceback
from uuid import uuid4

import numpy as np
from PIL import Image

from openpi.control.catalog import Catalog
from openpi.control.catalog import read_json
from openpi.control.server import ModelServer
from openpi.control.server import ViewerLeases


def repository_root():
    source_root = Path(__file__).resolve().parents[3]
    if (source_root / "pyproject.toml").is_file():
        return source_root
    # Wheel installs have no checkout beside site-packages; use the launch directory.
    return Path.cwd()


class Console:
    def __init__(self, args):
        self.args = args
        self.catalog = Catalog(args.checkpoints)
        self.lock = threading.RLock()
        self.operation = threading.Lock()
        self.cancel = threading.Event()
        self.closing = threading.Event()
        self.viewers = ViewerLeases()
        self.token = secrets.token_urlsafe(32)
        self.server = ModelServer(
            args.output,
            config=args.config,
            device=args.device,
            memory_fraction=args.memory_fraction,
            model_lock=args.model_lock,
        )
        self.settings = {"horizon": 30, "execute_steps": 15, "max_chunks": 1, "control_hz": 30, "velocity_mode": "urdf"}
        self.settings_revision = 0
        self.selected = None
        self.samples, self.sample_index, self.frames = [], 0, {}
        self.phase, self.message = "preview", "选择 checkpoint 后启动模型，使用保存回放做只读推理。"
        self.events = deque(maxlen=8192)
        self.sequence, self.run, self.started, self.ended = 0, "", time.monotonic(), None
        self.output, self.completed = None, 0
        entries = self.catalog.scan(force=True)
        if args.checkpoint_id:
            self._select(args.checkpoint_id, 0)
        elif entries:
            self._select(min(entries.values(), key=lambda x: (x.task, -x.step)).id, 0)
        threading.Thread(target=self._lifecycle, name="openpi-console-lifecycle", daemon=True).start()

    def _select(self, checkpoint_id, sample_index):
        entry = self.catalog.get(checkpoint_id)
        path = entry.path / "validation/observations.json"
        samples = json.loads(path.read_text()) if path.is_file() else []
        for sample in samples:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", sample["task"]):
                raise ValueError("Invalid replay task name")
            if sample.get("role") != "target_support":
                raise ValueError("Only saved target-support inputs are allowed")
        if samples and not 0 <= sample_index < len(samples):
            raise ValueError("Invalid replay selection")
        frames = {}
        if samples:
            with np.load(
                entry.path / f"validation/replay-{samples[sample_index]['task']}.npz", allow_pickle=False
            ) as saved:
                for ui, key in [("left", "left_wrist"), ("main", "main"), ("right", "right_wrist")]:
                    buffer = io.BytesIO()
                    Image.fromarray(saved["image_" + key]).save(buffer, format="JPEG", quality=88)
                    frames[ui] = buffer.getvalue()
        if self.selected and (self.selected.id != entry.id or self.selected.signature != entry.signature):
            self.server.stop()
        with self.lock:
            self.selected, self.samples, self.sample_index, self.frames = entry, samples, sample_index, frames
            self.events.clear()
            self.run, self.started, self.ended = uuid4().hex, time.monotonic(), 0.0
            self.output, self.completed = None, 0
            self.phase, self.message = "preview", "选择已应用；启动模型后运行保存回放。"

    def guard(self):
        if self.cancel.is_set() or self.closing.is_set():
            raise InterruptedError("本轮已取消")
        if self.viewers.expired():
            raise InterruptedError("控制台页面已退出")

    def viewer_seen(self, viewer):
        self.viewers.seen(viewer)

    def viewer_leave(self, viewer):
        self.viewers.leave(viewer)

    def _lifecycle(self):
        while not self.closing.wait(0.25):
            if self.viewers.expired() and self.server.phase in {"loading", "ready"}:
                self.cancel.set()
                if self.operation.acquire(blocking=False):
                    try:
                        self.server.stop()
                    finally:
                        self.operation.release()

    def snapshot(self):
        with self.lock:
            inventory = self.catalog.snapshot()
            entry = self.catalog.entries.get(self.selected.id) if self.selected else None
            ready = bool(entry and self.selected.signature == entry.signature)
            sample = self.samples[self.sample_index] if self.samples else None
            return {
                "phase": self.phase,
                "message": self.message,
                "busy": self.operation.locked(),
                "server": self.server.snapshot(self.selected.id if self.selected else None),
                "inventory": inventory,
                "selected_id": self.selected.id if self.selected else None,
                "task": self.selected.task if self.selected else None,
                "checkpoint_step": self.selected.step if self.selected else None,
                "samples": self.samples,
                "sample_index": self.sample_index,
                "sample": sample,
                "settings": self.settings.copy(),
                "settings_revision": self.settings_revision,
                "policy_ready": ready,
                "replay_ready": ready and bool(self.samples),
                "capabilities": {"saved_replay": True, "live_cameras": False, "hardware_motion": False},
                "input_label": "保存的目标支持回放",
                "hardware_motion": False,
                "completed_chunks": self.completed,
                "policy_output": str(self.output) if self.output else None,
            }

    @staticmethod
    def validate_settings(payload):
        keys = ["horizon", "execute_steps", "max_chunks", "control_hz"]
        if any(type(payload.get(k)) is not int for k in keys):
            raise ValueError("推理参数必须是整数")
        values = {k: payload[k] for k in keys}
        if not 1 <= values["execute_steps"] <= values["horizon"] <= 30:
            raise ValueError("需要 1 ≤ IK 前缀步数 ≤ action horizon ≤ 30")
        if not 1 <= values["max_chunks"] <= 20 or values["control_hz"] != 30:
            raise ValueError("只读回放每轮 1–20 次推理，模型时间基准为 30 Hz")
        mode = payload.get("velocity_mode", "urdf")
        if mode not in {"urdf", "collection"}:
            raise ValueError("Unknown velocity mode")
        return dict(values, velocity_mode=mode)

    def act(self, action, payload):
        self.viewer_seen(payload.get("viewer", ""))
        if action == "stop":
            self.cancel.set()
            return
        if action not in {"select", "configure", "server_start", "server_stop", "shadow"}:
            raise ValueError("当前为保存回放模式，此操作不可用")
        if not self.operation.acquire(blocking=False):
            raise ValueError("已有操作正在进行")
        self.cancel.clear()

        def work():
            try:
                if action == "select":
                    if type(payload.get("sample", 0)) is not int:
                        raise ValueError("Invalid sample index")
                    self._select(payload.get("checkpoint_id"), payload.get("sample", 0))
                elif action == "configure":
                    settings = self.validate_settings(payload)
                    if settings["velocity_mode"] == "collection" and not self.args.yam_owner:
                        raise ValueError("采集速度模式需要显式指定本机 --yam-owner")
                    with self.lock:
                        self.settings, self.settings_revision = settings, self.settings_revision + 1
                        self.message = "推理参数已应用。"
                elif action == "server_stop":
                    self.server.stop()
                    self.phase, self.message = "preview", "模型已关闭。"
                else:
                    if self.selected is None or payload.get("checkpoint_id") != self.selected.id:
                        raise ValueError("所选模型已变化，请先应用当前选择")
                    entry = self.catalog.get(self.selected.id, signature=self.selected.signature)
                    if action == "server_start":
                        self.phase, self.message = "server_starting", "校验文件并加载模型…"
                        self.server.start(entry, self.guard)
                        self.guard()
                        self.phase, self.message = "server_ready", "所选模型已就绪。"
                    else:
                        if payload.get("settings_revision") != self.settings_revision:
                            raise ValueError("参数版本已变化，请重新应用")
                        if not self.server.snapshot(entry.id)["selected_match"]:
                            raise ValueError("请先启动所选模型")
                        self._infer(entry)
            except BaseException as error:
                self.phase, self.message = "rejected", f"{type(error).__name__}: {error}"
                traceback.print_exc()
            finally:
                if self.cancel.is_set() and self.server.phase == "loading":
                    self.server.stop()
                self.operation.release()

        threading.Thread(target=work, name="openpi-console-operation", daemon=True).start()

    def emit(self, kind, **data):
        with self.lock:
            self.sequence += 1
            self.events.append(dict(seq=self.sequence, t=time.monotonic() - self.started, kind=kind, **data))

    def telemetry_snapshot(self, after, run):
        with self.lock:
            reset = run != self.run or bool(self.events and after < self.events[0]["seq"] - 1)
            events = list(self.events) if reset else [x for x in self.events if x["seq"] > after]
            return {
                "run": self.run,
                "reset": reset,
                "events": events[-2000:],
                "seq": self.sequence,
                "now": self.ended if self.ended is not None else time.monotonic() - self.started,
                "active": self.operation.locked() and self.phase == "shadow",
            }

    def _mapping(self, entry, sample, directory):
        command = [
            str(self.args.ik_python or sys.executable),
            "-m",
            "openpi.control.yam.replay",
            "--checkpoint",
            str(entry.path),
            "--sample",
            sample["task"],
            "--actions",
            str(directory / "actions.npy"),
            "--metadata",
            str(directory / "metadata.json"),
            "--output",
            str(directory / "mapping"),
            "--steps",
            str(self.settings["execute_steps"]),
            "--velocity-mode",
            self.settings["velocity_mode"],
        ]
        for flag, value in [("--support-poses", self.args.support_poses), ("--yam-owner", self.args.yam_owner)]:
            if value:
                command.extend([flag, str(value)])
        env = {**os.environ, "PYTHONPATH": str(repository_root() / "src") + os.pathsep + os.getenv("PYTHONPATH", "")}
        with (directory / "mapping.log").open("w") as log:
            result = subprocess.run(
                command,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=300,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(f"动作映射失败，日志：{directory / 'mapping.log'}")
        return read_json(directory / "mapping/report.json")

    def _infer(self, entry):
        if not self.samples or entry.schema != "yam1-hybrid-inference-v1":
            raise ValueError("所选模型没有此后端支持的保存回放；可独立启动模型 WebSocket 服务")
        sample = self.samples[self.sample_index]
        with np.load(entry.path / f"validation/replay-{sample['task']}.npz", allow_pickle=False) as saved:
            observation = {
                "state": saved["state"],
                "state_valid": saved["state_valid"],
                "camera_valid": saved["camera_valid"],
                "prompt": str(saved["prompt"]),
                "images": {k: saved["image_" + k] for k in ["main", "left_wrist", "right_wrist"]},
            }
        with self.lock:
            self.phase, self.run, self.started, self.ended = "shadow", uuid4().hex, time.monotonic(), None
            self.events.clear()
            self.output = self.args.output / "runs" / self.run
            self.output.mkdir(parents=True)
            self.completed = 0
        report = {
            "checkpoint_id": entry.id,
            "step": entry.step,
            "metadata": self.server.metadata,
            "input": sample,
            "settings": self.settings.copy(),
            "hardware_access": False,
            "hardware_motion": False,
            "requests": [],
        }
        self.emit("start", hardware_motion=False, settings=self.settings.copy())
        try:
            for index in range(self.settings["max_chunks"]):
                self.guard()
                self.emit("decision", index=index)
                started = time.monotonic()
                result = self.server.infer(observation)
                elapsed = time.monotonic() - started
                actions = np.asarray(result["actions"])
                if actions.shape != (30, 14) or not np.isfinite(actions).all():
                    raise ValueError("模型没有返回有限的 [30,14] 动作")
                self.guard()
                directory = self.output / f"prediction-{index:03d}"
                directory.mkdir()
                np.save(directory / "actions.npy", actions, allow_pickle=False)
                (directory / "metadata.json").write_text(json.dumps(self.server.metadata, indent=2))
                self.emit(
                    "prediction",
                    index=index,
                    actions=actions.tolist(),
                    horizon=self.settings["horizon"],
                    execute_steps=self.settings["execute_steps"],
                    inference_s=elapsed,
                )
                mapping = self._mapping(entry, sample, directory)
                if mapping["ik_status"] == "passed":
                    self.emit("ik", index=index, points=mapping["points"], step_period_s=1 / 30)
                record = {
                    "index": index,
                    "shape": list(actions.shape),
                    "finite": True,
                    "inference_ms": elapsed * 1000,
                    "geometry_status": mapping["geometry_status"],
                    "ik_status": mapping["ik_status"],
                    "reason": mapping.get("reason"),
                }
                report["requests"].append(record)
                self.completed += 1
                (self.output / "report.json").write_text(json.dumps(report, indent=2))
            rejected = sum(x["ik_status"] == "rejected" for x in report["requests"])
            self.phase = "complete"
            self.message = f"只读推理完成 {self.completed} 次；坐标映射通过；IK 拒绝 {rejected} 次。"
            report["status"] = "complete"
        except BaseException as error:
            report.update(status="interrupted_or_failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            self.emit("end")
            self.ended = time.monotonic() - self.started
            (self.output / "report.json").write_text(json.dumps(report, indent=2))

    def close(self):
        self.closing.set()
        self.cancel.set()
        if self.operation.acquire(timeout=20):
            try:
                self.server.stop()
            finally:
                self.operation.release()
        else:
            self.server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, default=repository_root() / "checkpoints")
    parser.add_argument("--output", type=Path, default=repository_root() / ".openpi-control")
    parser.add_argument("--checkpoint-id")
    parser.add_argument("--config", default="pi05_yam_hybrid")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=0.52)
    parser.add_argument("--model-lock", type=Path)
    parser.add_argument("--ik-python", type=Path)
    parser.add_argument("--support-poses", type=Path)
    parser.add_argument("--yam-owner", type=Path)
    parser.add_argument("--host", choices=["127.0.0.1", "localhost"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 0 < args.memory_fraction <= 1:
        parser.error("memory-fraction must be in (0,1]")
    args.checkpoints, args.output = args.checkpoints.absolute(), args.output.absolute()
    app = Console(args)
    from openpi.control.http import serve

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        serve(app)
    finally:
        app.close()


if __name__ == "__main__":
    main()
