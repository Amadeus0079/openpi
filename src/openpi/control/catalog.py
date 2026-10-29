"""Discover complete local checkpoints without reading multi-gigabyte weights.

A task is the relative training-run directory; numeric children are its steps.
Thus both task/step and config/task/recipe/step work without a task registry.
The root and run directories may be symlinks to a data disk.
"""

from dataclasses import asdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time


def read_json(path: Path) -> dict:
    if path.stat().st_size > 4 << 20:
        raise ValueError(f"Metadata too large: {path.name}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected metadata object: {path.name}")
    return value


@dataclass(frozen=True)
class Checkpoint:
    id: str
    task: str
    step: int
    path: Path
    format: str
    schema: str
    model_sha256: str | None
    signature: tuple

    def public(self):
        return {k: v for k, v in asdict(self).items() if k not in {"path", "signature"}}


class Catalog:
    def __init__(self, root: Path, *, refresh_seconds=3.0):
        self.root = root.absolute()
        self.refresh_seconds = refresh_seconds
        self.entries = {}
        self.errors = []
        self.scanned = float("-inf")

    def _inspect(self, directory: Path) -> Checkpoint:
        step = int(directory.name)
        relative = directory.relative_to(self.root)
        manifest_path = directory / "inference-manifest.json"
        manifest = read_json(manifest_path) if manifest_path.is_file() else {}
        if manifest and (type(manifest.get("step")) is not int or manifest["step"] != step):
            raise ValueError("目录步数与发布清单不一致")
        complete = directory / "complete.json"
        if complete.is_file() and read_json(complete).get("step") != step:
            raise ValueError("checkpoint 完成标记步数不一致")
        files = manifest.get("files", {})
        if not isinstance(files, dict):
            raise ValueError("Invalid publication file list")
        if manifest.get("schema") == "yam1-hybrid-inference-v1":
            required = {
                "model.safetensors",
                "run.json",
                "complete.json",
                "assets/yam_hybrid/config.json",
                "assets/yam_hybrid/norm_stats.json",
                "assets/yam_hybrid/deployment/tokenizer.model",
                "assets/yam_hybrid/deployment/target/target-calibration.json",
            }
            if not required.issubset(files):
                raise ValueError("YAM-1 发布清单不完整")
        for name, record in files.items():
            file = directory / name
            if not file.resolve().is_relative_to(directory.resolve()) or not file.is_file():
                raise ValueError("发布清单中的文件缺失或路径无效")
            if file.stat().st_size != record["size"]:
                raise ValueError("发布文件尚未完整接收")
        weights = directory / "model.safetensors"
        if weights.is_file() and weights.stat().st_size:
            kind = "pytorch"
        elif (directory / "params").is_dir() and any((directory / "params").iterdir()):
            kind = "jax"
            weights = directory / "params"
        else:
            raise ValueError("没有完整模型文件")
        if not (directory / "assets").is_dir():
            raise ValueError("缺少 checkpoint assets")
        stat = weights.stat()
        metadata_stat = manifest_path.stat() if manifest_path.exists() else stat
        task = relative.parent.as_posix()
        return Checkpoint(
            id=relative.as_posix(),
            task=task,
            step=step,
            path=directory,
            format=kind,
            schema=manifest.get("schema", "openpi"),
            model_sha256=files.get("model.safetensors", {}).get("sha256"),
            signature=(str(directory.resolve()), stat.st_size, stat.st_mtime_ns, metadata_stat.st_mtime_ns),
        )

    def scan(self, *, force=False):
        if not force and time.monotonic() - self.scanned < self.refresh_seconds:
            return self.entries
        entries, errors, seen = {}, [], set()
        stack = [(self.root, 0)]
        while stack:
            directory, depth = stack.pop()
            try:
                real = directory.resolve(strict=True)
                if real in seen or depth > 10:
                    continue
                seen.add(real)
                if len(seen) > 10000:
                    errors.append({"id": ".", "reason": "目录数量过多，请缩小 checkpoints 根目录"})
                    break
                candidate = any(
                    (directory / name).exists()
                    for name in ["model.safetensors", "params", "inference-manifest.json", "complete.json"]
                )
                if directory.name.isdecimal() and directory != self.root and candidate:
                    try:
                        entry = self._inspect(directory)
                        entries[entry.id] = entry
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        errors.append({"id": directory.relative_to(self.root).as_posix(), "reason": str(error)})
                    continue
                with os.scandir(directory) as children:
                    stack.extend(
                        (Path(c.path), depth + 1)
                        for c in sorted(children, key=lambda c: c.name, reverse=True)
                        if not c.name.startswith(".") and c.is_dir(follow_symlinks=True)
                    )
            except (OSError, ValueError) as error:
                errors.append({"id": str(directory), "reason": str(error)})
        self.entries, self.errors, self.scanned = entries, errors, time.monotonic()
        return entries

    def get(self, checkpoint_id: str, *, signature=None) -> Checkpoint:
        self.scan(force=True)
        if checkpoint_id not in self.entries:
            raise ValueError("所选 checkpoint 已不可用，请刷新列表")
        entry = self.entries[checkpoint_id]
        if signature is not None and entry.signature != signature:
            raise ValueError("所选 checkpoint 已被替换，请重新选择并加载")
        return entry

    def snapshot(self):
        self.scan()
        tasks = {}
        for entry in sorted(self.entries.values(), key=lambda c: (c.task, -c.step, c.id)):
            tasks.setdefault(entry.task, []).append(entry.public())
        return {
            "root": str(self.root),
            "tasks": [{"id": task, "steps": steps} for task, steps in tasks.items()],
            "unavailable": self.errors,
            "integrity": "size_checked; SHA256 verified again before model loading",
        }
