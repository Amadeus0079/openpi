import json
import os

import pytest

from openpi.control.catalog import Catalog


def checkpoint(root, name, *, complete=True):
    path = root / name
    path.mkdir(parents=True)
    (path / "assets").mkdir()
    (path / "model.safetensors").write_bytes(b"model")
    if complete:
        manifest = {
            "schema": "test-publication",
            "step": int(path.name),
            "files": {"model.safetensors": {"size": 5, "sha256": "0" * 64}},
        }
        (path / "inference-manifest.json").write_text(json.dumps(manifest))
        (path / "complete.json").write_text(json.dumps({"step": int(path.name)}))
    return path


def test_nested_tasks_numeric_steps_and_data_disk_symlinks(tmp_path):
    disk = tmp_path / "disk"
    checkpoint(disk, "task/M0/100")
    checkpoint(disk, "task/M0/2000")
    checkpoint(disk, "task/M1/100")
    root = tmp_path / "checkpoints"
    root.symlink_to(disk, target_is_directory=True)
    (disk / "loop").symlink_to(disk, target_is_directory=True)
    inventory = Catalog(root).snapshot()
    assert [x["id"] for x in inventory["tasks"]] == ["task/M0", "task/M1"]
    assert [x["step"] for x in inventory["tasks"][0]["steps"]] == [2000, 100]


def test_incomplete_download_bad_metadata_and_hidden_staging_are_not_selectable(tmp_path):
    good = checkpoint(tmp_path, "good/100")
    checkpoint(tmp_path, ".4000.download/task/4000")
    partial = checkpoint(tmp_path, "partial/100")
    (partial / "model.safetensors").write_bytes(b"m")
    wrong_step = checkpoint(tmp_path, "wrong/100")
    (wrong_step / "complete.json").write_text('{"step":200}')
    malformed = checkpoint(tmp_path, "malformed/100")
    (malformed / "inference-manifest.json").write_text("[]")
    catalog = Catalog(tmp_path)
    assert list(catalog.scan()) == [good.relative_to(tmp_path).as_posix()]
    assert len(catalog.errors) == 3


def test_new_completion_removal_and_replacement_invalidate_selection(tmp_path):
    path = checkpoint(tmp_path, "task/100")
    catalog = Catalog(tmp_path)
    first = catalog.get("task/100")
    (path / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="不可用"):
        catalog.get(first.id)
    (path / "model.safetensors").write_bytes(b"model")
    with pytest.raises(ValueError, match="替换"):
        catalog.get(first.id, signature=first.signature)
    assert catalog.get(first.id).step == 100
    checkpoint(tmp_path, "task/200")
    assert len(catalog.scan(force=True)) == 2


def test_atomic_same_size_replacement_with_preserved_mtime_invalidates_selection(tmp_path):
    path = checkpoint(tmp_path, "task/100")
    catalog = Catalog(tmp_path)
    first = catalog.get("task/100")
    weights = path / "model.safetensors"
    before = weights.stat()
    replacement = path / "replacement"
    replacement.write_bytes(b"other")
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    replacement.replace(weights)
    with pytest.raises(ValueError, match="替换"):
        catalog.get(first.id, signature=first.signature)


def test_path_traversal_and_manifest_escape_are_rejected(tmp_path):
    checkpoint(tmp_path, "task/100")
    outside = tmp_path / "outside"
    outside.write_bytes(b"x")
    path = checkpoint(tmp_path, "escape/100")
    manifest = json.loads((path / "inference-manifest.json").read_text())
    manifest["files"]["../../outside"] = {"size": 1}
    (path / "inference-manifest.json").write_text(json.dumps(manifest))
    catalog = Catalog(tmp_path)
    assert list(catalog.scan()) == ["task/100"]
    with pytest.raises(ValueError, match="不可用"):
        catalog.get("../../outside")


def test_stock_pytorch_jax_and_numeric_task_directory(tmp_path):
    checkpoint(tmp_path, "2026/pytorch/100", complete=False)
    jax = tmp_path / "config/jax/200"
    (jax / "params").mkdir(parents=True)
    (jax / "params/_METADATA").write_text("{}")
    (jax / "assets").mkdir()
    catalog = Catalog(tmp_path)
    assert {x.format for x in catalog.scan().values()} == {"jax", "pytorch"}


def test_yam_publication_without_required_assets_is_not_listed(tmp_path):
    path = checkpoint(tmp_path, "yam/M0/4000")
    manifest = json.loads((path / "inference-manifest.json").read_text())
    manifest["schema"] = "yam1-hybrid-inference-v1"
    (path / "inference-manifest.json").write_text(json.dumps(manifest))
    catalog = Catalog(tmp_path)
    assert not catalog.scan()
    assert "发布清单不完整" in catalog.errors[0]["reason"]
