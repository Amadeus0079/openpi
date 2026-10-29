from types import SimpleNamespace

import pytest

from openpi.control.server import ModelServer
from openpi.control.server import ViewerLeases


def test_viewer_leases_require_valid_identity_and_expire():
    leases = ViewerLeases(timeout=0.01)
    with pytest.raises(ValueError, match="viewer"):
        leases.seen("bad")
    leases.seen("a" * 32)
    assert not leases.expired()
    leases.leave("a" * 32)
    assert leases.expired()


def test_exited_child_does_not_receive_signals_and_lease_is_released(tmp_path):
    server = ModelServer(tmp_path, config="test", device="cpu", memory_fraction=0.5)
    process = SimpleNamespace(poll=lambda: 0)
    server.process = process
    server.phase = "ready"
    server.lease = (tmp_path / "model.lock").open("a")
    state = server.snapshot(None)
    assert state["phase"] == "stopped"
    assert server.process is None
    assert server.lease is None


def test_cancel_while_waiting_for_shared_model_lock_does_not_spawn(tmp_path):
    import fcntl

    lock = (tmp_path / "shared.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    server = ModelServer(tmp_path, config="test", device="cpu", memory_fraction=0.5, model_lock=lock.name)
    entry = SimpleNamespace(id="task/100", path=tmp_path / "100", signature=())
    calls = []

    def guard():
        calls.append(True)
        if len(calls) > 1:
            raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError, match="cancelled"):
        server.start(entry, guard)
    assert server.process is None
    assert server.lease is None
    lock.close()
