import json
import threading
from types import SimpleNamespace
import urllib.error
import urllib.request

import pytest

from openpi.control.app import Console
from openpi.control.catalog_test import checkpoint
from openpi.control.http import make_server

urlopen = urllib.request.build_opener(urllib.request.ProxyHandler({})).open


@pytest.fixture
def console(tmp_path):
    root = tmp_path / "checkpoints"
    checkpoint(root, "task/M0/4000")
    args = SimpleNamespace(
        checkpoints=root,
        output=tmp_path / "output",
        config="pi05_yam_hybrid",
        device="cpu",
        memory_fraction=0.5,
        model_lock=None,
        checkpoint_id=None,
        host="127.0.0.1",
        port=0,
        ik_python=None,
        support_poses=None,
        yam_owner=None,
    )
    app = Console(args)
    server = make_server(app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield app, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    app.close()
    thread.join(timeout=2)


def test_http_inventory_empty_replay_and_static_assets(console):
    _, base = console
    with urlopen(base) as response:
        html = response.read().decode()
    assert 'name="openpi-token"' in html
    assert "TOKEN_VALUE" not in html
    with urlopen(base + "/static/workbench.js") as response:
        assert b"renderSteps" in response.read()
    with urlopen(base + "/api/state") as response:
        state = json.load(response)
    assert state["selected_id"] == "task/M0/4000"
    assert state["inventory"]["tasks"][0]["steps"][0]["step"] == 4000
    assert state["policy_ready"]
    assert not state["replay_ready"]
    assert not state["capabilities"]["hardware_motion"]


def test_http_token_and_hardware_actions(console):
    app, base = console
    viewer = "b" * 32
    body = json.dumps({"viewer": viewer}).encode()
    request = urllib.request.Request(base + "/api/server_start", data=body, method="POST")
    with pytest.raises(urllib.error.HTTPError) as error:
        urlopen(request)
    assert error.value.code == 403
    request = urllib.request.Request(
        base + "/api/policy", data=body, method="POST", headers={"X-Replay-Token": app.token}
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        urlopen(request)
    assert error.value.code == 409
    assert app.server.process is None


def test_unknown_static_path_and_non_loopback_host_are_rejected(console):
    _, base = console
    for suffix in ["/static/../../pyproject.toml", "/reference/main/first"]:
        with pytest.raises(urllib.error.HTTPError) as error:
            urlopen(base + suffix)
        assert error.value.code == 404
    request = urllib.request.Request(base + "/api/state", headers={"Host": "unrelated.example"})
    with pytest.raises(urllib.error.HTTPError) as error:
        urlopen(request)
    assert error.value.code == 403


def test_parameter_bounds_and_velocity_mode():
    base = {"horizon": 30, "execute_steps": 15, "max_chunks": 1, "control_hz": 30}
    assert Console.validate_settings(base)["velocity_mode"] == "urdf"
    for update in [
        {"horizon": True},
        {"execute_steps": 31},
        {"control_hz": 60},
        {"max_chunks": 21},
        {"velocity_mode": "unlimited"},
    ]:
        with pytest.raises(ValueError, match=r"参数|需要|只读|velocity"):
            Console.validate_settings(dict(base, **update))


def test_selection_clears_previous_predictions_and_counts(console):
    app, _ = console
    app.completed = 3
    app.emit("prediction", index=0, actions=[[1.0] * 14])
    previous_run = app.run
    app.act("select", {"viewer": "c" * 32, "checkpoint_id": "task/M0/4000", "sample": 0})
    assert app.operation.acquire(timeout=2)
    app.operation.release()
    state = app.snapshot()
    assert state["completed_chunks"] == 0
    assert app.run != previous_run
    assert not app.telemetry_snapshot(0, previous_run)["events"]
