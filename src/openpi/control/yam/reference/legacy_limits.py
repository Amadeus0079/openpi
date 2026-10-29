"""Read the local collection velocity rule without importing robot hardware code."""

import ast
import hashlib

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8 << 20), b""):
            digest.update(data)
    return digest.hexdigest()


def collection_speed_limits(owner):
    """Read the original collection daemon's dt-scaled arm slew rule."""
    controller = owner / "gello/safety/controller.py"
    launcher = owner / "experiments/yam_safety_daemon.py"
    tree = ast.parse(controller.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SafetyController")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    defaults = dict(zip([x.arg for x in init.args.kwonlyargs], init.args.kw_defaults, strict=True))
    step = float(ast.literal_eval(defaults["arm_step_limit_rad"]))
    factors = [
        float(n.value.left.value)
        for n in ast.walk(cls)
        if isinstance(n, ast.AugAssign)
        and isinstance(n.target, ast.Name)
        and n.target.id == "step_limit"
        and isinstance(n.op, ast.Mult)
        and isinstance(n.value, ast.BinOp)
        and isinstance(n.value.op, ast.Mult)
        and isinstance(n.value.left, ast.Constant)
        and isinstance(n.value.right, ast.Name)
        and n.value.right.id == "dt"
    ]
    calls = [
        n
        for n in ast.walk(ast.parse(launcher.read_text()))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "SafetyController"
    ]
    if len(factors) != 1 or len(calls) != 1 or any(k.arg not in ("robot", "metadata") for k in calls[0].keywords):
        raise ValueError("Original YAM collection speed rule changed; recheck its configuration")
    hz = factors[0]
    if not np.isfinite(step * hz) or step <= 0 or hz <= 0:
        raise ValueError("Invalid original collection slew rule")
    return {
        "arm_step_limit_rad_at_reference_hz": step,
        "reference_hz": hz,
        "arm_velocity_limit_rad_s": step * hz,
        "source": str(controller),
        "source_sha256": sha(controller),
        "daemon_launcher_sha256": sha(launcher),
        "rule": "step_limit *= reference_hz * min(1/reference_hz, elapsed_dt)",
    }
