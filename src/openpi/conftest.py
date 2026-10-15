import os

import pytest


def set_jax_cpu_backend_if_no_gpu() -> None:
    try:
        import pynvml  # noqa: PLC0415 -- NVIDIA support is optional on Ascend
    except ImportError:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        return
    try:
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        # No GPU found.
        os.environ["JAX_PLATFORMS"] = "cpu"


def pytest_configure(config: pytest.Config) -> None:
    set_jax_cpu_backend_if_no_gpu()
