"""Checkpoint IO for FSDP's FP32 master parameters and shared weights."""

from collections.abc import Mapping
from pathlib import Path

import safetensors.torch
import torch


def load_fp32_weights(model: torch.nn.Module, weight_path: str | Path) -> None:
    """Load directly into FP32 storage, preserving resumed master-weight precision.

    PI0Pytorch initializes many parameters as BF16. Casting after loading would
    round an FP32 training checkpoint to BF16 before widening it again.
    """
    model.float()
    safetensors.torch.load_model(model, weight_path)


def shared_weight_aliases(model: torch.nn.Module) -> dict[str, str]:
    """Identify semantic ties before FSDP turns parameters into flat-buffer views."""
    groups: dict[int, list[str]] = {}
    for name, value in model.state_dict(keep_vars=True).items():
        groups.setdefault(id(value), []).append(name)
    return {name: min(names) for names in groups.values() for name in names if len(names) > 1 and name != min(names)}


def save_fp32_weights(state: Mapping[str, torch.Tensor], weight_path: str | Path, aliases: Mapping[str, str]) -> None:
    """Save one entry per tied weight, compatible with safetensors.load_model.

    FSDP's CPU state dictionary can contain separate copies of the same model
    parameter. Storage-based deduplication cannot identify those semantic ties.
    """
    for alias, canonical in aliases.items():
        if alias not in state or canonical not in state or not torch.equal(state[alias], state[canonical]):
            raise ValueError(f"Tied checkpoint weights differ: {alias} and {canonical}")
    tensors = {name: value.contiguous().clone() for name, value in state.items() if name not in aliases}
    safetensors.torch.save_file(tensors, weight_path, metadata=dict(aliases))
