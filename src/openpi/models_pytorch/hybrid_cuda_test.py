"""Small real-CUDA compatibility checks; these do not load policy checkpoints."""

import copy
import sys

import pytest
import torch

from openpi.models_pytorch.gemma_pytorch_test import _tiny_joint_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hybrid_flags_use_cuda_reference_paths(dtype):
    torch.manual_seed(42)
    baseline = _tiny_joint_model(head_dim=64).to(device="cuda", dtype=dtype).eval()
    optimized = copy.deepcopy(baseline)
    optimized.fuse_qkv = True
    optimized.use_grouped_mlp = True
    optimized.use_npu_fusion_attention = True
    inputs = [torch.randn(1, n, 16, device="cuda", dtype=dtype) for n in (3, 2)]
    condition = torch.randn(1, 16, device="cuda", dtype=dtype)
    outputs = []
    with torch.inference_mode():
        for model in (baseline, optimized):
            result, _ = model(
                inputs_embeds=inputs,
                attention_mask=torch.zeros(1, 1, 5, 5, device="cuda"),
                position_ids=torch.arange(5, device="cuda")[None],
                adarms_cond=[None, condition],
            )
            outputs.append(result)
    for a, b in zip(*outputs, strict=True):
        assert torch.isfinite(b).all()
        torch.testing.assert_close(
            a, b, atol=0.03 if dtype == torch.bfloat16 else 1e-5, rtol=0.03 if dtype == torch.bfloat16 else 1e-5
        )
    assert "torch_npu" not in sys.modules
