import copy
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openpi.models import gemma
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch import ascend_optimizations as ascend
from openpi.models_pytorch import pi0_pytorch


def test_grouped_pair_gradcheck():
    inputs = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    left = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
    right = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(ascend._GroupedLinearPair.apply, (inputs, left, right))  # noqa: SLF001


@pytest.mark.parametrize("bias", [True, False])
def test_fused_qkv_preserves_parameters_outputs_and_gradients(bias):
    attention = nn.Module()
    attention.q_proj = nn.Linear(16, 32, bias=bias)
    attention.k_proj = nn.Linear(16, 8, bias=bias)
    attention.v_proj = nn.Linear(16, 8, bias=bias)
    fused = copy.deepcopy(attention)
    inputs = torch.randn(2, 5, 16)
    outputs = []
    for model, fuse in ((attention, False), (fused, True)):
        hidden = inputs.clone().requires_grad_()
        result = ascend.project_qkv(model, hidden, fuse=fuse)
        sum(x.square().sum() for x in result).backward()
        outputs.append((result, hidden.grad))
    assert attention.state_dict().keys() == fused.state_dict().keys()
    for a, b in zip(outputs[0][0], outputs[1][0], strict=True):
        torch.testing.assert_close(a, b)
    torch.testing.assert_close(outputs[0][1], outputs[1][1])
    for a, b in zip(attention.parameters(), fused.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad)


def test_beta_inverse_cdf_and_distribution():
    torch.manual_seed(1)
    expected = torch.rand(100000).pow(2 / 3)
    torch.manual_seed(1)
    actual = pi0_pytorch.sample_beta(1.5, 1.0, 100000, "cpu")
    torch.testing.assert_close(actual, expected)
    assert abs(actual.mean().item() - 0.6) < 0.005
    assert (actual >= 0).all()
    assert (actual <= 1).all()


@pytest.mark.parametrize("pi05", [False, True])
def test_cached_time_and_masks_keep_checkpoint_format(monkeypatch, pi05):
    monkeypatch.setattr(gemma, "get_config", lambda _: SimpleNamespace(width=16))
    monkeypatch.setattr(pi0_pytorch, "PaliGemmaWithExpertModel", lambda *a, **kw: nn.Module())
    model = pi0_pytorch.PI0Pytorch(Pi0Config(pi05=pi05, action_horizon=3, pytorch_compile_mode=None))
    time = torch.tensor([0.01, 0.7])
    sine_input = model.time_pos_scaling_factor[None, :] * time[:, None]
    cached = torch.cat((sine_input.sin(), sine_input.cos()), dim=-1)
    original = pi0_pytorch.create_sinusoidal_pos_embedding(time, 16, 4e-3, 4.0, time.device)
    torch.testing.assert_close(cached.double(), original, atol=1e-4, rtol=1e-4)
    _, _, mask, _ = model.embed_suffix(torch.randn(2, 32), torch.randn(2, 3, 32), time)
    expected = [1, 0, 0] if pi05 else [1, 1, 0, 0]
    assert mask.tolist() == [expected, expected]
    assert "time_pos_scaling_factor" not in model.state_dict()
    assert "suffix_att_mask_template" not in model.state_dict()


@pytest.mark.skipif(os.getenv("OPENPI_TEST_NPU") != "1", reason="Opt-in native NPU numerical checks")
@pytest.mark.parametrize("width", [64, 256])
def test_native_fusion_attention_forward_backward(width):
    import torch_npu  # noqa: F401, PLC0415 -- registers native NPU operators

    torch.npu.set_device(0)
    torch.manual_seed(3)
    # Prefix + suffix block mask with real padding, not just causal attention.
    pad = torch.ones(2, 40, dtype=torch.bool)
    pad[:, 25:28] = False
    blocks = torch.zeros_like(pad)
    blocks[:, 28] = True
    allowed = pi0_pytorch.make_att_2d_masks(pad, blocks)[:, None]
    mask = torch.where(allowed, 0.0, -2.3819763e38).to("npu")
    originals = [torch.randn(2, heads, 40, width, device="npu", dtype=torch.bfloat16) for heads in (8, 1, 1)]
    outputs, gradients = [], []
    valid = pad.to("npu")[:, :, None, None]
    for fused in (False, True):
        q, k, v = [x.clone().requires_grad_() for x in originals]
        if fused:
            out = ascend.npu_fusion_attention(q, k, v, mask, width**-0.5)
        else:
            # Eager Gemma accumulates softmax in FP32 then returns BF16.
            scores = (q @ ascend.repeat_kv_heads(k, 8).transpose(-1, -2)) * width**-0.5 + mask
            probabilities = scores.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
            out = (probabilities @ ascend.repeat_kv_heads(v, 8)).transpose(1, 2)
        (out.float().square() * valid).mean().backward()
        outputs.append(out.detach() * valid)
        gradients.append([x.grad for x in (q, k, v)])
    torch.testing.assert_close(outputs[0], outputs[1], atol=0.025, rtol=0.025)
    for a, b in zip(gradients[0], gradients[1], strict=True):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=0.08)


@pytest.mark.skipif(os.getenv("OPENPI_TEST_NPU") != "1", reason="Opt-in native NPU numerical checks")
def test_native_grouped_mlp_backward():
    import torch_npu  # noqa: F401, PLC0415

    torch.npu.set_device(0)
    originals = [
        torch.randn(*shape, device="npu", dtype=torch.bfloat16) / 8 for shape in ((2, 19, 64), (128, 64), (128, 64))
    ]
    outputs, gradients = [], []
    for grouped in (False, True):
        inputs, left, right = [x.clone().requires_grad_() for x in originals]
        if grouped:
            a, b = ascend._GroupedLinearPair.apply(inputs, left, right)  # noqa: SLF001
        else:
            a, b = inputs @ left.T, inputs @ right.T
        out = torch.nn.functional.gelu(a) * b
        out.float().square().mean().backward()
        outputs.append(out.detach())
        gradients.append([x.grad for x in (inputs, left, right)])
    torch.testing.assert_close(outputs[0], outputs[1], atol=0.01, rtol=0.03)
    for a, b in zip(gradients[0], gradients[1], strict=True):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=0.04)
