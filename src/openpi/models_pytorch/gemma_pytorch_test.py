import copy
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import DefaultDeviceType
from transformers import GemmaConfig
from transformers.models.gemma.modeling_gemma import GemmaModel

from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.training.pytorch_weights import load_fp32_weights
from openpi.training.pytorch_weights import save_fp32_weights
from openpi.training.pytorch_weights import shared_weight_aliases


def _tiny_joint_model(*, head_dim=2):
    config = GemmaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=head_dim,
        vocab_size=32,
        max_position_embeddings=32,
        use_adarms=False,
    )
    expert_config = copy.deepcopy(config)
    expert_config.use_adarms = True
    expert_config.adarms_cond_dim = 16
    joint = PaliGemmaWithExpertModel.__new__(PaliGemmaWithExpertModel)
    nn.Module.__init__(joint)
    joint.paligemma = nn.Module()
    joint.paligemma.config = SimpleNamespace(text_config=config)
    joint.paligemma.model = nn.Module()
    joint.paligemma.model.language_model = GemmaModel(config)
    joint.paligemma.language_model = joint.paligemma.model.language_model
    joint.gemma_expert = nn.Module()
    joint.gemma_expert.model = GemmaModel(expert_config)
    return joint.train()


def test_disabled_joint_checkpointing_stays_disabled_and_preserves_gradients(monkeypatch, tmp_path):
    # torch_npu can change this default even when all test tensors are on CPU.
    monkeypatch.setattr(DefaultDeviceType, "get_device_type", staticmethod(lambda: "cpu"))
    torch.manual_seed(7)
    disabled = _tiny_joint_model()
    enabled = copy.deepcopy(disabled)
    enabled.gemma_expert.model.gradient_checkpointing = True
    enabled.fuse_qkv = True
    enabled.use_grouped_mlp = True
    enabled.use_npu_fusion_attention = True  # CPU keeps the eager reference.
    original = torch.utils.checkpoint.checkpoint
    calls = []

    def counted_checkpoint(function, *args, **kwargs):
        calls.append(function)
        return original(function, *args, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", counted_checkpoint)
    inputs = [torch.randn(2, 3, 16), torch.randn(2, 2, 16)]
    cond = torch.randn(2, 16)
    outputs = []
    input_gradients = []
    for model in (disabled, enabled):
        rope_calls = []
        hook = model.paligemma.model.language_model.rotary_emb.register_forward_hook(
            lambda *args, calls=rope_calls: calls.append(True)
        )
        embeddings = [x.clone().requires_grad_() for x in inputs]
        result, _ = model(
            inputs_embeds=embeddings,
            attention_mask=torch.zeros(2, 1, 5, 5),
            position_ids=torch.arange(5)[None].expand(2, -1),
            adarms_cond=[None, cond],
        )
        sum(x.square().mean() for x in result).backward()
        assert len(rope_calls) == 1
        hook.remove()
        outputs.append(result)
        input_gradients.append([x.grad for x in embeddings])
        if model is disabled:
            assert not calls
            assert not model.gemma_expert.model.gradient_checkpointing
    assert calls
    for actual, expected in zip(outputs[0], outputs[1], strict=True):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(input_gradients[0], input_gradients[1], strict=True):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(disabled.parameters(), enabled.parameters(), strict=True):
        if actual.grad is None:
            assert expected.grad is None
        else:
            torch.testing.assert_close(actual.grad, expected.grad)
    # Optimized weights load through the ordinary, unfused model with no new keys.
    checkpoint = tmp_path / "model.safetensors"
    save_fp32_weights(enabled.state_dict(), checkpoint, shared_weight_aliases(enabled))
    restored = _tiny_joint_model()
    load_fp32_weights(restored, checkpoint)
    for name, weight in enabled.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], weight.float(), rtol=0, atol=0)


@pytest.mark.skipif(os.getenv("OPENPI_TEST_NPU") != "1", reason="Opt-in native NPU numerical checks")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_native_joint_fusions_with_layer_checkpointing(dtype):
    import torch_npu  # noqa: F401, PLC0415

    torch.npu.set_device(0)
    torch.manual_seed(9)
    baseline = _tiny_joint_model(head_dim=64).to(device="npu", dtype=dtype)
    optimized = copy.deepcopy(baseline)
    optimized.fuse_qkv = True
    optimized.use_grouped_mlp = True
    optimized.use_npu_fusion_attention = True
    inputs = [torch.randn(2, count, 16, device="npu", dtype=dtype) for count in (3, 2)]
    cond = torch.randn(2, 16, device="npu", dtype=dtype)
    results, gradients = [], []
    for model in (baseline, optimized):
        model.gemma_expert.model.gradient_checkpointing = True
        embeddings = [x.clone().requires_grad_() for x in inputs]
        with torch.autocast("npu", dtype=torch.bfloat16):
            output, _ = model(
                inputs_embeds=embeddings,
                attention_mask=torch.zeros(2, 1, 5, 5, device="npu"),
                position_ids=torch.arange(5, device="npu")[None].expand(2, -1),
                adarms_cond=[None, cond],
            )
        sum(x.float().square().mean() for x in output).backward()
        results.append(output)
        gradients.append([x.grad for x in embeddings])
    for a, b in zip(results[0], results[1], strict=True):
        torch.testing.assert_close(a, b, atol=0.04, rtol=0.04)
    for a, b in zip(gradients[0], gradients[1], strict=True):
        torch.testing.assert_close(a, b, atol=0.004, rtol=0.08)
    for a, b in zip(baseline.parameters(), optimized.parameters(), strict=True):
        if a.grad is None:
            assert b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, atol=0.01, rtol=0.08)
