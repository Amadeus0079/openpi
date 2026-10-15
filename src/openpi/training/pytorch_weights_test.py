import pytest
import safetensors.torch
import torch

from openpi.training.pytorch_weights import load_fp32_weights
from openpi.training.pytorch_weights import save_fp32_weights
from openpi.training.pytorch_weights import shared_weight_aliases


@pytest.mark.parametrize("initial_dtype", [torch.bfloat16, torch.float32])
def test_resume_retains_fp32_master_weights(tmp_path, initial_dtype):
    original = torch.nn.Linear(2, 2)
    with torch.no_grad():
        original.weight.copy_(torch.tensor([[1.0012345, -0.00234567], [0.12345678, -2.0012345]]))
        original.bias.copy_(torch.tensor([0.0001234567, -0.9876543]))
    assert not torch.equal(original.weight, original.weight.to(torch.bfloat16).float())
    checkpoint = tmp_path / "model.safetensors"
    safetensors.torch.save_model(original, checkpoint)

    resumed = torch.nn.Linear(2, 2).to(initial_dtype)
    load_fp32_weights(resumed, checkpoint)

    for name, expected in original.state_dict().items():
        actual = resumed.state_dict()[name]
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class TiedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(3, 2)
        self.lm_head = torch.nn.Linear(2, 3, bias=False)
        self.lm_head.weight = self.embed_tokens.weight


def test_fsdp_checkpoint_loads_with_tied_embedding_and_fp32_precision(tmp_path):
    original = TiedModel()
    with torch.no_grad():
        original.embed_tokens.weight.copy_(torch.tensor([[1.0012345, 2.012345], [3.123456, 4.001234], [5.123, 6.2345]]))
    aliases = shared_weight_aliases(original)
    assert aliases == {"lm_head.weight": "embed_tokens.weight"}
    # FSDP may return copies that no longer share backing storage.
    state = {name: value.clone() for name, value in original.state_dict().items()}
    checkpoint = tmp_path / "model.safetensors"
    save_fp32_weights(state, checkpoint, aliases)
    resumed = TiedModel().bfloat16()
    load_fp32_weights(resumed, checkpoint)
    assert resumed.embed_tokens.weight is resumed.lm_head.weight
    torch.testing.assert_close(resumed.embed_tokens.weight, original.embed_tokens.weight, rtol=0, atol=0)


def test_rejects_conflicting_copies_of_tied_weights(tmp_path):
    model = TiedModel()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    state["lm_head.weight"].add_(1)
    with pytest.raises(ValueError, match="Tied checkpoint weights differ"):
        save_fp32_weights(state, tmp_path / "invalid.safetensors", shared_weight_aliases(model))
