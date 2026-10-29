# Adapted from cann-recipes-embodied-ai/manipulation/pi05/train (d5cac59).
# Copyright (c) 2026 Institute of Software, Chinese Academy of Sciences (ISCAS).
# Copyright (c) 2026 HUAWEI CORPORATION.
# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# Modified for OpenPI parameter names, native FSDP, and optional NPU imports.
"""NPU kernels adapted from cann-recipes-embodied-ai/pi05 (d5cac59).

Projections retain their original parameters and checkpoint names. CPU paths
are deliberately available for checking outputs and gradients without an NPU.
"""

import torch
import torch.nn.functional as F  # noqa: N812


def project_qkv(attention, hidden_states, *, fuse=False):
    projections = (attention.q_proj, attention.k_proj, attention.v_proj)
    if not fuse:
        return tuple(projection(hidden_states) for projection in projections)
    weights = torch.cat([projection.weight for projection in projections], dim=0)
    biases = [projection.bias for projection in projections]
    if any(bias is not None for bias in biases):
        if not all(bias is not None for bias in biases):
            raise ValueError("Fused QKV requires either all biases or no biases")
        bias = torch.cat(biases)
    else:
        bias = None
    output = F.linear(hidden_states, weights, bias)
    return output.split([projection.out_features for projection in projections], dim=-1)


def repeat_kv_heads(states, num_heads):
    batch, kv_heads, sequence, width = states.shape
    if num_heads == kv_heads:
        return states
    if num_heads % kv_heads:
        raise ValueError("Attention heads must be divisible by KV heads")
    return (
        states[:, :, None]
        .expand(batch, kv_heads, num_heads // kv_heads, sequence, width)
        .reshape(batch, num_heads, sequence, width)
    )


def npu_fusion_attention(query, key, value, attention_mask, scaling):
    import torch_npu  # noqa: PLC0415 -- optional dependency, used only on NPU

    key = repeat_kv_heads(key, query.shape[1])
    value = repeat_kv_heads(value, query.shape[1])
    # OpenPI: zero means attend; native Ascend: True means masked out.
    mask = attention_mask
    if mask is not None:
        mask = mask[..., : key.shape[-2]]
        if mask.dtype != torch.bool:
            mask = mask < 0
    result = torch_npu.npu_fusion_attention(
        query,
        key,
        value,
        head_num=query.shape[1],
        input_layout="BNSD",
        atten_mask=mask,
        scale=scaling,
        keep_prob=1.0,
        sparse_mode=1,
    )
    output = result[0] if isinstance(result, tuple) else result
    return output.transpose(1, 2).contiguous()


class _GroupedLinearPair(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, left_weight, right_weight):
        ctx.save_for_backward(inputs, left_weight, right_weight)
        if inputs.device.type != "npu":
            return F.linear(inputs, left_weight), F.linear(inputs, right_weight)
        import torch_npu  # noqa: PLC0415 -- optional dependency

        return tuple(torch_npu.npu_grouped_matmul([inputs, inputs], [left_weight.T, right_weight.T], group_type=-1))

    @staticmethod
    def backward(ctx, grad_left, grad_right):
        inputs, left_weight, right_weight = ctx.saved_tensors
        grad_inputs = grad_left @ left_weight + grad_right @ right_weight if ctx.needs_input_grad[0] else None
        inputs_2d = inputs.reshape(-1, inputs.shape[-1])
        grad_left_weight = grad_left.reshape(-1, grad_left.shape[-1]).T @ inputs_2d if ctx.needs_input_grad[1] else None
        grad_right_weight = (
            grad_right.reshape(-1, grad_right.shape[-1]).T @ inputs_2d if ctx.needs_input_grad[2] else None
        )
        return grad_inputs, grad_left_weight, grad_right_weight


def grouped_mlp(mlp, inputs):
    gate_weight, up_weight = mlp.gate_proj.weight, mlp.up_proj.weight
    if inputs.device.type == "npu" and torch.is_autocast_enabled("npu"):
        # DDP retains FP32 parameters. These explicit casts also preserve the
        # autograd path back to FP32 master weights, like autocast F.linear.
        dtype = torch.get_autocast_dtype("npu")
        inputs = inputs.to(dtype)
        gate_weight, up_weight = gate_weight.to(dtype), up_weight.to(dtype)
    gate, up = _GroupedLinearPair.apply(inputs, gate_weight, up_weight)
    return mlp.down_proj(mlp.act_fn(gate) * up)
