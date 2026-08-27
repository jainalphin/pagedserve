"""Triton residual-add plus LayerNorm used by the decode hot path."""

import torch
import triton
import triton.language as tl


@triton.jit
def _residual_layernorm_kernel(
    residual,
    update,
    weight,
    bias,
    residual_output,
    normalized_output,
    row_stride,
    width: tl.constexpr,
    padded_width: tl.constexpr,
    epsilon: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, padded_width)
    mask = columns < width
    offsets = row * row_stride + columns
    summed = (
        tl.load(residual + offsets, mask=mask, other=0.0).to(tl.float32)
        + tl.load(update + offsets, mask=mask, other=0.0).to(tl.float32)
    )
    mean = tl.sum(summed, axis=0) / width
    centered = tl.where(mask, summed - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / width
    inverse_std = tl.rsqrt(variance + epsilon)
    scale = tl.load(weight + columns, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(bias + columns, mask=mask, other=0.0).to(tl.float32)
    tl.store(residual_output + offsets, summed, mask=mask)
    tl.store(
        normalized_output + offsets,
        centered * inverse_std * scale + shift,
        mask=mask,
    )


def fused_residual_layer_norm(
    residual: torch.Tensor,
    update: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    epsilon: float,
):
    """Return both residual+update and its FP32-accumulated LayerNorm."""
    if not residual.is_cuda:
        raise ValueError("Triton residual LayerNorm requires CUDA")
    if residual.shape != update.shape or residual.dtype != update.dtype:
        raise ValueError("Residual and update tensors must match")
    if residual.ndim < 2 or not residual.is_contiguous() or not update.is_contiguous():
        raise ValueError("Residual and update tensors must be contiguous matrices")
    width = residual.shape[-1]
    if weight.shape != (width,) or bias.shape != (width,):
        raise ValueError("LayerNorm parameters do not match the hidden dimension")
    if weight.device != residual.device or bias.device != residual.device:
        raise ValueError("LayerNorm tensors must share one CUDA device")
    if weight.dtype != residual.dtype or bias.dtype != residual.dtype:
        raise ValueError("LayerNorm parameters must match the activation dtype")
    if width > 65536:
        raise ValueError("Hidden dimension is too large for fused LayerNorm")

    flattened = residual.reshape(-1, width)
    flattened_update = update.reshape(-1, width)
    residual_output = torch.empty_like(flattened)
    normalized_output = torch.empty_like(flattened)
    padded_width = triton.next_power_of_2(width)
    _residual_layernorm_kernel[(flattened.shape[0],)](
        flattened,
        flattened_update,
        weight,
        bias,
        residual_output,
        normalized_output,
        flattened.stride(0),
        width=width,
        padded_width=padded_width,
        epsilon=epsilon,
        num_warps=4 if padded_width <= 2048 else 8,
    )
    return (
        residual_output.reshape(residual.shape),
        normalized_output.reshape(residual.shape),
    )
