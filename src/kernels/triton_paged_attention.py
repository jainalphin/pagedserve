"""Fused Triton paged attention for single-token decode."""

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_attention_kernel(
    queries,
    key_pool,
    value_pool,
    block_table,
    context_lengths,
    output,
    layer_id,
    scale,
    query_stride_batch,
    query_stride_head,
    query_stride_dim,
    key_stride_layer,
    key_stride_block,
    key_stride_head,
    key_stride_token,
    key_stride_dim,
    value_stride_layer,
    value_stride_block,
    value_stride_head,
    value_stride_token,
    value_stride_dim,
    table_stride_batch,
    table_stride_block,
    context_length_stride,
    output_stride_batch,
    output_stride_head,
    output_stride_dim,
    KV_BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
):
    """Compute one decode-attention output for each request and head."""
    request_index = tl.program_id(0)
    head_index = tl.program_id(1)

    dimension_offsets = tl.arange(0, PADDED_HEAD_DIM)
    dimension_mask = dimension_offsets < HEAD_DIM
    token_offsets = tl.arange(0, KV_BLOCK_SIZE)

    query_offsets = (
        request_index * query_stride_batch
        + head_index * query_stride_head
        + dimension_offsets * query_stride_dim
    )
    query = tl.load(
        queries + query_offsets,
        mask=dimension_mask,
        other=0.0,
    ).to(tl.float32)

    context_length = tl.load(
        context_lengths + request_index * context_length_stride
    )
    logical_block_count = tl.cdiv(context_length, KV_BLOCK_SIZE)
    # Triton may specialize layer IDs 0 and 1 as Python integers. Adding the
    # value to an int64 tensor works for both specialized constants and runtime
    # scalar arguments while keeping all subsequent pool offsets in int64.
    layer_index_64 = tl.zeros((1,), dtype=tl.int64) + layer_id

    # Online-softmax state. Keeping it in FP32 avoids accumulating decode
    # probabilities and weighted values in the cache's lower precision dtype.
    running_max = tl.full((1,), -float("inf"), dtype=tl.float32)
    running_sum = tl.zeros((1,), dtype=tl.float32)
    accumulator = tl.zeros((PADDED_HEAD_DIM,), dtype=tl.float32)

    for logical_block in tl.range(0, logical_block_count):
        physical_block = tl.load(
            block_table
            + request_index * table_stride_batch
            + logical_block * table_stride_block
        )
        # Large KV pools can make layer_id * layer_stride exceed int32 even
        # though each individual stride and block ID fits. Cast before doing
        # either multiplication so pointer offsets cannot wrap.
        physical_block_64 = physical_block.to(tl.int64)
        token_positions = logical_block * KV_BLOCK_SIZE + token_offsets
        token_mask = token_positions < context_length

        key_offsets = (
            layer_index_64 * key_stride_layer
            + physical_block_64 * key_stride_block
            + head_index * key_stride_head
            + token_offsets[:, None] * key_stride_token
            + dimension_offsets[None, :] * key_stride_dim
        )
        key_mask = token_mask[:, None] & dimension_mask[None, :]
        keys = tl.load(
            key_pool + key_offsets,
            mask=key_mask,
            other=0.0,
        ).to(tl.float32)

        scores = tl.sum(keys * query[None, :], axis=1) * scale
        scores = tl.where(token_mask, scores, -float("inf"))

        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, block_max)
        previous_correction = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max)

        value_offsets = (
            layer_index_64 * value_stride_layer
            + physical_block_64 * value_stride_block
            + head_index * value_stride_head
            + token_offsets[:, None] * value_stride_token
            + dimension_offsets[None, :] * value_stride_dim
        )
        values = tl.load(
            value_pool + value_offsets,
            mask=key_mask,
            other=0.0,
        ).to(tl.float32)

        accumulator = (
            accumulator * previous_correction
            + tl.sum(probabilities[:, None] * values, axis=0)
        )
        running_sum = (
            running_sum * previous_correction
            + tl.sum(probabilities, axis=0)
        )
        running_max = new_max

    output_offsets = (
        request_index * output_stride_batch
        + head_index * output_stride_head
        + dimension_offsets * output_stride_dim
    )
    tl.store(
        output + output_offsets,
        accumulator / running_sum,
        mask=dimension_mask,
    )


def paged_decode_attention_triton(
    queries: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    block_table: torch.Tensor,
    context_lengths: torch.Tensor,
    layer_id: int,
) -> torch.Tensor:
    """Run fused paged attention for a batch of single-token decode queries."""
    if not queries.is_cuda:
        raise ValueError("Triton paged attention requires CUDA tensors")
    tensors = (key_pool, value_pool, block_table, context_lengths)
    if any(tensor.device != queries.device for tensor in tensors):
        raise ValueError("All Triton paged-attention tensors must share one device")
    if queries.ndim != 3:
        raise ValueError("Queries must have shape [batch, heads, head_dim]")
    if key_pool.ndim != 5 or value_pool.shape != key_pool.shape:
        raise ValueError("KV pools must have matching 5D shapes")
    if block_table.ndim != 2:
        raise ValueError("Block table must have shape [batch, logical_blocks]")
    if context_lengths.ndim != 1:
        raise ValueError("Context lengths must have shape [batch]")

    batch_size, num_heads, head_dim = queries.shape
    if block_table.shape[0] != batch_size or context_lengths.shape[0] != batch_size:
        raise ValueError("Decode metadata batch size does not match queries")
    if key_pool.shape[2] != num_heads or key_pool.shape[4] != head_dim:
        raise ValueError("Query heads and dimensions must match the KV cache")
    if queries.dtype != key_pool.dtype or value_pool.dtype != key_pool.dtype:
        raise ValueError("Queries and KV pools must have matching dtypes")
    if queries.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Triton paged attention requires a floating-point dtype")
    if block_table.dtype not in (torch.int32, torch.int64):
        raise ValueError("Block table must use int32 or int64")
    if context_lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("Context lengths must use int32 or int64")
    if not 0 <= layer_id < key_pool.shape[0]:
        raise ValueError("Invalid layer_id")
    if head_dim > 256:
        raise ValueError("Triton paged attention currently supports head_dim <= 256")

    kv_block_size = key_pool.shape[3]
    if kv_block_size <= 0 or kv_block_size & (kv_block_size - 1):
        raise ValueError("KV block size must be a power of two")

    padded_head_dim = triton.next_power_of_2(head_dim)
    output = torch.empty(
        (batch_size, num_heads, head_dim),
        dtype=queries.dtype,
        device=queries.device,
    )
    grid = (batch_size, num_heads)

    _paged_decode_attention_kernel[grid](
        queries,
        key_pool,
        value_pool,
        block_table,
        context_lengths,
        output,
        layer_id,
        head_dim ** -0.5,
        *queries.stride(),
        *key_pool.stride(),
        *value_pool.stride(),
        *block_table.stride(),
        *context_lengths.stride(),
        *output.stride(),
        KV_BLOCK_SIZE=kv_block_size,
        HEAD_DIM=head_dim,
        PADDED_HEAD_DIM=padded_head_dim,
        num_warps=4,
    )
    return output
