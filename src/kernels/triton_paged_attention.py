"""Fused Triton paged attention for single-token decode."""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=[
        "KV_BLOCK_SIZE",
        "HEAD_DIM",
        "BATCH_BUCKET",
        "CONTEXT_BUCKET",
        "KV_QUANTIZED",
        "WRITE_KV",
    ],
)
@triton.jit
def _paged_decode_attention_kernel(
    queries,
    new_keys,
    new_values,
    key_pool,
    value_pool,
    key_scales,
    value_scales,
    block_table,
    context_lengths,
    reserved_block_ids,
    reserved_block_offsets,
    output,
    layer_id,
    scale,
    query_stride_batch,
    query_stride_head,
    query_stride_dim,
    new_key_stride_batch,
    new_key_stride_head,
    new_key_stride_dim,
    new_value_stride_batch,
    new_value_stride_head,
    new_value_stride_dim,
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
    key_scale_stride_layer,
    key_scale_stride_block,
    key_scale_stride_head,
    key_scale_stride_token,
    value_scale_stride_layer,
    value_scale_stride_block,
    value_scale_stride_head,
    value_scale_stride_token,
    table_stride_batch,
    table_stride_block,
    context_length_stride,
    reserved_block_id_stride,
    reserved_block_offset_stride,
    output_stride_batch,
    output_stride_head,
    output_stride_dim,
    KV_BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    KV_QUANTIZED: tl.constexpr,
    WRITE_KV: tl.constexpr,
    BATCH_BUCKET: tl.constexpr,
    CONTEXT_BUCKET: tl.constexpr,
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

    if WRITE_KV:
        reserved_block = tl.load(
            reserved_block_ids + request_index * reserved_block_id_stride
        ).to(tl.int64)
        reserved_offset = tl.load(
            reserved_block_offsets + request_index * reserved_block_offset_stride
        ).to(tl.int64)
        new_key_offsets = (
            request_index * new_key_stride_batch
            + head_index * new_key_stride_head
            + dimension_offsets * new_key_stride_dim
        )
        new_value_offsets = (
            request_index * new_value_stride_batch
            + head_index * new_value_stride_head
            + dimension_offsets * new_value_stride_dim
        )
        current_key = tl.load(
            new_keys + new_key_offsets,
            mask=dimension_mask,
            other=0.0,
        ).to(tl.float32)
        current_value = tl.load(
            new_values + new_value_offsets,
            mask=dimension_mask,
            other=0.0,
        ).to(tl.float32)
        current_key_pool_offsets = (
            layer_index_64 * key_stride_layer
            + reserved_block * key_stride_block
            + head_index * key_stride_head
            + reserved_offset * key_stride_token
            + dimension_offsets * key_stride_dim
        )
        current_value_pool_offsets = (
            layer_index_64 * value_stride_layer
            + reserved_block * value_stride_block
            + head_index * value_stride_head
            + reserved_offset * value_stride_token
            + dimension_offsets * value_stride_dim
        )
        if KV_QUANTIZED:
            key_scale = tl.max(tl.abs(current_key), axis=0) / 127.0
            value_scale = tl.max(tl.abs(current_value), axis=0) / 127.0
            key_scale = tl.where(key_scale > 0.0, key_scale, 1.0)
            value_scale = tl.where(value_scale > 0.0, value_scale, 1.0)
            scaled_key = current_key / key_scale
            scaled_value = current_value / value_scale
            rounded_key = tl.where(
                scaled_key >= 0.0,
                tl.floor(scaled_key + 0.5),
                tl.ceil(scaled_key - 0.5),
            )
            rounded_value = tl.where(
                scaled_value >= 0.0,
                tl.floor(scaled_value + 0.5),
                tl.ceil(scaled_value - 0.5),
            )
            quantized_key = tl.maximum(-127.0, tl.minimum(127.0, rounded_key))
            quantized_value = tl.maximum(-127.0, tl.minimum(127.0, rounded_value))
            current_key_for_attention = quantized_key * key_scale
            current_value_for_attention = quantized_value * value_scale
            tl.store(
                key_pool + current_key_pool_offsets,
                quantized_key,
                mask=dimension_mask,
            )
            tl.store(
                value_pool + current_value_pool_offsets,
                quantized_value,
                mask=dimension_mask,
            )
            current_key_scale_offset = (
                layer_index_64 * key_scale_stride_layer
                + reserved_block * key_scale_stride_block
                + head_index * key_scale_stride_head
                + reserved_offset * key_scale_stride_token
            )
            current_value_scale_offset = (
                layer_index_64 * value_scale_stride_layer
                + reserved_block * value_scale_stride_block
                + head_index * value_scale_stride_head
                + reserved_offset * value_scale_stride_token
            )
            tl.store(key_scales + current_key_scale_offset, key_scale)
            tl.store(value_scales + current_value_scale_offset, value_scale)
        else:
            current_key_for_attention = current_key
            current_value_for_attention = current_value
            tl.store(
                key_pool + current_key_pool_offsets,
                current_key,
                mask=dimension_mask,
            )
            tl.store(
                value_pool + current_value_pool_offsets,
                current_value,
                mask=dimension_mask,
            )

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
        if KV_QUANTIZED:
            key_scale_offsets = (
                layer_index_64 * key_scale_stride_layer
                + physical_block_64 * key_scale_stride_block
                + head_index * key_scale_stride_head
                + token_offsets * key_scale_stride_token
            )
            key_scale = tl.load(
                key_scales + key_scale_offsets,
                mask=token_mask,
                other=0.0,
            ).to(tl.float32)
            keys = keys * key_scale[:, None]
        if WRITE_KV:
            current_token_mask = token_positions == context_length - 1
            keys = tl.where(
                current_token_mask[:, None],
                current_key_for_attention[None, :],
                keys,
            )

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
        if KV_QUANTIZED:
            value_scale_offsets = (
                layer_index_64 * value_scale_stride_layer
                + physical_block_64 * value_scale_stride_block
                + head_index * value_scale_stride_head
                + token_offsets * value_scale_stride_token
            )
            value_scale = tl.load(
                value_scales + value_scale_offsets,
                mask=token_mask,
                other=0.0,
            ).to(tl.float32)
            values = values * value_scale[:, None]
        if WRITE_KV:
            values = tl.where(
                current_token_mask[:, None],
                current_value_for_attention[None, :],
                values,
            )

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
    key_scales: torch.Tensor,
    value_scales: torch.Tensor,
    block_table: torch.Tensor,
    context_lengths: torch.Tensor,
    layer_id: int,
    new_keys: torch.Tensor = None,
    new_values: torch.Tensor = None,
    reserved_block_ids: torch.Tensor = None,
    reserved_block_offsets: torch.Tensor = None,
    output: torch.Tensor = None,
    maximum_context_length: int = None,
    validate_inputs: bool = True,
) -> torch.Tensor:
    """Run fused paged attention for a batch of single-token decode queries."""
    if not queries.is_cuda:
        raise ValueError("Triton paged attention requires CUDA tensors")
    tensors = (
        key_pool,
        value_pool,
        key_scales,
        value_scales,
        block_table,
        context_lengths,
    )
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
    write_kv = new_keys is not None or new_values is not None
    if write_kv:
        if new_keys is None or new_values is None:
            raise ValueError("New keys and values must be provided together")
        if reserved_block_ids is None or reserved_block_offsets is None:
            raise ValueError("Fused K/V writes require reserved block metadata")
        if new_keys.shape != queries.shape or new_values.shape != queries.shape:
            raise ValueError("New K/V tensors must match the query shape")
        if new_keys.dtype != queries.dtype or new_values.dtype != queries.dtype:
            raise ValueError("New K/V tensors must match the query dtype")
        write_tensors = (
            new_keys,
            new_values,
            reserved_block_ids,
            reserved_block_offsets,
        )
        if any(tensor.device != queries.device for tensor in write_tensors):
            raise ValueError("Fused K/V write tensors must share the query device")
        if reserved_block_ids.shape != (batch_size,):
            raise ValueError("Reserved block IDs must match the decode batch")
        if reserved_block_offsets.shape != (batch_size,):
            raise ValueError("Reserved block offsets must match the decode batch")
        if reserved_block_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Reserved block IDs must be integer tensors")
        if reserved_block_offsets.dtype not in (torch.int32, torch.int64):
            raise ValueError("Reserved block offsets must be integer tensors")
    if block_table.shape[0] != batch_size or context_lengths.shape[0] != batch_size:
        raise ValueError("Decode metadata batch size does not match queries")
    if key_pool.shape[2] != num_heads or key_pool.shape[4] != head_dim:
        raise ValueError("Query heads and dimensions must match the KV cache")
    if queries.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Triton paged attention requires a floating-point dtype")
    if value_pool.dtype != key_pool.dtype:
        raise ValueError("Key and value pools must have matching dtypes")
    kv_quantized = key_pool.dtype == torch.int8
    if not kv_quantized and queries.dtype != key_pool.dtype:
        raise ValueError("Floating-point queries and KV pools must match dtypes")
    if kv_quantized:
        expected_scale_shape = key_pool.shape[:-1]
        if key_scales.shape != expected_scale_shape or value_scales.shape != expected_scale_shape:
            raise ValueError("INT8 scale pools must match KV pools without head_dim")
        if key_scales.dtype != torch.float32 or value_scales.dtype != torch.float32:
            raise ValueError("INT8 KV scales must use FP32")
    elif key_pool.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("KV pools must use a supported floating-point dtype or INT8")
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

    if validate_inputs:
        lengths = context_lengths.to(torch.int64)
        if bool((lengths <= 0).any().item()):
            raise ValueError("Context lengths must be positive")
        if bool((lengths > block_table.shape[1] * kv_block_size).any().item()):
            raise ValueError("A context length exceeds its block-table capacity")
        logical_columns = torch.arange(
            block_table.shape[1], device=block_table.device
        ).unsqueeze(0)
        used_blocks = logical_columns < torch.div(
            lengths + kv_block_size - 1, kv_block_size, rounding_mode="floor"
        ).unsqueeze(1)
        used_physical_blocks = block_table[used_blocks]
        if bool(
            (
                (used_physical_blocks < 0)
                | (used_physical_blocks >= key_pool.shape[1])
            ).any().item()
        ):
            raise ValueError("Block table contains an invalid physical block ID")
        if write_kv:
            reserved_ids_long = reserved_block_ids.to(torch.int64)
            reserved_offsets_long = reserved_block_offsets.to(torch.int64)
            if bool(
                (
                    (reserved_ids_long < 0)
                    | (reserved_ids_long >= key_pool.shape[1])
                ).any().item()
            ):
                raise ValueError("A reserved physical block ID is invalid")
            if bool(
                (
                    (reserved_offsets_long < 0)
                    | (reserved_offsets_long >= kv_block_size)
                ).any().item()
            ):
                raise ValueError("A reserved block offset is invalid")
            batch_rows = torch.arange(batch_size, device=block_table.device)
            expected_ids = block_table[
                batch_rows,
                (lengths - 1) // kv_block_size,
            ].to(torch.int64)
            expected_offsets = (lengths - 1) % kv_block_size
            if bool((reserved_ids_long != expected_ids).any().item()):
                raise ValueError("Reserved block IDs do not match the block table")
            if bool((reserved_offsets_long != expected_offsets).any().item()):
                raise ValueError("Reserved offsets do not match context lengths")
        max_context = int(lengths.max().item())
        positions = torch.arange(max_context, device=block_table.device)
        cached_lengths = lengths - (1 if write_kv else 0)
        valid_tokens = positions.unsqueeze(0) < cached_lengths.unsqueeze(1)
        batch_indices = torch.arange(
            batch_size, device=block_table.device
        ).unsqueeze(1).expand_as(valid_tokens)[valid_tokens]
        token_indices = positions.unsqueeze(0).expand_as(valid_tokens)[valid_tokens]
        physical_indices = block_table[
            batch_indices, token_indices // kv_block_size
        ].to(torch.long)
        block_offsets = token_indices % kv_block_size
        if kv_quantized:
            used_finite_tensors = (
                key_scales[layer_id, physical_indices, :, block_offsets],
                value_scales[layer_id, physical_indices, :, block_offsets],
            )
        else:
            used_finite_tensors = (
                key_pool[layer_id, physical_indices, :, block_offsets, :],
                value_pool[layer_id, physical_indices, :, block_offsets, :],
            )
        finite_inputs = [queries]
        if write_kv:
            finite_inputs.extend((new_keys, new_values))
        if any(not bool(torch.isfinite(tensor).all().item()) for tensor in finite_inputs) or any(
            not bool(torch.isfinite(tensor).all().item())
            for tensor in used_finite_tensors
        ):
            raise ValueError("Paged-attention inputs contain NaN or Inf")

    padded_head_dim = triton.next_power_of_2(head_dim)
    expected_output_shape = (batch_size, num_heads, head_dim)
    if output is None:
        output = torch.empty(
            expected_output_shape,
            dtype=queries.dtype,
            device=queries.device,
        )
    elif (
        output.shape != expected_output_shape
        or output.dtype != queries.dtype
        or output.device != queries.device
    ):
        raise ValueError("Output buffer must match query shape, dtype, and device")
    if maximum_context_length is None:
        maximum_context_length = int(context_lengths.max().item())
    if maximum_context_length <= 0:
        raise ValueError("maximum_context_length must be positive")
    batch_bucket = triton.next_power_of_2(batch_size)
    context_bucket = triton.next_power_of_2(maximum_context_length)
    grid = (batch_size, num_heads)
    key_scale_strides = key_scales.stride() if kv_quantized else (0, 0, 0, 0)
    value_scale_strides = value_scales.stride() if kv_quantized else (0, 0, 0, 0)
    kernel_new_keys = new_keys if write_kv else queries
    kernel_new_values = new_values if write_kv else queries
    kernel_reserved_block_ids = (
        reserved_block_ids if write_kv else context_lengths
    )
    kernel_reserved_block_offsets = (
        reserved_block_offsets if write_kv else context_lengths
    )

    _paged_decode_attention_kernel[grid](
        queries,
        kernel_new_keys,
        kernel_new_values,
        key_pool,
        value_pool,
        key_scales,
        value_scales,
        block_table,
        context_lengths,
        kernel_reserved_block_ids,
        kernel_reserved_block_offsets,
        output,
        layer_id,
        head_dim ** -0.5,
        *queries.stride(),
        *kernel_new_keys.stride(),
        *kernel_new_values.stride(),
        *key_pool.stride(),
        *value_pool.stride(),
        *key_scale_strides,
        *value_scale_strides,
        *block_table.stride(),
        *context_lengths.stride(),
        *kernel_reserved_block_ids.stride(),
        *kernel_reserved_block_offsets.stride(),
        *output.stride(),
        KV_BLOCK_SIZE=kv_block_size,
        HEAD_DIM=head_dim,
        PADDED_HEAD_DIM=padded_head_dim,
        KV_QUANTIZED=kv_quantized,
        WRITE_KV=write_kv,
        BATCH_BUCKET=batch_bucket,
        CONTEXT_BUCKET=context_bucket,
    )
    return output
