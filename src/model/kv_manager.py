import math

import torch
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Tuple, Union, Optional, Set


@dataclass
class RequestInfo:
    block_ids: List[int] = field(default_factory=list)
    sequence_length:int = field(default_factory=int)
    reserved_position: Optional[int] = None
    reserved_block_id: Optional[int] = None
    reserved_block_offset: Optional[int] = None
    written_layer_ids: Set[int] = field(default_factory=set)


@dataclass(frozen=True)
class DecodeMetadata:
    """Device metadata shared by every attention layer in one decode iteration."""

    request_ids: Tuple[Hashable, ...]
    block_table: torch.Tensor
    context_lengths: torch.Tensor
    maximum_context_length: int
    reserved_block_ids: torch.Tensor
    reserved_block_offsets: torch.Tensor
    token_offsets: torch.Tensor
    token_offsets_are_identity: bool

class KVCacheManager:
    def __init__(self, block_size, total_memory, tensor_dtype, device,
                 num_layers, num_kv_heads, head_dim, cache_dtype=None):
        self.block_size = block_size
        self.total_memory = total_memory
        self.tensor_dtype = tensor_dtype
        self.cache_dtype = cache_dtype or tensor_dtype
        if self.cache_dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.int8,
        ):
            raise ValueError("KV cache dtype must be FP16, BF16, FP32, or INT8")
        self.device = device

        # 2 × layers × KV heads × block size × head dimension × bytes per element
        bytes_per_element = torch.empty(0, dtype=self.cache_dtype).element_size()
        scale_bytes = 0
        if self.cache_dtype == torch.int8:
            # One FP32 symmetric scale per layer, physical block, head and token,
            # independently for K and V.
            scale_bytes = 2 * num_layers * num_kv_heads * block_size * 4
        self.bytes_per_block = (
            2 * num_layers * num_kv_heads * block_size * head_dim * bytes_per_element
            + scale_bytes
        )
        self.total_available_blocks = self.total_memory // self.bytes_per_block

        if self.total_available_blocks == 0:
            raise ValueError(
                f"Memory budget is too small; one block requires {self.bytes_per_block} bytes"
            )

        self.free_blocks = list(range(self.total_available_blocks))
        self.peak_allocated_blocks = 0
        self._decode_token_offset_cache = {}
        self._decode_metadata_buffers = {}

        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


        # "heads, tokens, head_dim" divided in blocks for each layer?
        pool_shape = (
            self.num_layers,
            self.total_available_blocks,
            self.num_kv_heads,
            self.block_size,
            self.head_dim
        )

        self.key_pool = torch.empty(pool_shape, dtype=self.cache_dtype, device=self.device)
        self.value_pool = torch.empty(pool_shape, dtype=self.cache_dtype, device=self.device)
        if self.cache_dtype == torch.int8:
            scale_shape = pool_shape[:-1]
            self.key_scale_pool = torch.empty(
                scale_shape, dtype=torch.float32, device=self.device
            )
            self.value_scale_pool = torch.empty(
                scale_shape, dtype=torch.float32, device=self.device
            )
        else:
            # A real device pointer keeps the Triton call signature identical;
            # the compile-time KV_QUANTIZED branch never loads these tensors.
            self.key_scale_pool = torch.empty(1, dtype=torch.float32, device=self.device)
            self.value_scale_pool = torch.empty(1, dtype=torch.float32, device=self.device)

        self.requests = defaultdict(RequestInfo)

    @property
    def is_quantized(self):
        return self.cache_dtype == torch.int8

    @staticmethod
    def _quantize_vectors(tensor):
        values = tensor.to(torch.float32)
        scale = values.abs().amax(dim=-1) / 127.0
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        quantized = torch.round(values / scale.unsqueeze(-1)).clamp(-127, 127)
        return quantized.to(torch.int8), scale

    def allocate_block(self):
        if not self.free_blocks:
            raise RuntimeError("KV cache full")
        block_id = self.free_blocks.pop()
        allocated_blocks = self.total_available_blocks - len(self.free_blocks)
        self.peak_allocated_blocks = max(
            self.peak_allocated_blocks,
            allocated_blocks,
        )
        return block_id

    def free_block(self, block_id):
        self.free_blocks.append(block_id)

    def reset_usage_stats(self):
        self.peak_allocated_blocks = self.total_available_blocks - len(
            self.free_blocks
        )

    def usage_summary(self):
        peak_bytes = self.peak_allocated_blocks * self.bytes_per_block
        return {
            "total_blocks": self.total_available_blocks,
            "free_blocks_at_end": len(self.free_blocks),
            "peak_allocated_blocks": self.peak_allocated_blocks,
            "peak_allocated_tokens": self.peak_allocated_blocks * self.block_size,
            "peak_allocated_bytes": peak_bytes,
            "peak_utilization_percent": (
                100 * self.peak_allocated_blocks / self.total_available_blocks
            ),
            "bytes_per_block": self.bytes_per_block,
            "bytes_per_token": self.bytes_per_block // self.block_size,
        }

    def reserve_token_slot(self, request_id):
        request_info = self.requests[request_id]
        if request_info.reserved_position is not None:
            raise RuntimeError("the previous token was not committed")

        if request_info.written_layer_ids:
            raise RuntimeError("Written layers is not empty")

        sequence_length = request_info.sequence_length
        logical_block = sequence_length // self.block_size
        block_offset = sequence_length % self.block_size

        if block_offset == 0:
            block_id = self.allocate_block()
            request_info.block_ids.append(block_id)

        else:
            block_id = request_info.block_ids[logical_block]

        request_info.reserved_position = sequence_length
        request_info.reserved_block_id = block_id
        request_info.reserved_block_offset = block_offset
        return request_info.reserved_position, request_info.reserved_block_id, request_info.reserved_block_offset


    def write_layer_kv(self, request_id, layer_id, new_key, new_value):
        assert new_key.shape == (self.num_kv_heads, self.head_dim)
        assert new_value.shape == (self.num_kv_heads, self.head_dim)
        assert 0 <= layer_id <= self.num_layers - 1

        request_info = self.requests[request_id]
        self._write_reserved_layer(request_info, layer_id, new_key, new_value)

    def _write_reserved_layer(self, request_info, layer_id, new_key, new_value):
        if request_info.reserved_position is None:
            raise RuntimeError("No token slot is reserved")

        if layer_id in request_info.written_layer_ids:
            raise RuntimeError("Duplicate layer write")

        reserved_block_id = request_info.reserved_block_id
        reserved_block_offset = request_info.reserved_block_offset

        if self.is_quantized:
            quantized_key, key_scale = self._quantize_vectors(new_key)
            quantized_value, value_scale = self._quantize_vectors(new_value)
            self.key_pool[layer_id, reserved_block_id, :, reserved_block_offset, :] = quantized_key
            self.value_pool[layer_id, reserved_block_id, :, reserved_block_offset, :] = quantized_value
            self.key_scale_pool[layer_id, reserved_block_id, :, reserved_block_offset] = key_scale
            self.value_scale_pool[layer_id, reserved_block_id, :, reserved_block_offset] = value_scale
        else:
            self.key_pool[layer_id, reserved_block_id, :, reserved_block_offset, :] = new_key
            self.value_pool[layer_id, reserved_block_id, :, reserved_block_offset, :] = new_value
        request_info.written_layer_ids.add(layer_id)


    def is_layer_written(self, request_id, layer_id):
        return layer_id in self.requests[request_id].written_layer_ids

    def commit_token(self, request_id):
        request_info = self.requests[request_id]
        if request_info.reserved_position is None:
            raise RuntimeError("No token slot is reserved.")

        if request_info.reserved_position != request_info.sequence_length:
            raise RuntimeError("Reserved position is not matching sequence length")

        for layer in range(self.num_layers):
            if layer not in request_info.written_layer_ids:
                raise RuntimeError("Layer not written")

        request_info.sequence_length += 1
        request_info.reserved_position = None
        request_info.reserved_block_id = None
        request_info.reserved_block_offset = None
        request_info.written_layer_ids.clear()

    def store_prefill_request(self, request_id, layer_kv_cache):
        self.append_prefill_chunk(request_id, layer_kv_cache, start_position=0)

    def append_prefill_chunk(self, request_id, layer_kv_cache, start_position):
        if start_position < 0:
            raise ValueError("Prefill start position cannot be negative")
        if request_id in self.requests:
            request_info = self.requests[request_id]
            if request_info.sequence_length != start_position:
                raise RuntimeError("Prefill chunk does not continue the cached sequence")
            if request_info.reserved_position is not None:
                raise RuntimeError("Cannot append a prefill chunk with a reserved token")
        elif start_position != 0:
            raise RuntimeError("The first prefill chunk must start at position zero")

        if len(layer_kv_cache) != self.num_layers:
            raise ValueError("Prefill cache does not contain every model layer")

        number_of_prompt_token = layer_kv_cache[0][0].shape[1]
        if number_of_prompt_token == 0:
            raise ValueError("A prefill cache cannot be empty")

        expected_shape = (
            self.num_kv_heads,
            number_of_prompt_token,
            self.head_dim,
        )
        for layer_key_cache, layer_value_cache in layer_kv_cache:
            if layer_key_cache.shape != expected_shape:
                raise ValueError("Unexpected prefill key-cache shape")
            if layer_value_cache.shape != expected_shape:
                raise ValueError("Unexpected prefill value-cache shape")

        request_info = self.requests[request_id]
        expected_existing_blocks = math.ceil(start_position / self.block_size)
        if len(request_info.block_ids) != expected_existing_blocks:
            raise RuntimeError("KV block table does not match the prefill start position")

        end_position = start_position + number_of_prompt_token
        required_blocks = math.ceil(end_position / self.block_size)
        missing_blocks = required_blocks - len(request_info.block_ids)
        allocated_blocks = []
        try:
            for _ in range(missing_blocks):
                allocated_blocks.append(self.allocate_block())
        except Exception:
            for block_id in allocated_blocks:
                self.free_block(block_id)
            raise
        request_info.block_ids.extend(allocated_blocks)

        # Stack every layer so each physical block is written with two batched
        # device copies instead of copying every token and layer independently.
        stacked_keys = torch.stack(
            [layer_key for layer_key, _ in layer_kv_cache],
            dim=0,
        )
        stacked_values = torch.stack(
            [layer_value for _, layer_value in layer_kv_cache],
            dim=0,
        )
        if self.is_quantized:
            stacked_keys, stacked_key_scales = self._quantize_vectors(stacked_keys)
            stacked_values, stacked_value_scales = self._quantize_vectors(stacked_values)

        first_logical_block = start_position // self.block_size
        last_logical_block = (end_position - 1) // self.block_size
        for logical_block in range(first_logical_block, last_logical_block + 1):
            block_start = logical_block * self.block_size
            segment_start = max(start_position, block_start)
            segment_end = min(end_position, block_start + self.block_size)
            source_start = segment_start - start_position
            source_end = segment_end - start_position
            block_offset = segment_start - block_start
            block_end = block_offset + (segment_end - segment_start)
            physical_block = request_info.block_ids[logical_block]

            self.key_pool[
                :, physical_block, :, block_offset:block_end, :
            ] = stacked_keys[:, :, source_start:source_end, :]
            self.value_pool[
                :, physical_block, :, block_offset:block_end, :
            ] = stacked_values[:, :, source_start:source_end, :]
            if self.is_quantized:
                self.key_scale_pool[
                    :, physical_block, :, block_offset:block_end
                ] = stacked_key_scales[:, :, source_start:source_end]
                self.value_scale_pool[
                    :, physical_block, :, block_offset:block_end
                ] = stacked_value_scales[:, :, source_start:source_end]

        request_info.sequence_length = end_position

    def store_prefill(self, request_id, kv_cache, batch_index=0):
        request_cache = []
        for layer_key_cache, layer_value_cache in kv_cache:
            if not 0 <= batch_index < layer_key_cache.shape[0]:
                raise IndexError("batch_index is outside the prefill cache batch")
            request_cache.append(
                (
                    layer_key_cache[batch_index],
                    layer_value_cache[batch_index],
                )
            )

        self.store_prefill_request(request_id, request_cache)


    def get_context_metadata(self, request_id):
        request_info = self.requests[request_id]
        if request_info.reserved_position is None:
            raise RuntimeError("No token slot is reserved")

        context_length = request_info.sequence_length + 1
        num_logical_block = math.ceil(context_length / self.block_size)

        physical_blocks = []
        valid_token_size = []
        for i in range(num_logical_block):
            block_start = i * self.block_size
            block = request_info.block_ids[i]
            physical_blocks.append(block)
            valid_token_size.append(min(self.block_size, context_length-block_start))

        return physical_blocks, valid_token_size, context_length


    def write_layer_kv_batch(
        self,
        request_ids,
        layer_id,
        new_keys,
        new_values,
        decode_metadata=None,
    ):
        if not request_ids:
            return
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("request_ids must be unique")
        if not 0 <= layer_id < self.num_layers:
            raise ValueError("Invalid layer_id")

        expected_shape = (len(request_ids), self.num_kv_heads, self.head_dim)
        if new_keys.shape != expected_shape or new_values.shape != expected_shape:
            raise ValueError("Unexpected batched K/V shape")

        request_infos = []
        for request_id in request_ids:
            request_info = self.requests[request_id]
            if request_info.reserved_position is None:
                raise RuntimeError(f"Request {request_id} has no reserved token slot")
            if layer_id in request_info.written_layer_ids:
                raise RuntimeError(f"Request {request_id} already wrote layer {layer_id}")
            request_infos.append(request_info)

        if decode_metadata is not None:
            if decode_metadata.request_ids != tuple(request_ids):
                raise ValueError("Decode metadata request order does not match")
            block_ids = decode_metadata.reserved_block_ids
            block_offsets = decode_metadata.reserved_block_offsets
        else:
            block_ids = torch.tensor(
                [request_info.reserved_block_id for request_info in request_infos],
                dtype=torch.long,
                device=self.key_pool.device,
            )
            block_offsets = torch.tensor(
                [request_info.reserved_block_offset for request_info in request_infos],
                dtype=torch.long,
                device=self.key_pool.device,
            )

        if self.is_quantized:
            new_keys, key_scales = self._quantize_vectors(new_keys)
            new_values, value_scales = self._quantize_vectors(new_values)
            self.key_scale_pool[layer_id, block_ids, :, block_offsets] = key_scales
            self.value_scale_pool[layer_id, block_ids, :, block_offsets] = value_scales
        self.key_pool[layer_id, block_ids, :, block_offsets, :] = new_keys # [batch_size, heads, head_dim]
        self.value_pool[layer_id, block_ids, :, block_offsets, :] = new_values # [batch_size, heads, head_dim]

        for request_id in request_ids:
            self.requests[request_id].written_layer_ids.add(layer_id)

    def validate_reserved_layer_write(self, request_ids, layer_id):
        if not 0 <= layer_id < self.num_layers:
            raise ValueError("Invalid layer_id")
        for request_id in request_ids:
            request_info = self.requests[request_id]
            if request_info.reserved_position is None:
                raise RuntimeError(f"Request {request_id} has no reserved token slot")
            if layer_id in request_info.written_layer_ids:
                raise RuntimeError(f"Request {request_id} already wrote layer {layer_id}")

    def mark_reserved_layer_written(self, request_ids, layer_id):
        """Record a successful device-side K/V write performed by Triton."""
        self.validate_reserved_layer_write(request_ids, layer_id)
        for request_id in request_ids:
            self.requests[request_id].written_layer_ids.add(layer_id)


    def free_request(self, request_id):
        if request_id not in self.requests:
            raise RuntimeError("Request id not found")

        request_info = self.requests[request_id]
        if request_info.reserved_position is not None:
            raise RuntimeError("Token slot is reserved")

        if len(request_info.written_layer_ids) > 0:
            raise RuntimeError("Written layers are not empty")

        if len(request_info.block_ids) != len(set(request_info.block_ids)):
            raise RuntimeError("Request contains duplicate block IDs")

        for block_id in request_info.block_ids:
            self.free_block(block_id)

        del self.requests[request_id]


    def gather_layer(self, request_id, layer_id):
        request_info = self.requests[request_id]
        block_ids = request_info.block_ids

        layer_key = self.key_pool[layer_id, block_ids, :, :, :]
        layer_value = self.value_pool[layer_id, block_ids, :, :, :]

        # Move kv_heads before the blocks:
        # before: block_id, kv head, block_size, embedding_dim
        # after: kv head, block_id, block_size, embedding_dim
        layer_key = layer_key.permute(1, 0, 2, 3)
        layer_value = layer_value.permute(1, 0, 2, 3)

        # combine block_id, block_size
        layer_key = layer_key.reshape(self.num_kv_heads, -1, self.head_dim)
        layer_value = layer_value.reshape(self.num_kv_heads, -1, self.head_dim)

        if self.is_quantized:
            key_scales = self.key_scale_pool[layer_id, block_ids].permute(1, 0, 2)
            value_scales = self.value_scale_pool[layer_id, block_ids].permute(1, 0, 2)
            key_scales = key_scales.reshape(self.num_kv_heads, -1, 1)
            value_scales = value_scales.reshape(self.num_kv_heads, -1, 1)
            layer_key = layer_key.to(self.tensor_dtype) * key_scales.to(self.tensor_dtype)
            layer_value = layer_value.to(self.tensor_dtype) * value_scales.to(self.tensor_dtype)

        layer_key = layer_key[:, :request_info.sequence_length, :]
        layer_value = layer_value[:, :request_info.sequence_length, :]

        layer_key = layer_key.unsqueeze(0)
        layer_value = layer_value.unsqueeze(0)

        return layer_key, layer_value

    def validate_decode_layer(self, request_ids, layer_id):
        if not 0 <= layer_id < self.num_layers:
            raise ValueError("Invalid layer_id")
        for request_id in request_ids:
            if layer_id not in self.requests[request_id].written_layer_ids:
                raise RuntimeError(
                    f"Request {request_id} has not written layer {layer_id}"
                )

    def build_decode_metadata(self, request_ids, layer_id=None, token_offsets=None):
        if not request_ids:
            raise ValueError("request_ids cannot be empty")
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("request_ids must be unique")

        request_infos = []
        for request_id in request_ids:
            if request_id not in self.requests:
                raise RuntimeError(f"Unknown request: {request_id}")
            request_infos.append(self.requests[request_id])
        for request_id, request_info in zip(request_ids, request_infos):
            if request_info.reserved_position is None:
                raise RuntimeError(f"Request {request_id} has no reserved decode token")
            if not request_info.block_ids:
                raise RuntimeError(f"Request {request_id} has no KV blocks")

        if layer_id is not None:
            self.validate_decode_layer(request_ids, layer_id)

        block_ids = [request_info.block_ids for request_info in request_infos]
        context_lengths = [
            request_info.sequence_length + 1 for request_info in request_infos
        ]
        maximum_context_length = max(context_lengths)
        if token_offsets is None:
            token_offsets = tuple(range(len(request_ids)))
        else:
            token_offsets = tuple(token_offsets)
        if len(token_offsets) != len(request_ids):
            raise ValueError("Decode token offsets must match request_ids")
        if any(offset < 0 for offset in token_offsets):
            raise ValueError("Decode token offsets cannot be negative")
        token_offsets_are_identity = token_offsets == tuple(range(len(request_ids)))
        max_blocks = max(len(ids) for ids in block_ids)

        block_ids = [
            ids + [ids[-1]] * (max_blocks - len(ids)) for ids in block_ids
        ]

        packed_rows = [
            ids
            + [
                context_length,
                request_info.reserved_block_id,
                request_info.reserved_block_offset,
            ]
            for ids, context_length, request_info in zip(
                block_ids,
                context_lengths,
                request_infos,
            )
        ]
        packed_host = torch.tensor(packed_rows, dtype=torch.int32)
        if torch.is_inference_mode_enabled():
            packed_key = (self.key_pool.device, tuple(packed_host.shape))
            packed_metadata = self._decode_metadata_buffers.get(packed_key)
            if packed_metadata is None:
                packed_metadata = torch.empty(
                    packed_host.shape,
                    dtype=torch.int32,
                    device=self.key_pool.device,
                )
                self._decode_metadata_buffers[packed_key] = packed_metadata
            packed_metadata.copy_(packed_host, non_blocking=True)
        else:
            packed_metadata = packed_host.to(self.key_pool.device)
        block_table = packed_metadata[:, :max_blocks]
        context_lengths = packed_metadata[:, max_blocks]
        reserved_block_ids = packed_metadata[:, max_blocks + 1]
        reserved_block_offsets = packed_metadata[:, max_blocks + 2]
        offset_cache_key = (self.key_pool.device, token_offsets)
        token_offsets_tensor = self._decode_token_offset_cache.get(offset_cache_key)
        if token_offsets_tensor is None or not torch.is_inference_mode_enabled():
            token_offsets_tensor = torch.tensor(
                token_offsets,
                dtype=torch.long,
                device=self.key_pool.device,
            )
            if torch.is_inference_mode_enabled():
                if len(self._decode_token_offset_cache) >= 128:
                    self._decode_token_offset_cache.pop(
                        next(iter(self._decode_token_offset_cache))
                    )
                self._decode_token_offset_cache[offset_cache_key] = token_offsets_tensor

        return DecodeMetadata(
            request_ids=tuple(request_ids),
            block_table=block_table,
            context_lengths=context_lengths,
            maximum_context_length=maximum_context_length,
            reserved_block_ids=reserved_block_ids,
            reserved_block_offsets=reserved_block_offsets,
            token_offsets=token_offsets_tensor,
            token_offsets_are_identity=token_offsets_are_identity,
        )

    def gather_decode_layer_batch(self, request_ids, layer_id, metadata=None):
        """Gather variable-length decode contexts with one batched pool lookup."""
        if not request_ids:
            raise ValueError("request_ids cannot be empty")
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("request_ids must be unique")
        if not 0 <= layer_id < self.num_layers:
            raise ValueError("Invalid layer_id")

        request_infos = [self.requests[request_id] for request_id in request_ids]
        for request_id, request_info in zip(request_ids, request_infos):
            if request_info.reserved_position is None:
                raise RuntimeError(f"Request {request_id} has no reserved decode token")
            if layer_id not in request_info.written_layer_ids:
                raise RuntimeError(f"Request {request_id} has not written layer {layer_id}")
            if not request_info.block_ids:
                raise RuntimeError(f"Request {request_id} has no KV blocks")

        if metadata is None:
            metadata = self.build_decode_metadata(request_ids, layer_id)
        elif metadata.request_ids != tuple(request_ids):
            raise ValueError("Decode metadata request order does not match")
        block_table = metadata.block_table.to(torch.long)

        # [batch, blocks, heads, block, dim] -> [batch, heads, tokens, dim]
        keys = self.key_pool[layer_id, block_table].permute(0, 2, 1, 3, 4)
        values = self.value_pool[layer_id, block_table].permute(0, 2, 1, 3, 4)
        keys = keys.reshape(len(request_ids), self.num_kv_heads, -1, self.head_dim)
        values = values.reshape(len(request_ids), self.num_kv_heads, -1, self.head_dim)

        if self.is_quantized:
            key_scales = self.key_scale_pool[layer_id, block_table].permute(0, 2, 1, 3)
            value_scales = self.value_scale_pool[layer_id, block_table].permute(0, 2, 1, 3)
            key_scales = key_scales.reshape(len(request_ids), self.num_kv_heads, -1, 1)
            value_scales = value_scales.reshape(len(request_ids), self.num_kv_heads, -1, 1)
            keys = keys.to(self.tensor_dtype) * key_scales.to(self.tensor_dtype)
            values = values.to(self.tensor_dtype) * value_scales.to(self.tensor_dtype)

        maximum_context = metadata.maximum_context_length
        keys = keys[:, :, :maximum_context, :]
        values = values[:, :, :maximum_context, :]
        length_tensor = metadata.context_lengths.to(torch.long)
        valid_positions = torch.arange(
            maximum_context,
            device=self.key_pool.device,
        ).unsqueeze(0) < length_tensor.unsqueeze(1)

        # Padded blocks may point at arbitrary cache data; zero them before the
        # value matmul so masked probabilities cannot propagate uninitialized NaNs.
        invalid_positions = ~valid_positions[:, None, :, None]
        keys = keys.masked_fill(invalid_positions, 0)
        values = values.masked_fill(invalid_positions, 0)
        return keys, values, valid_positions
