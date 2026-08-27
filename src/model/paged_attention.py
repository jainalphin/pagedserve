from typing import Dict, Sequence, Tuple

import torch
from torch.nn import functional as F
from src.model.iteration import IterationItem
from src.model.kv_manager import DecodeMetadata, KVCacheManager


class PagedAttention:
    def __init__(
        self,
        kv_manager: KVCacheManager,
        decode_attention_backend: str = "torch",
    ):
        if decode_attention_backend not in ("torch", "triton"):
            raise ValueError(
                f"Unknown decode attention backend: {decode_attention_backend!r}. "
                "Expected 'torch' or 'triton'."
            )

        self.kv_manager = kv_manager
        self.decode_attention_backend = decode_attention_backend
        self.num_layers = kv_manager.num_layers
        self.num_kv_heads = kv_manager.num_kv_heads
        self.head_dim = kv_manager.head_dim
        self.scale = self.head_dim ** -0.5
        self._inference_buffers = {}
        self._inference_indices = {}

    def _inference_buffer(self, name, reference, shape=None):
        """Reuse hot-path storage only under the engine's inference-mode guard."""
        shape = tuple(reference.shape if shape is None else shape)
        if not torch.is_inference_mode_enabled():
            return torch.empty(shape, dtype=reference.dtype, device=reference.device)
        key = (name, reference.device, reference.dtype, shape)
        buffer = self._inference_buffers.get(key)
        if buffer is None:
            buffer = torch.empty(shape, dtype=reference.dtype, device=reference.device)
            self._inference_buffers[key] = buffer
        return buffer

    def inference_index_tensor(self, name, offsets, device):
        """Reuse immutable scheduler offset patterns during inference."""
        offsets = tuple(offsets)
        if not torch.is_inference_mode_enabled():
            return torch.tensor(offsets, dtype=torch.long, device=device)
        key = (name, device, offsets)
        indices = self._inference_indices.get(key)
        if indices is None:
            if len(self._inference_indices) >= 128:
                self._inference_indices.pop(next(iter(self._inference_indices)))
            indices = torch.tensor(offsets, dtype=torch.long, device=device)
            self._inference_indices[key] = indices
        return indices

    def _index_decode_tokens(self, name, tensor, metadata):
        shape = (metadata.token_offsets.numel(), *tensor.shape[1:])
        output = self._inference_buffer(name, tensor, shape)
        torch.index_select(tensor, 0, metadata.token_offsets, out=output)
        return output

    def attention_score(self, request_id, layer_id, query):
        assert query.shape == (self.num_kv_heads, self.head_dim)
        assert 0 <= layer_id < self.num_layers

        # query:     [heads, head_dim]
        # After:     [heads, 1, head_dim]
        query = query.unsqueeze(1)

        request_info = self.kv_manager.requests[request_id]

        if layer_id not in request_info.written_layer_ids:
            raise RuntimeError(f"Layer {layer_id} has not written the reserved token's KV")

        physical_blocks, valid_token_size, context_length = self.kv_manager.get_context_metadata(request_id)

        attention_scores = []
        for physical_block_id, valid_tokens in zip(physical_blocks, valid_token_size):
            key_block = self.kv_manager.key_pool[layer_id, physical_block_id, :, :valid_tokens, :]
            if self.kv_manager.is_quantized:
                key_scale = self.kv_manager.key_scale_pool[
                    layer_id, physical_block_id, :, :valid_tokens
                ]
                key_block = (
                    key_block.to(query.dtype) * key_scale.to(query.dtype).unsqueeze(-1)
                )
            # before: [heads, valid_tokens, head_dim]
            # after:  [heads, head_dim, valid_tokens]
            key_block = key_block.transpose(-2, -1)
            attention_score = torch.matmul(query, key_block)  # [heads, 1, valid_tokens]
            attention_score = attention_score.squeeze(1) * self.scale # [heads, valid_tokens]
            attention_scores.append(attention_score)

        attention_scores = torch.cat(attention_scores, dim=-1) # [num_heads, context_length]
        return attention_scores


    def compute_weighted_value_sum(self, request_id, layer_id, attention_scores):
        # Scalar Torch reference path; Triton decode uses fused online softmax.
        softmax_probabilities = torch.softmax(attention_scores, dim=-1)
        output = torch.zeros(
            self.num_kv_heads,
            self.head_dim,
            dtype=attention_scores.dtype,
            device=attention_scores.device,
        )

        physical_blocks, valid_token_size, context_length = self.kv_manager.get_context_metadata(request_id)

        start = 0
        for physical_block_id, valid_tokens in zip(physical_blocks, valid_token_size):

            value_block = self.kv_manager.value_pool[layer_id, physical_block_id, :, :valid_tokens, :] # [heads, valid_tokens, head_dim]
            if self.kv_manager.is_quantized:
                value_scale = self.kv_manager.value_scale_pool[
                    layer_id, physical_block_id, :, :valid_tokens
                ]
                value_block = (
                    value_block.to(attention_scores.dtype)
                    * value_scale.to(attention_scores.dtype).unsqueeze(-1)
                )
            block_prob = softmax_probabilities[:, start:start + valid_tokens] # [heads, valid_tokens]
            block_prob = block_prob.unsqueeze(1) # [heads, 1, valid_tokens]
            weighted_sum = torch.matmul(block_prob, value_block).squeeze(dim=1)
            output += weighted_sum
            start += valid_tokens

        return output

    def forward(self, request_id, layer_id, query):
        attention_score = self.attention_score(request_id, layer_id, query)
        return self.compute_weighted_value_sum(request_id, layer_id, attention_score)


    def forward_batch(
        self,
        request_ids,
        layer_id,
        queries,
        decode_metadata: DecodeMetadata = None,
        output: torch.Tensor = None,
        new_keys: torch.Tensor = None,
        new_values: torch.Tensor = None,
        _trusted_decode_metadata: bool = False,
    ):
        if not request_ids:
            return None

        if len(request_ids) != len(queries):
            raise RuntimeError("Number of request_ids and queries do not match")

        batch_size = len(request_ids)
        assert queries.shape == (batch_size, self.num_kv_heads, self.head_dim)

        if decode_metadata is None:
            decode_metadata = self.kv_manager.build_decode_metadata(request_ids)
        elif (
            not _trusted_decode_metadata
            and decode_metadata.request_ids != tuple(request_ids)
        ):
            raise ValueError("Decode metadata request order does not match")
        write_kv = new_keys is not None or new_values is not None
        if write_kv:
            if new_keys is None or new_values is None:
                raise ValueError("New keys and values must be provided together")
            expected_kv_shape = (batch_size, self.num_kv_heads, self.head_dim)
            if new_keys.shape != expected_kv_shape or new_values.shape != expected_kv_shape:
                raise ValueError("New K/V tensors must match the decode batch")
            if not _trusted_decode_metadata:
                self.kv_manager.validate_reserved_layer_write(request_ids, layer_id)
        else:
            if not _trusted_decode_metadata:
                self.kv_manager.validate_decode_layer(request_ids, layer_id)

        if self.decode_attention_backend == "triton":
            # Keep Triton optional: importing it is only necessary when this
            # backend is selected on a CUDA system.
            from src.kernels.triton_paged_attention import (
                paged_decode_attention_triton,
            )

            computed_output = paged_decode_attention_triton(
                queries,
                self.kv_manager.key_pool,
                self.kv_manager.value_pool,
                self.kv_manager.key_scale_pool,
                self.kv_manager.value_scale_pool,
                decode_metadata.block_table,
                decode_metadata.context_lengths,
                layer_id,
                new_keys=new_keys,
                new_values=new_values,
                reserved_block_ids=(
                    decode_metadata.reserved_block_ids if write_kv else None
                ),
                reserved_block_offsets=(
                    decode_metadata.reserved_block_offsets if write_kv else None
                ),
                output=output,
                maximum_context_length=decode_metadata.maximum_context_length,
                validate_inputs=False,
            )
            if write_kv:
                self.kv_manager.mark_reserved_layer_written(request_ids, layer_id)
            return computed_output

        if write_kv:
            self.kv_manager.write_layer_kv_batch(
                request_ids,
                layer_id,
                new_keys,
                new_values,
                decode_metadata=decode_metadata,
            )

        keys, values, valid_positions = self.kv_manager.gather_decode_layer_batch(
            request_ids,
            layer_id,
            decode_metadata,
        )
        attention_mask = valid_positions[:, None, None, :]
        computed_output = F.scaled_dot_product_attention(
            queries.unsqueeze(2),
            keys,
            values,
            attn_mask=attention_mask,
            dropout_p=0.0,
        )
        computed_output = computed_output.squeeze(2)
        if output is not None:
            output.copy_(computed_output)
            return output
        return computed_output

    def causal_prefill_batch(self, queries, keys, values):
        """Run equal-length fresh prompts through one batched SDPA call."""
        if queries.ndim != 4:
            raise ValueError("Batched prefill queries must be [batch, tokens, heads, dim]")
        if keys.shape != queries.shape or values.shape != queries.shape:
            raise ValueError("Batched prefill Q/K/V tensors must have matching shapes")

        output = F.scaled_dot_product_attention(
            queries.transpose(1, 2),
            keys.transpose(1, 2),
            values.transpose(1, 2),
            dropout_p=0.0,
            is_causal=True,
        )
        return output.transpose(1, 2)

    def causal_prefill(
        self,
        queries,
        keys,
        values,
        past_keys=None,
        past_values=None,
    ):
        token_count = queries.shape[0]
        queries = queries.transpose(0, 1)
        keys = keys.transpose(0, 1)
        values = values.transpose(0, 1)

        past_length = 0
        if past_keys is not None or past_values is not None:
            if past_keys is None or past_values is None:
                raise ValueError("Past prefill keys and values must be provided together")
            past_length = past_keys.shape[1]
            keys = torch.cat((past_keys, keys), dim=1)
            values = torch.cat((past_values, values), dim=1)

        scores = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale
        query_positions = past_length + torch.arange(
            token_count,
            device=scores.device,
        )
        key_positions = torch.arange(
            past_length + token_count,
            device=scores.device,
        )
        causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        return torch.matmul(probabilities, values).transpose(0, 1)

    def forward_iteration(
        self,
        items: Sequence[IterationItem],
        layer_id: int,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        decode_metadata: DecodeMetadata = None,
    ) -> Tuple[torch.Tensor, Dict[object, Tuple[torch.Tensor, torch.Tensor]]]:
        expected_shape = (queries.shape[0], self.num_kv_heads, self.head_dim)
        if queries.shape != expected_shape:
            raise ValueError("Unexpected flattened query shape")
        if keys.shape != expected_shape or values.shape != expected_shape:
            raise ValueError("Flattened Q/K/V tensors must have matching shapes")

        outputs = self._inference_buffer("iteration_outputs", queries)
        prefill_kv = {}

        decode_items = [item for item in items if item.phase == "decode"]
        if decode_items:
            decode_request_ids = [item.request_id for item in decode_items]
            if decode_metadata is None:
                raise RuntimeError("Decode items require iteration metadata")
            decode_only = (
                decode_metadata.token_offsets_are_identity
                and decode_metadata.token_offsets.numel() == queries.shape[0]
            )
            if decode_only:
                decode_queries = queries
                decode_keys = keys
                decode_values = values
                decode_output_buffer = outputs
            else:
                decode_queries = self._index_decode_tokens(
                    "decode_queries", queries, decode_metadata
                )
                decode_keys = self._index_decode_tokens(
                    "decode_keys", keys, decode_metadata
                )
                decode_values = self._index_decode_tokens(
                    "decode_values", values, decode_metadata
                )
                decode_output_buffer = self._inference_buffer(
                    "decode_outputs", decode_queries
                )
            decode_outputs = self.forward_batch(
                decode_request_ids,
                layer_id,
                decode_queries,
                decode_metadata=decode_metadata,
                output=decode_output_buffer,
                new_keys=decode_keys,
                new_values=decode_values,
            )
            if not decode_only:
                outputs.index_copy_(
                    0, decode_metadata.token_offsets, decode_outputs
                )

        fresh_prefill_groups = {}
        for item in items:
            if item.phase == "prefill" and item.position_ids[0] == 0:
                fresh_prefill_groups.setdefault(item.token_count, []).append(item)

        for token_count, group in fresh_prefill_groups.items():
            group_queries = torch.stack(
                [queries[item.start_offset:item.end_offset] for item in group]
            )
            group_keys = torch.stack(
                [keys[item.start_offset:item.end_offset] for item in group]
            )
            group_values = torch.stack(
                [values[item.start_offset:item.end_offset] for item in group]
            )
            group_outputs = self.causal_prefill_batch(
                group_queries,
                group_keys,
                group_values,
            )
            group_offsets = self.inference_index_tensor(
                "fresh_prefill_group_offsets",
                (
                    offset
                    for item in group
                    for offset in range(item.start_offset, item.end_offset)
                ),
                outputs.device,
            )
            outputs[group_offsets] = group_outputs.reshape(
                len(group) * token_count,
                self.num_kv_heads,
                self.head_dim,
            )
            for item, item_keys, item_values in zip(group, group_keys, group_values):
                prefill_kv[item.request_id] = (
                    item_keys.transpose(0, 1),
                    item_values.transpose(0, 1),
                )

        for item in items:
            if item.phase == "decode":
                continue
            if item.position_ids[0] == 0:
                continue
            item_slice = slice(item.start_offset, item.end_offset)
            item_queries = queries[item_slice]

            item_keys = keys[item_slice]
            item_values = values[item_slice]
            past_keys, past_values = self.kv_manager.gather_layer(
                item.request_id,
                layer_id,
            )
            outputs[item_slice] = self.causal_prefill(
                item_queries,
                item_keys,
                item_values,
                past_keys.squeeze(0),
                past_values.squeeze(0),
            )
            prefill_kv[item.request_id] = (
                item_keys.transpose(0, 1),
                item_values.transpose(0, 1),
            )

        return outputs, prefill_kv
