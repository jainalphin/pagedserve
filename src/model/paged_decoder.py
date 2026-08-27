import math
from typing import List

import torch
from dataclasses import dataclass, field

from torch import nn
from torch.nn import functional as F

from src.model.iteration import IterationBatch
from src.model.kv_manager import KVCacheManager
from src.model.paged_attention import PagedAttention


@dataclass
class TransformerConfig:
    vocab_size: int = field(default=1000)
    hidden_size: int = field(default=64)
    num_layers: int = field(default=2)
    num_heads: int = field(default=4)
    head_dim: int = field(default=16)
    mlp_hidden_size: int = field(default=256)
    max_sequence_length: int = field(default=128)
    activation_function: str = field(default="gelu")
    layer_norm_epsilon: float = field(default=1e-5)
    tie_word_embeddings: bool = field(default=False)
    lm_head_bias: bool = field(default=True)

class PagedSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        # GPT-2 stores Q/K/V as one Conv1D projection. Keeping the same packed
        # layout performs one GEMM and one weight read instead of three.
        self.qkv_linear = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.output_linear = nn.Linear(config.hidden_size, config.hidden_size)

    def project_qkv(self, hidden_states):
        return self.qkv_linear(hidden_states).split(self.config.hidden_size, dim=-1)

    def split_heads(self, input_data):
        batch_size, sequence_length, hidden_size = input_data.shape
        return input_data.reshape(batch_size, sequence_length, self.config.num_heads, self.config.head_dim).transpose(1, 2)

    def merge_heads(self, input_data):
        batch_size, num_heads, sequence_length, head_dim = input_data.shape
        return input_data.transpose(1,2).contiguous().reshape(batch_size, sequence_length, -1)

    def split_heads_flat(self, input_data):
        token_count, hidden_size = input_data.shape
        return input_data.reshape(
            token_count,
            self.config.num_heads,
            self.config.head_dim,
        )

    def merge_heads_flat(self, input_data):
        token_count, num_heads, head_dim = input_data.shape
        return input_data.reshape(token_count, self.config.hidden_size)

    def prefill(self, hidden_states):
        # hidden_state:  [batch, sequence_length, hidden_size]
        q, k, v = self.project_qkv(hidden_states)

        q = self.split_heads(q) # [B, heads, T, head_dim]
        k = self.split_heads(k)
        v = self.split_heads(v)

        scale = 1 / math.sqrt(self.config.head_dim)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale # [B, heads, T, T] == [B, heads, query_position, key_position]

        bool_mask = torch.ones_like(scores, dtype=torch.bool, device=scores.device)
        bool_mask = torch.triu(bool_mask, diagonal=1)
        scores = scores.masked_fill(bool_mask, float('-inf'))

        attn_probs = F.softmax(scores, dim=-1) # [B, heads, T, T]
        context = torch.matmul(attn_probs, v) # [B, heads, T, T] × [B, heads, T, head_dim]

        # [B, heads, T, head_dim] → [B, T, hidden_size] → [B, T, hidden_size]
        attn_output = self.output_linear(self.merge_heads(context))

        return attn_output, k, v

    def decode(self, hidden_states, request_ids, layer_id, kv_manger: KVCacheManager, paged_attn_manager: PagedAttention, decode_metadata):
        batch_size = len(request_ids)
        assert hidden_states.shape == (batch_size, 1, self.config.hidden_size)

        q, k, v = self.project_qkv(hidden_states)

        q = self.split_heads(q).squeeze(2) # [B, heads, 1, head_dim]
        k = self.split_heads(k).squeeze(2)
        v = self.split_heads(v).squeeze(2)

        kv_manger.write_layer_kv_batch(
            request_ids,
            layer_id,
            k,
            v,
            decode_metadata=decode_metadata,
        )
        output = paged_attn_manager.forward_batch(
            request_ids, layer_id, q, decode_metadata=decode_metadata
        )
        output = output.unsqueeze(2) # [B, heads, 1, head_dim]
        output = self.merge_heads(output)
        output = self.output_linear(output)
        return output

    def forward_iteration(self, hidden_states, items, layer_id, paged_attn_manager: PagedAttention, decode_metadata=None):
        queries, keys, values = self.project_qkv(hidden_states)
        queries = self.split_heads_flat(queries)
        keys = self.split_heads_flat(keys)
        values = self.split_heads_flat(values)

        context, prefill_kv = paged_attn_manager.forward_iteration(
            items=items,
            layer_id=layer_id,
            queries=queries,
            keys=keys,
            values=values,
            decode_metadata=decode_metadata,
        )
        output = self.output_linear(self.merge_heads_flat(context))
        return output, prefill_kv



class DecoderBlock(nn.Module):
    def __init__(self, config: TransformerConfig, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.self_attn = PagedSelfAttention(self.config)
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.mlp_up_proj = nn.Linear(config.hidden_size, config.mlp_hidden_size)
        self.mlp_down_proj = nn.Linear(config.mlp_hidden_size, config.hidden_size)

    def prefill(self, hidden_states):
        attention_residual = hidden_states
        attn_output, key_states, value_states = self.self_attn.prefill(self.input_layernorm(hidden_states))
        hidden_states = attn_output + attention_residual
        mlp_residual = hidden_states
        hidden_states = self.mlp_up_proj(self.post_attention_layernorm(hidden_states))
        hidden_states = self._activate(hidden_states)
        hidden_states = self.mlp_down_proj(hidden_states) + mlp_residual
        return hidden_states, key_states, value_states

    def decode(self, hidden_states, request_ids, kv_manger: KVCacheManager, paged_attn_manager: PagedAttention, decode_metadata):
        attention_residual = hidden_states
        hidden_states = self.self_attn.decode(self.input_layernorm(hidden_states),
                                                                  request_ids,
                                                                  self.layer_id,
                                                                  kv_manger,
                                                                  paged_attn_manager,
                                                                  decode_metadata)

        hidden_states = hidden_states + attention_residual
        mlp_residual = hidden_states
        hidden_states = self.mlp_up_proj(self.post_attention_layernorm(hidden_states))
        hidden_states = self._activate(hidden_states)
        hidden_states = self.mlp_down_proj(hidden_states) + mlp_residual
        return hidden_states

    def forward_iteration(self, hidden_states, items, paged_attn_manager: PagedAttention, decode_metadata=None):
        attention_residual = hidden_states
        attention_output, prefill_kv = self.self_attn.forward_iteration(
            self.input_layernorm(hidden_states),
            items,
            self.layer_id,
            paged_attn_manager,
            decode_metadata,
        )
        hidden_states = attention_output + attention_residual

        mlp_residual = hidden_states
        hidden_states = self.mlp_up_proj(self.post_attention_layernorm(hidden_states))
        hidden_states = self._activate(hidden_states)
        hidden_states = self.mlp_down_proj(hidden_states) + mlp_residual
        return hidden_states, prefill_kv

    def _activate(self, hidden_states):
        if self.config.activation_function == "gelu":
            return F.gelu(hidden_states)
        if self.config.activation_function == "gelu_tanh":
            return F.gelu(hidden_states, approximate="tanh")
        raise ValueError(
            f"Unsupported activation function: {self.config.activation_function}"
        )



class PagedDecoderLM(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embedding_table = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_sequence_length, config.hidden_size)
        self.layers = nn.ModuleList([DecoderBlock(config, layer_id=layer_id) for layer_id in range(config.num_layers)])
        self.final_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.output_layer = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=config.lm_head_bias,
        )
        if config.tie_word_embeddings:
            self.output_layer.weight = self.embedding_table.weight

    def embedding_helper(self, input_ids, position_ids):
        assert input_ids.size() == position_ids.size()
        if position_ids.dtype != torch.long:
            raise ValueError("Position IDs must use torch.long")

        token_embeddings = self.embedding_table(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        hidden_states =  token_embeddings + position_embeddings
        return hidden_states

    def prefill(self, input_ids):
        batch_size, prompt_length = input_ids.shape
        if prompt_length > self.config.max_sequence_length:
            raise ValueError("Prompt exceeds maximum sequence length")

        position_ids = torch.arange(prompt_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, prompt_length) # [batch_size, prompt_length]
        hidden_states = self.embedding_helper(input_ids, position_ids)

        layer_kv_cache = []
        for layer in self.layers:
            hidden_states, key_states, value_states = layer.prefill(hidden_states)
            layer_kv_cache.append((key_states, value_states))

        logits = self.output_layer(self.final_layernorm(hidden_states)) # [B,T,vocab_size]
        return logits, layer_kv_cache


    def decode_batch(self, input_ids, request_ids, kv_manager: KVCacheManager, paged_attn_manager: PagedAttention):
        position_ids = []
        for request_id in request_ids:
            reserved_position, _, _ = kv_manager.reserve_token_slot(request_id)
            position_ids.append(reserved_position)

        if any(
            position_id >= self.config.max_sequence_length
            for position_id in position_ids
        ):
            raise ValueError("Position ID exceeds maximum sequence length")

        position_ids = torch.tensor(position_ids, dtype=torch.long, device=input_ids.device).unsqueeze(1)
        hidden_states = self.embedding_helper(input_ids, position_ids)

        # Block tables and lengths depend on the iteration, not the layer. Build
        # them once and retain the same device tensors throughout the layer loop.
        decode_metadata = kv_manager.build_decode_metadata(request_ids)

        for layer in self.layers:
            hidden_states = layer.decode(
                hidden_states,
                request_ids,
                kv_manager,
                paged_attn_manager,
                decode_metadata,
            )

        logits = self.output_layer(self.final_layernorm(hidden_states))  # [B,T,vocab_size]

        for request_id in request_ids:
            kv_manager.commit_token(request_id)

        return logits

    def forward_iteration(self, iteration_batch: IterationBatch, kv_manager: KVCacheManager, paged_attn_manager: PagedAttention):
        if iteration_batch.input_ids.device != next(self.parameters()).device:
            raise ValueError("Iteration inputs and model must be on the same device")
        if paged_attn_manager.kv_manager is not kv_manager:
            raise ValueError("PagedAttention and the model must use the same KV manager")

        # Creates an empty cache collector for every prefill request
        prefill_cache = {item.request_id: [] for item in iteration_batch.items if item.phase == "prefill"}

        for item in iteration_batch.items:
            if item.position_ids[0] < 0:
                raise ValueError("Position IDs cannot be negative")
            if item.position_ids[-1] >= self.config.max_sequence_length:
                raise ValueError("Position ID exceeds maximum sequence length")
            if item.phase == "prefill":
                prefill_start = item.position_ids[0]
                expected_positions = tuple(
                    range(prefill_start, prefill_start + item.token_count)
                )
                if item.position_ids != expected_positions:
                    raise ValueError("Prefill positions must be contiguous")
                if item.request_id in kv_manager.requests:
                    request_info = kv_manager.requests[item.request_id]
                    if request_info.sequence_length != prefill_start:
                        raise RuntimeError("Prefill chunk does not match the KV-cache length")
                elif prefill_start != 0:
                    raise RuntimeError("The first prefill chunk must start at zero")
            else:
                if item.request_id not in kv_manager.requests:
                    raise RuntimeError("Cannot decode an unknown request")
                reserved_position, _, _ = kv_manager.reserve_token_slot(item.request_id)
                if item.position_ids != (reserved_position,):
                    raise ValueError("Decode position does not match the KV-cache length")

        hidden_states = self.embedding_helper(iteration_batch.input_ids, iteration_batch.position_ids)

        decode_request_ids = [
            item.request_id for item in iteration_batch.items if item.phase == "decode"
        ]
        decode_token_offsets = [
            item.start_offset for item in iteration_batch.items if item.phase == "decode"
        ]
        decode_metadata = (
            kv_manager.build_decode_metadata(
                decode_request_ids,
                token_offsets=decode_token_offsets,
            )
            if decode_request_ids
            else None
        )

        for layer in self.layers:
            # Entire flattened tensor then passes through every decoder layer
            hidden_states, layer_prefill_kv = layer.forward_iteration(
                hidden_states,
                iteration_batch.items,
                paged_attn_manager,
                decode_metadata,
            )
            for request_id, key_value in layer_prefill_kv.items():
                prefill_cache[request_id].append(key_value)

        output_items = [
            item for item in iteration_batch.items if item.produces_output
        ]
        if output_items:
            output_offsets = [item.end_offset - 1 for item in output_items]
            if output_offsets == list(range(hidden_states.shape[0])):
                output_hidden_states = hidden_states
            else:
                output_offsets = paged_attn_manager.inference_index_tensor(
                    "model_output_offsets",
                    output_offsets,
                    hidden_states.device,
                )
                output_hidden_states = hidden_states.index_select(0, output_offsets)
            logits = self.output_layer(self.final_layernorm(output_hidden_states))
        else:
            logits = hidden_states.new_empty((0, self.config.vocab_size))

        for item in iteration_batch.items:
            if item.phase == "prefill":
                kv_manager.append_prefill_chunk(
                    item.request_id,
                    prefill_cache[item.request_id],
                    start_position=item.position_ids[0],
                )
            else:
                kv_manager.commit_token(item.request_id)

        return logits



