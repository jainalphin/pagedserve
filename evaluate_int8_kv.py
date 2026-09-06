"""Evaluate INT8 KV capacity, Triton decode latency, error, and proxy perplexity."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from src.model.gpt2 import load_gpt2_pretrained
from src.model.kv_manager import KVCacheManager
from src.model.paged_attention import PagedAttention


EVALUATION_TEXT = (
    "Paged attention stores key and value vectors in fixed-size physical pages. "
    "A logical block table lets each request grow without a contiguous allocation. "
    "The decode kernel reads those pages directly and maintains an online softmax. "
)


def latency_us(function, warmup, iterations):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def make_manager(model, dtype, budget):
    return KVCacheManager(
        block_size=16,
        total_memory=budget,
        tensor_dtype=next(model.parameters()).dtype,
        cache_dtype=dtype,
        device="cuda",
        num_layers=model.config.num_layers,
        num_kv_heads=model.config.num_heads,
        head_dim=model.config.head_dim,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="openai-community/gpt2")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--prefix-tokens", type=int, default=32)
    parser.add_argument("--kv-budget-mib", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("INT8 in-kernel dequantization requires CUDA and Triton")
    if args.iterations < 100:
        parser.error("--iterations must be at least 100")
    model, tokenizer = load_gpt2_pretrained(args.model_id)
    model = model.to(device="cuda", dtype=torch.float16).eval()
    token_ids = tokenizer.encode(EVALUATION_TEXT * 16)[: args.tokens]
    if len(token_ids) < args.tokens or not 1 < args.prefix_tokens < len(token_ids) - 1:
        parser.error("the requested token/prefix lengths are invalid")
    tokens = torch.tensor(token_ids, dtype=torch.long, device="cuda")
    budget = args.kv_budget_mib * 2**20
    managers = {
        "fp16": make_manager(model, torch.float16, budget),
        "int8": make_manager(model, torch.int8, budget),
    }
    prefix = tokens[: args.prefix_tokens].unsqueeze(0)
    with torch.inference_mode():
        prefix_logits, layer_cache = model.prefill(prefix)
        request_cache = [(key[0], value[0]) for key, value in layer_cache]
        for manager in managers.values():
            manager.store_prefill_request("eval", request_cache)

        collected = {name: [prefix_logits[:, -1]] for name in managers}
        for position in range(args.prefix_tokens, len(token_ids) - 1):
            input_token = tokens[position].reshape(1, 1)
            for name, manager in managers.items():
                logits = model.decode_batch(
                    input_token,
                    ["eval"],
                    manager,
                    PagedAttention(manager, "triton"),
                )
                collected[name].append(logits[:, -1])

    targets = tokens[args.prefix_tokens :].reshape(-1)
    fp_logits = torch.cat(collected["fp16"], dim=0)
    int8_logits = torch.cat(collected["int8"], dim=0)
    perplexities = {
        name: torch.exp(
            F.cross_entropy(torch.cat(logits, dim=0).float(), targets)
        ).item()
        for name, logits in collected.items()
    }

    attention_latencies = {}
    query = torch.randn(
        1, model.config.num_heads, model.config.head_dim,
        device="cuda", dtype=torch.float16,
    )
    for name, manager in managers.items():
        manager.reserve_token_slot("eval")
        for layer_id in range(model.config.num_layers):
            manager.write_layer_kv(
                "eval",
                layer_id,
                torch.randn_like(query[0]),
                torch.randn_like(query[0]),
            )
        metadata = manager.build_decode_metadata(["eval"])
        attention = PagedAttention(manager, "triton")
        attention_latencies[name] = latency_us(
            lambda: attention.forward_batch(
                ["eval"], model.config.num_layers - 1, query, metadata
            ),
            args.warmup,
            args.iterations,
        )

    capacity = {
        name: {
            "bytes_per_token": manager.bytes_per_block / manager.block_size,
            "token_capacity": manager.total_available_blocks * manager.block_size,
        }
        for name, manager in managers.items()
    }
    result = {
        "settings": {**vars(args), "output": str(args.output)},
        "capacity": capacity,
        "capacity_multiplier": (
            capacity["int8"]["token_capacity"] / capacity["fp16"]["token_capacity"]
        ),
        "decode_attention_latency_us": attention_latencies,
        "mean_absolute_logit_error": (int8_logits - fp_logits).abs().mean().item(),
        "maximum_absolute_logit_error": (int8_logits - fp_logits).abs().max().item(),
        "proxy_perplexity": perplexities,
        "proxy_perplexity_delta": perplexities["int8"] - perplexities["fp16"],
        "note": "Perplexity is a deterministic engineering proxy on EVALUATION_TEXT, not a corpus benchmark.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
