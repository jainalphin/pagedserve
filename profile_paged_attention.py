"""Capture a Chrome trace for metadata, Torch gather, and Triton direct reads."""

import argparse
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from src.model.kv_manager import KVCacheManager
from src.model.paged_attention import PagedAttention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Profiling the Triton kernel requires CUDA")
    dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]
    layers, heads, head_dim, block_size = 3, 12, 64, 16
    blocks = args.batch_size * ((args.context_length + 15) // 16) + 32
    bytes_per_block = 2 * layers * heads * block_size * head_dim * torch.empty(
        (), dtype=dtype
    ).element_size()
    manager = KVCacheManager(
        block_size=block_size,
        total_memory=blocks * bytes_per_block,
        tensor_dtype=dtype,
        device="cuda",
        num_layers=layers,
        num_kv_heads=heads,
        head_dim=head_dim,
    )
    request_ids = list(range(args.batch_size))
    for request_id in request_ids:
        prompt = args.context_length - 1
        manager.store_prefill_request(
            request_id,
            [
                (
                    torch.randn(heads, prompt, head_dim, device="cuda", dtype=dtype),
                    torch.randn(heads, prompt, head_dim, device="cuda", dtype=dtype),
                )
                for _ in range(layers)
            ],
        )
        manager.reserve_token_slot(request_id)
        for layer_id in range(layers):
            manager.write_layer_kv(
                request_id,
                layer_id,
                torch.randn(heads, head_dim, device="cuda", dtype=dtype),
                torch.randn(heads, head_dim, device="cuda", dtype=dtype),
            )
    queries = torch.randn(
        args.batch_size, heads, head_dim, device="cuda", dtype=dtype
    )
    torch_attention = PagedAttention(manager, "torch")
    triton_attention = PagedAttention(manager, "triton")
    metadata = manager.build_decode_metadata(request_ids)
    for _ in range(5):
        triton_attention.forward_batch(request_ids, 2, queries, metadata)
    torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as profiler:
        with record_function("metadata_construction_once_per_iteration"):
            metadata = manager.build_decode_metadata(request_ids)
        for _ in range(args.iterations):
            with record_function("torch_kv_gather_allocations_plus_sdpa"):
                torch_attention.forward_batch(request_ids, 2, queries, metadata)
            with record_function("triton_direct_noncontiguous_paged_reads"):
                triton_attention.forward_batch(request_ids, 2, queries, metadata)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(args.trace))
    print(
        profiler.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=30
        )
    )
    print(f"Chrome trace: {args.trace}")


if __name__ == "__main__":
    main()
