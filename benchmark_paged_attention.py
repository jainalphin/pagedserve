"""CUDA microbenchmark: Torch gather+SDPA versus direct Triton paged reads."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F


CONTEXTS = (1, 15, 16, 17, 31, 32, 33, 128, 512, 1024)
BATCHES = (1, 8, 32)


def parse_csv_ints(value):
    parsed = tuple(int(item) for item in value.split(","))
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("values must be positive comma-separated integers")
    return parsed


def torch_attention(queries, key_pool, value_pool, table, context_length):
    keys = key_pool[0, table.to(torch.long)].permute(0, 2, 1, 3, 4)
    values = value_pool[0, table.to(torch.long)].permute(0, 2, 1, 3, 4)
    batch, heads, _, _, dimension = keys.shape
    keys = keys.reshape(batch, heads, -1, dimension)[:, :, :context_length]
    values = values.reshape(batch, heads, -1, dimension)[:, :, :context_length]
    return F.scaled_dot_product_attention(
        queries.unsqueeze(2), keys, values, dropout_p=0.0
    ).squeeze(2)


def elapsed_microseconds(function, warmup, iterations):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def temporary_bytes(function):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = function()
    torch.cuda.synchronize()
    temporary = torch.cuda.max_memory_allocated() - baseline
    del output
    return temporary


def benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires an NVIDIA CUDA GPU")
    from src.kernels.triton_paged_attention import paged_decode_attention_triton

    dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]
    results = []
    for batch_size in args.batch_sizes:
        for context_length in args.context_lengths:
            logical_blocks = (context_length + args.block_size - 1) // args.block_size
            physical_count = batch_size * logical_blocks + 17
            key_pool = torch.randn(
                1, physical_count, args.heads, args.block_size, args.head_dim,
                device="cuda", dtype=dtype,
            )
            value_pool = torch.randn_like(key_pool)
            table = torch.randperm(physical_count, device="cuda")[
                : batch_size * logical_blocks
            ].reshape(batch_size, logical_blocks).to(torch.int32)
            lengths = torch.full(
                (batch_size,), context_length, device="cuda", dtype=torch.int32
            )
            queries = torch.randn(
                batch_size, args.heads, args.head_dim, device="cuda", dtype=dtype
            )
            dummy_scales = torch.empty(1, device="cuda", dtype=torch.float32)

            def run_torch():
                return torch_attention(
                    queries, key_pool, value_pool, table, context_length
                )

            def run_triton():
                return paged_decode_attention_triton(
                    queries,
                    key_pool,
                    value_pool,
                    dummy_scales,
                    dummy_scales,
                    table,
                    lengths,
                    0,
                    validate_inputs=False,
                )

            expected = run_torch()
            actual = run_triton()
            torch.testing.assert_close(
                actual,
                expected,
                atol=3e-2 if dtype == torch.float16 else 3e-4,
                rtol=3e-2 if dtype == torch.float16 else 3e-4,
            )
            for backend, function in (("torch", run_torch), ("triton", run_triton)):
                results.append(
                    {
                        "backend": backend,
                        "dtype": args.dtype,
                        "batch_size": batch_size,
                        "context_length": context_length,
                        "latency_us": elapsed_microseconds(
                            function, args.warmup, args.iterations
                        ),
                        "temporary_gpu_bytes": temporary_bytes(function),
                    }
                )
            del key_pool, value_pool, queries, table, lengths, expected, actual
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float16", "float32"), required=True)
    parser.add_argument("--batch-sizes", type=parse_csv_ints, default=BATCHES)
    parser.add_argument("--context-lengths", type=parse_csv_ints, default=CONTEXTS)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 100:
        parser.error("--iterations must be at least 100")
    report = {
        "settings": vars(args) | {"json_output": str(args.json_output)},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "results": benchmark(args),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n")
    print("backend | dtype | batch | context | latency us | temporary MiB")
    for row in report["results"]:
        print(
            f"{row['backend']} | {row['dtype']} | {row['batch_size']} | "
            f"{row['context_length']} | {row['latency_us']:.2f} | "
            f"{row['temporary_gpu_bytes'] / 2**20:.3f}"
        )


if __name__ == "__main__":
    main()
