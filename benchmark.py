"""Reproducible benchmark for every model registered by PagedServe."""

import argparse
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from main import SUPPORTED_DTYPES, SUPPORTED_MODELS, build_scheduler
from src.scheduler.orca_scheduler import (
    ORCA_STRATEGY,
    SUPPORTED_SCHEDULING_STRATEGIES,
)


def positive_integer(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def non_negative_integer(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def synchronize_device():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def percentile(values, percent):
    """Return a linearly interpolated percentile for a non-empty sequence."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize(values):
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 95),
        "minimum": min(values),
        "maximum": max(values),
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def run_workload(scheduler, prompt, batch_size, max_new_tokens):
    request_ids = [
        scheduler.add_request(prompt, max_new_tokens=max_new_tokens)
        for _ in range(batch_size)
    ]

    synchronize_device()
    started_at = time.perf_counter()
    first_token_seconds = None

    while any(request_id not in scheduler.finished for request_id in request_ids):
        emitted_tokens = scheduler.step()
        synchronize_device()
        if first_token_seconds is None and emitted_tokens:
            first_token_seconds = time.perf_counter() - started_at

    elapsed_seconds = time.perf_counter() - started_at
    generated_tokens = sum(
        len(scheduler.finished[request_id].generated_token_ids)
        for request_id in request_ids
    )
    return {
        "latency_seconds": elapsed_seconds,
        "first_token_seconds": first_token_seconds,
        "generated_tokens": generated_tokens,
        "tokens_per_second": generated_tokens / elapsed_seconds,
    }


def benchmark_model(
    model_name,
    prompt,
    batch_size,
    max_new_tokens,
    warmup_runs,
    measured_runs,
    seed,
    scheduling_strategy,
    prefill_chunk_size,
    max_batch_size,
    kv_cache_memory_mb,
    execution_dtype,
):
    # This makes the locally initialized reference model reproducible.
    torch.manual_seed(seed)
    synchronize_device()
    load_started_at = time.perf_counter()
    scheduler = build_scheduler(
        model_name,
        scheduling_strategy=scheduling_strategy,
        prefill_chunk_size=prefill_chunk_size,
        max_batch_size=max_batch_size,
        kv_cache_memory_mb=kv_cache_memory_mb,
        execution_dtype=execution_dtype,
    )
    synchronize_device()
    load_seconds = time.perf_counter() - load_started_at

    if batch_size > scheduler.max_batch_size:
        raise ValueError(
            f"batch size {batch_size} exceeds scheduler maximum "
            f"{scheduler.max_batch_size}"
        )

    for _ in range(warmup_runs):
        run_workload(scheduler, prompt, batch_size, max_new_tokens)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    raw_runs = [
        run_workload(scheduler, prompt, batch_size, max_new_tokens)
        for _ in range(measured_runs)
    ]
    latencies = [run["latency_seconds"] for run in raw_runs]
    first_token_times = [run["first_token_seconds"] for run in raw_runs]
    total_tokens = sum(run["generated_tokens"] for run in raw_runs)
    total_seconds = sum(latencies)
    model = scheduler.model_engine

    return {
        "model": model_name,
        "strategy": scheduling_strategy,
        "prefill_chunk_size": (
            prefill_chunk_size if scheduling_strategy != ORCA_STRATEGY else None
        ),
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
        "max_batch_size": scheduler.max_batch_size,
        "kv_cache_memory_mb": scheduler.kv_manager.total_memory / (1024 * 1024),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "prompt_tokens": len(scheduler.tokenizer.encode(prompt)),
        "load_seconds": load_seconds,
        "latency_seconds": summarize(latencies),
        "first_token_seconds": summarize(first_token_times),
        "aggregate_tokens_per_second": total_tokens / total_seconds,
        "total_generated_tokens": total_tokens,
        "expected_generated_tokens": batch_size * max_new_tokens * measured_runs,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if torch.cuda.is_available()
            else None
        ),
        "raw_runs": raw_runs,
    }


def system_metadata():
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "logical_cpu_count": os.cpu_count(),
        "pytorch_intraop_threads": torch.get_num_threads(),
        "pytorch_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": torch.cuda.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        metadata["cuda_device"] = {
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_mb": properties.total_memory / (1024 * 1024),
            "cuda_runtime_version": torch.version.cuda,
        }
    else:
        metadata["cuda_device"] = None
    return metadata


def print_results(report):
    metadata = report["system"]
    settings = report["settings"]
    print(
        f"Platform: {metadata['platform']} | PyTorch {metadata['pytorch_version']} | "
        f"threads {metadata['pytorch_intraop_threads']} | seed {settings['seed']}"
    )
    print(
        "model | strategy | device/dtype | params | load (s) | latency median/p95 (s) | "
        "TTFT median/p95 (s) | tokens/s | tokens"
    )
    print("-" * 125)
    for result in report["results"]:
        latency = result["latency_seconds"]
        first_token = result["first_token_seconds"]
        print(
            f"{result['model']} | "
            f"{result['strategy']} | "
            f"{result['device']}/{result['dtype']} | "
            f"{result['parameter_count']:,} | "
            f"{result['load_seconds']:.6f} | "
            f"{latency['median']:.6f}/{latency['p95']:.6f} | "
            f"{first_token['median']:.6f}/{first_token['p95']:.6f} | "
            f"{result['aggregate_tokens_per_second']:.2f} | "
            f"{result['total_generated_tokens']}/"
            f"{result['expected_generated_tokens']}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark PagedServe model loading and generation"
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=SUPPORTED_MODELS,
        help="model to benchmark; repeat for multiple models (default: all)",
    )
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--batch-size", type=positive_integer, default=1)
    parser.add_argument("--max-new-tokens", type=positive_integer, default=16)
    parser.add_argument("--warmup-runs", type=non_negative_integer, default=1)
    parser.add_argument("--runs", type=positive_integer, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--strategy",
        action="append",
        choices=SUPPORTED_SCHEDULING_STRATEGIES,
        help="strategy to benchmark; repeat to compare strategies (default: orca)",
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=positive_integer,
        default=16,
        help="prompt tokens per Sarathi prefill chunk",
    )
    parser.add_argument(
        "--max-batch-size",
        type=positive_integer,
        help="maximum requests per scheduler iteration",
    )
    parser.add_argument(
        "--kv-cache-memory-mb",
        type=positive_integer,
        help="KV-cache memory budget in MiB",
    )
    parser.add_argument(
        "--dtype",
        choices=SUPPORTED_DTYPES,
        default="float32",
        help="execution dtype; benchmark different dtypes separately",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="optional path for metadata, summary statistics, and every raw run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_names = args.model or SUPPORTED_MODELS
    strategies = args.strategy or (ORCA_STRATEGY,)
    report = {
        "system": system_metadata(),
        "settings": {
            "models": list(model_names),
            "strategies": list(strategies),
            "prompt": args.prompt,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.runs,
            "seed": args.seed,
            "prefill_chunk_size": args.prefill_chunk_size,
            "max_batch_size": args.max_batch_size,
            "kv_cache_memory_mb": args.kv_cache_memory_mb,
            "dtype": args.dtype,
        },
        "results": [],
    }

    for model_name in model_names:
        for scheduling_strategy in strategies:
            print(f"Benchmarking {model_name} with {scheduling_strategy}...")
            report["results"].append(
                benchmark_model(
                    model_name=model_name,
                    prompt=args.prompt,
                    batch_size=args.batch_size,
                    max_new_tokens=args.max_new_tokens,
                    warmup_runs=args.warmup_runs,
                    measured_runs=args.runs,
                    seed=args.seed,
                    scheduling_strategy=scheduling_strategy,
                    prefill_chunk_size=args.prefill_chunk_size,
                    max_batch_size=args.max_batch_size,
                    kv_cache_memory_mb=args.kv_cache_memory_mb,
                    execution_dtype=args.dtype,
                )
            )

    print()
    print_results(report)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Raw results written to {args.json_output}")


if __name__ == "__main__":
    main()
