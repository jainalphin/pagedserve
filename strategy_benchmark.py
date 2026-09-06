"""Benchmark prefill-induced decode stalls for Orca and Sarathi scheduling."""

import argparse
import json
import math
import time
from pathlib import Path

from benchmark import (
    non_negative_integer,
    positive_integer,
    summarize,
    synchronize_device,
    system_metadata,
)
from main import DISTILGPT2_MODEL, SUPPORTED_DTYPES, SUPPORTED_MODELS, build_scheduler
from src.scheduler.orca_scheduler import ORCA_STRATEGY, SARATHI_STRATEGY


SHORT_PROMPT = "Once upon a time"
LONG_PROMPT_BASE = (
    "Paged attention stores key and value tensors in reusable blocks while continuous "
    "batching combines active decode requests with newly arriving prompts. Chunked "
    "prefill divides long prompts into smaller pieces so decode requests can continue "
    "making progress without waiting behind one large prefill operation. Decode maximal "
    "batching then combines one prefill chunk with as many active decode tokens as the "
    "batch can hold. "
)


def run_trial(
    scheduler,
    strategy,
    active_decode_count,
    active_max_new_tokens,
    long_prompt,
):
    # Establish the exact same active-decode state before switching strategies.
    scheduler.scheduling_strategy = ORCA_STRATEGY
    active_ids = [
        scheduler.add_request(SHORT_PROMPT, max_new_tokens=active_max_new_tokens)
        for _ in range(active_decode_count)
    ]
    scheduler.step()
    if not all(request_id in scheduler.active for request_id in active_ids):
        raise RuntimeError("An active request ended before the stall measurement")

    scheduler.scheduling_strategy = strategy
    long_request_id = scheduler.add_request(long_prompt, max_new_tokens=1)
    iteration_seconds = []
    synchronize_device()
    started_at = time.perf_counter()

    while long_request_id not in scheduler.finished:
        synchronize_device()
        iteration_started_at = time.perf_counter()
        emitted = scheduler.step()
        synchronize_device()
        iteration_seconds.append(time.perf_counter() - iteration_started_at)
        if not all(request_id in emitted for request_id in active_ids):
            raise RuntimeError("A decode request stalled for an entire iteration")

    long_prompt_ttft = time.perf_counter() - started_at
    scheduler.run_until_complete()
    return {
        "long_prompt_ttft_seconds": long_prompt_ttft,
        "maximum_decode_stall_seconds": max(iteration_seconds),
        "prefill_iterations": len(iteration_seconds),
        "iteration_seconds": iteration_seconds,
    }


def benchmark_strategy(
    model_name,
    strategy,
    prefill_chunk_size,
    active_decode_count,
    long_prompt,
    warmup_runs,
    measured_runs,
    max_batch_size,
    kv_cache_memory_mb,
    execution_dtype,
):
    scheduler = build_scheduler(
        model_name,
        scheduling_strategy=strategy,
        prefill_chunk_size=prefill_chunk_size,
        max_batch_size=max_batch_size,
        kv_cache_memory_mb=kv_cache_memory_mb,
        execution_dtype=execution_dtype,
    )
    if active_decode_count >= scheduler.max_batch_size:
        raise ValueError("active decodes must leave one slot for a prefill chunk")

    prompt_tokens = len(scheduler.tokenizer.encode(long_prompt))
    sarathi_iterations = math.ceil(prompt_tokens / prefill_chunk_size)
    active_max_new_tokens = sarathi_iterations + 3

    for _ in range(warmup_runs):
        run_trial(
            scheduler,
            strategy,
            active_decode_count,
            active_max_new_tokens,
            long_prompt,
        )

    raw_runs = [
        run_trial(
            scheduler,
            strategy,
            active_decode_count,
            active_max_new_tokens,
            long_prompt,
        )
        for _ in range(measured_runs)
    ]
    return {
        "model": model_name,
        "strategy": strategy,
        "device": str(next(scheduler.model_engine.parameters()).device),
        "dtype": str(next(scheduler.model_engine.parameters()).dtype),
        "prompt_tokens": prompt_tokens,
        "prefill_chunk_size": (
            prefill_chunk_size if strategy == SARATHI_STRATEGY else None
        ),
        "long_prompt_ttft_seconds": summarize(
            [run["long_prompt_ttft_seconds"] for run in raw_runs]
        ),
        "maximum_decode_stall_seconds": summarize(
            [run["maximum_decode_stall_seconds"] for run in raw_runs]
        ),
        "raw_runs": raw_runs,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare prefill-induced decode stalls across schedulers"
    )
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default=DISTILGPT2_MODEL)
    parser.add_argument("--prefill-chunk-size", type=positive_integer, default=128)
    parser.add_argument("--active-decodes", type=positive_integer, default=3)
    parser.add_argument("--long-prompt-repetitions", type=positive_integer, default=7)
    parser.add_argument("--warmup-runs", type=non_negative_integer, default=2)
    parser.add_argument("--runs", type=positive_integer, default=10)
    parser.add_argument("--max-batch-size", type=positive_integer)
    parser.add_argument("--kv-cache-memory-mb", type=positive_integer)
    parser.add_argument(
        "--dtype",
        choices=SUPPORTED_DTYPES,
        default="float32",
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    long_prompt = LONG_PROMPT_BASE * args.long_prompt_repetitions
    report = {
        "system": system_metadata(),
        "settings": {
            "model": args.model,
            "active_decodes": args.active_decodes,
            "prefill_chunk_size": args.prefill_chunk_size,
            "long_prompt_repetitions": args.long_prompt_repetitions,
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.runs,
            "max_batch_size": args.max_batch_size,
            "kv_cache_memory_mb": args.kv_cache_memory_mb,
            "dtype": args.dtype,
        },
        "results": [],
    }

    for strategy in (ORCA_STRATEGY, SARATHI_STRATEGY):
        print(f"Benchmarking decode stalls with {strategy}...")
        report["results"].append(
            benchmark_strategy(
                model_name=args.model,
                strategy=strategy,
                prefill_chunk_size=args.prefill_chunk_size,
                active_decode_count=args.active_decodes,
                long_prompt=long_prompt,
                warmup_runs=args.warmup_runs,
                measured_runs=args.runs,
                max_batch_size=args.max_batch_size,
                kv_cache_memory_mb=args.kv_cache_memory_mb,
                execution_dtype=args.dtype,
            )
        )

    print("strategy | long-prompt TTFT median/p95 (ms) | decode stall median/p95 (ms)")
    print("-" * 85)
    for result in report["results"]:
        ttft = result["long_prompt_ttft_seconds"]
        stall = result["maximum_decode_stall_seconds"]
        print(
            f"{result['strategy']} | "
            f"{ttft['median'] * 1000:.3f}/{ttft['p95'] * 1000:.3f} | "
            f"{stall['median'] * 1000:.3f}/{stall['p95'] * 1000:.3f}"
        )

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Raw results written to {args.json_output}")


if __name__ == "__main__":
    main()
