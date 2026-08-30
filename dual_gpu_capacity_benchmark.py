"""Measure aggregate capacity from independent replicas on multiple GPUs.

Each GPU receives one model replica and an equal share of the offered request
rate. GPU memory is not combined: every worker owns its model and KV cache.
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_rate(value):
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("request rate must be positive")
    return parsed


def positive_float(value):
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def request_shape(value):
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "request shape must be INPUT_TOKENS:OUTPUT_TOKENS:WEIGHT"
        )
    try:
        parsed = (int(parts[0]), int(parts[1]), float(parts[2]))
    except ValueError as error:
        raise argparse.ArgumentTypeError("invalid request shape") from error
    if any(number <= 0 for number in parsed):
        raise argparse.ArgumentTypeError("request shape values must be positive")
    return parsed


def percentile(values, percent):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def latency_summary(values):
    if not values:
        return None
    return {
        "median": statistics.median(values),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark data-parallel inference replicas on two T4 GPUs"
    )
    parser.add_argument(
        "--engine",
        choices=("hf", "pagedserve", "vllm"),
        required=True,
    )
    parser.add_argument("--pagedserve-strategy", choices=("orca", "sarathi"), default="orca")
    parser.add_argument(
        "--decode-attention-backend",
        choices=("torch", "triton"),
        default="torch",
    )
    parser.add_argument("--model-id", default="openai-community/gpt2")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--input-length", type=positive_int, required=True)
    parser.add_argument("--output-length", type=positive_int, required=True)
    parser.add_argument("--num-requests-per-replica", type=positive_int, default=120)
    parser.add_argument(
        "--duration-seconds",
        type=positive_float,
        help="generate arrivals for this duration on each replica",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--arrival-pattern",
        choices=("fixed", "poisson"),
        default="fixed",
    )
    parser.add_argument(
        "--request-shape",
        type=request_shape,
        action="append",
    )
    parser.add_argument("--request-rate", type=positive_rate, action="append", required=True)
    parser.add_argument("--gpu", action="append", help="physical GPU id (default: 0 and 1)")
    parser.add_argument("--max-batch-size", type=positive_int, default=64)
    parser.add_argument("--kv-cache-memory-mb", type=positive_int)
    parser.add_argument(
        "--kv-cache-memory-utilization",
        type=float,
        help=(
            "PagedServe-only override. Omit it for a fair comparison using "
            "--gpu-memory-utilization for both engines."
        ),
    )
    parser.add_argument("--kv-cache-safety-mb", type=positive_int, default=3072)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--ttft-slo-ms", type=float)
    parser.add_argument("--tpot-slo-ms", type=float)
    parser.add_argument("--e2e-slo-ms", type=float)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/pagedserve-dual-gpu"),
    )
    return parser.parse_args()


def build_worker_command(args, worker_index, output_path):
    replica_rate_divisor = len(args.gpu)
    command = [
        sys.executable,
        str(Path(__file__).with_name("comparison_benchmark.py")),
        "--engine",
        args.engine,
        "--model-id",
        args.model_id,
        "--dtype",
        args.dtype,
        "--input-length",
        str(args.input_length),
        "--output-length",
        str(args.output_length),
        "--num-requests",
        str(args.num_requests_per_replica),
        "--max-batch-size",
        str(args.max_batch_size),
        "--seed",
        str(args.seed + worker_index),
        "--arrival-pattern",
        args.arrival_pattern,
        "--json-output",
        str(output_path),
    ]
    for total_rate in args.request_rate:
        command.extend(("--request-rate", str(total_rate / replica_rate_divisor)))
    if args.duration_seconds is not None:
        command.extend(("--duration-seconds", str(args.duration_seconds)))
    for input_length, output_length, weight in args.request_shape or []:
        command.extend(
            (
                "--request-shape",
                f"{input_length}:{output_length}:{weight}",
            )
        )

    if args.engine == "pagedserve":
        pagedserve_memory_utilization = (
            args.kv_cache_memory_utilization
            if args.kv_cache_memory_utilization is not None
            else args.gpu_memory_utilization
        )
        command.extend(
            (
                "--pagedserve-strategy",
                args.pagedserve_strategy,
                "--decode-attention-backend",
                args.decode_attention_backend,
                "--kv-cache-memory-utilization",
                str(pagedserve_memory_utilization),
                "--kv-cache-safety-mb",
                str(args.kv_cache_safety_mb),
                "--warmup-batch-size",
                str(args.max_batch_size),
            )
        )
        if args.kv_cache_memory_mb is not None:
            command.extend(("--kv-cache-memory-mb", str(args.kv_cache_memory_mb)))
    elif args.engine == "vllm":
        command.extend(
            ("--gpu-memory-utilization", str(args.gpu_memory_utilization))
        )

    for option, value in (
        ("--ttft-slo-ms", args.ttft_slo_ms),
        ("--tpot-slo-ms", args.tpot_slo_ms),
        ("--e2e-slo-ms", args.e2e_slo_ms),
    ):
        if value is not None:
            command.extend((option, str(value)))
    return command


def combine_rate(worker_results, total_offered_rate):
    raw_requests = [
        request
        for result in worker_results
        for request in result["raw_requests"]
        if (
            request["error"] is None
            and request["generated_tokens"] == request["requested_output_tokens"]
            and request["ttft_seconds"] is not None
        )
    ]
    ttfts = [request["ttft_seconds"] for request in raw_requests]
    queue_delays = [request["queue_delay_seconds"] for request in raw_requests]
    engine_ttfts = [request["engine_ttft_seconds"] for request in raw_requests]
    tpots = [request["tpot_seconds"] for request in raw_requests]
    itls = [
        interval
        for request in raw_requests
        for interval in request["inter_token_seconds"]
    ]
    e2es = [request["e2e_seconds"] for request in raw_requests]
    durations = [result["duration_seconds"] for result in worker_results]
    measurement_duration = max(durations)
    successful_requests = sum(
        result["successful_requests"] for result in worker_results
    )
    generated_tokens = sum(
        result["generated_tokens"] for result in worker_results
    )
    all_generated_tokens = sum(
        result.get(
            "all_generated_tokens_including_failed_requests",
            result["generated_tokens"],
        )
        for result in worker_results
    )
    coalesced_token_callbacks = sum(
        result.get("coalesced_token_callbacks", 0)
        for result in worker_results
    )
    coalesced_generated_tokens = sum(
        result.get("coalesced_generated_tokens", 0)
        for result in worker_results
    )
    requests_with_coalesced_tokens = sum(
        result.get("requests_with_coalesced_tokens", 0)
        for result in worker_results
    )
    max_tokens_per_callback = max(
        (result.get("max_tokens_per_callback", 1) for result in worker_results),
        default=1,
    )
    slo_good_counts = [result.get("slo_good_requests") for result in worker_results]
    realized_rates = [
        result["realized_arrival_rate"]
        for result in worker_results
        if result["realized_arrival_rate"] is not None
    ]
    shape_groups = {}
    for request in raw_requests:
        shape = (request["input_tokens"], request["requested_output_tokens"])
        shape_groups.setdefault(shape, []).append(request)
    latency_by_request_shape = []
    for (input_tokens, output_tokens), requests in sorted(
        shape_groups.items(),
        key=lambda item: (
            -1 if item[0][0] is None else item[0][0],
            item[0][1],
        ),
    ):
        latency_by_request_shape.append(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "request_count": len(requests),
                "ttft_seconds": latency_summary(
                    [request["ttft_seconds"] for request in requests]
                ),
                "tpot_seconds": latency_summary(
                    [request["tpot_seconds"] for request in requests]
                ),
                "itl_seconds": latency_summary(
                    [
                        interval
                        for request in requests
                        for interval in request["inter_token_seconds"]
                    ]
                ),
                "e2e_seconds": latency_summary(
                    [request["e2e_seconds"] for request in requests]
                ),
            }
        )
    achieved_throughput = successful_requests / measurement_duration
    output_throughput = generated_tokens / measurement_duration
    return {
        "offered_request_rate": total_offered_rate,
        "realized_arrival_rate": sum(realized_rates) if realized_rates else None,
        "peak_outstanding_requests_per_gpu": [
            result["peak_outstanding_requests"] for result in worker_results
        ],
        "aggregate_measurement_duration_seconds": measurement_duration,
        "per_replica_duration_seconds": durations,
        "achieved_request_throughput": achieved_throughput,
        "output_token_throughput": output_throughput,
        "replica_rate_sum_request_throughput": sum(
            result["achieved_request_throughput"] for result in worker_results
        ),
        "replica_rate_sum_output_token_throughput": sum(
            result["output_token_throughput"] for result in worker_results
        ),
        "goodput_requests_per_second": (
            sum(slo_good_counts) / measurement_duration
            if all(value is not None for value in slo_good_counts)
            else None
        ),
        "successful_requests": successful_requests,
        "generated_tokens": generated_tokens,
        "all_generated_tokens_including_failed_requests": all_generated_tokens,
        "coalesced_token_callbacks": coalesced_token_callbacks,
        "coalesced_generated_tokens": coalesced_generated_tokens,
        "requests_with_coalesced_tokens": requests_with_coalesced_tokens,
        "max_tokens_per_callback": max_tokens_per_callback,
        "failed_requests": sum(
            len(result["failed_requests"]) for result in worker_results
        ),
        "ttft_seconds": latency_summary(ttfts),
        "queue_delay_seconds": latency_summary(queue_delays),
        "engine_ttft_seconds": latency_summary(engine_ttfts),
        "tpot_seconds": latency_summary(tpots),
        "itl_seconds": latency_summary(itls),
        "e2e_seconds": latency_summary(e2es),
        "latency_by_request_shape": latency_by_request_shape,
        "offered_load_delivery_ratio": achieved_throughput / total_offered_rate,
        "throughput_is_demand_limited": (
            achieved_throughput >= 0.99 * total_offered_rate
        ),
        "per_gpu_telemetry": [
            result["gpu_telemetry"] for result in worker_results
        ],
    }


def triplet(value):
    if value is None:
        return "n/a"
    return (
        f"{value['median'] * 1000:.2f}/"
        f"{value['p95'] * 1000:.2f}/"
        f"{value['p99'] * 1000:.2f}"
    )


def main():
    args = parse_args()
    args.gpu = args.gpu or ["0", "1"]
    if len(args.gpu) < 2 or len(args.gpu) != len(set(args.gpu)):
        raise ValueError("Provide at least two unique GPU ids")
    if (
        args.kv_cache_memory_utilization is not None
        and not 0 < args.kv_cache_memory_utilization <= 1
    ):
        raise ValueError("kv_cache_memory_utilization must be in (0, 1]")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    log_handles = []
    output_paths = []
    try:
        for worker_index, gpu_id in enumerate(args.gpu):
            output_path = args.output_dir / f"worker-{worker_index}.json"
            log_path = args.output_dir / f"worker-{worker_index}.log"
            command = build_worker_command(args, worker_index, output_path)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu_id
            log_handle = log_path.open("w")
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
                cwd=Path(__file__).parent,
            )
            processes.append((gpu_id, process, log_path))
            log_handles.append(log_handle)
            output_paths.append(output_path)

        failed = []
        for gpu_id, process, log_path in processes:
            return_code = process.wait()
            if return_code != 0:
                failed.append((gpu_id, return_code, log_path))
    finally:
        for log_handle in log_handles:
            log_handle.close()

    if failed:
        for gpu_id, return_code, log_path in failed:
            print(f"GPU {gpu_id} worker failed with code {return_code}: {log_path}")
            print("\n".join(log_path.read_text().splitlines()[-40:]))
        raise SystemExit(1)

    reports = [json.loads(path.read_text()) for path in output_paths]
    model_metadata = reports[0]["model_metadata"]
    model_config = model_metadata["config"]
    parameter_count = model_metadata["theoretical_parameter_count"]
    runtime_weight_bytes = model_metadata["theoretical_runtime_weight_bytes"]
    print(
        f"Model profile: parameters="
        f"{f'{parameter_count:,}' if parameter_count is not None else 'n/a'} | "
        f"runtime weights="
        f"{f'{runtime_weight_bytes / (1024 ** 2):.1f} MiB' if runtime_weight_bytes is not None else 'n/a'} | "
        f"layers/heads/KV-heads={model_config['num_layers']}/"
        f"{model_config['num_attention_heads']}/{model_config['num_key_value_heads']} | "
        f"hidden={model_config['hidden_size']} | context={model_config['max_position_embeddings']}"
    )
    for gpu_id, report in zip(args.gpu, reports):
        metadata = report["engine_metadata"]
        if args.engine == "pagedserve":
            print(
                f"GPU {gpu_id}: model={metadata['model_parameter_bytes'] / (1024 ** 2):.1f} MiB, "
                f"KV={metadata['kv_cache_memory_bytes'] / (1024 ** 2):.1f} MiB, "
                f"capacity={metadata['kv_cache_capacity_tokens']:,} tokens / "
                f"{metadata['kv_cache_capacity_max_length_requests']:,} max-length requests"
            )
        elif args.engine == "vllm":
            cache = metadata["vllm_cache_config"]
            print(
                f"GPU {gpu_id}: vLLM {metadata['vllm_version']} | "
                f"KV bytes={cache['kv_cache_memory_bytes']} | "
                f"KV tokens={cache['kv_cache_size_tokens']} | "
                f"max concurrency={cache['kv_cache_max_concurrency']}"
            )
        else:
            print(
                f"GPU {gpu_id}: Hugging Face Transformers "
                f"{metadata['transformers_version']} | sequential FCFS"
            )

    combined = [
        combine_rate(
            [report["results"][index] for report in reports],
            total_rate,
        )
        for index, total_rate in enumerate(args.request_rate)
    ]
    aggregate_report = {
        "profile_schema_version": 4,
        "engine": args.engine,
        "pagedserve_strategy": (
            args.pagedserve_strategy if args.engine == "pagedserve" else None
        ),
        "decode_attention_backend": (
            args.decode_attention_backend if args.engine == "pagedserve" else None
        ),
        "gpu_ids": args.gpu,
        "model_metadata": model_metadata,
        "comparison_contract": reports[0].get("comparison_contract"),
        "settings": {
            "dtype": args.dtype,
            "input_length": args.input_length,
            "output_length": args.output_length,
            "total_sequence_length": args.input_length + args.output_length,
            "arrival_pattern": args.arrival_pattern,
            "request_shapes": args.request_shape,
            "seed": args.seed,
            "num_requests_per_replica": args.num_requests_per_replica,
            "arrival_duration_seconds": args.duration_seconds,
            "max_batch_size": args.max_batch_size,
            "decode_attention_backend": args.decode_attention_backend,
            "offered_request_rates": args.request_rate,
            "kv_cache_memory_mb": args.kv_cache_memory_mb,
            "kv_cache_memory_utilization": (
                args.kv_cache_memory_utilization
                if args.kv_cache_memory_utilization is not None
                else args.gpu_memory_utilization
            ),
            "kv_cache_safety_mb": args.kv_cache_safety_mb,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "memory_budget_comparable_across_engines": (
                args.kv_cache_memory_mb is None
                and (
                    args.kv_cache_memory_utilization is None
                    or math.isclose(
                        args.kv_cache_memory_utilization,
                        args.gpu_memory_utilization,
                    )
                )
            ),
            "ttft_slo_ms": args.ttft_slo_ms,
            "tpot_slo_ms": args.tpot_slo_ms,
            "e2e_slo_ms": args.e2e_slo_ms,
        },
        "claim_guidance": {
            "throughput_capacity_requires_saturation_sweep": True,
            "sustainable_capacity_requires_an_slo": True,
            "mixed_workload_tail_latency_requires_shape_breakdown": True,
            "peak_vram_is_a_preallocated_budget_not_memory_efficiency": True,
            "memory_efficiency_metric": (
                "KV-cache token capacity under the shared total-device cap"
            ),
        },
        "per_gpu_engine_metadata": [
            report["engine_metadata"] for report in reports
        ],
        "worker_result_files": [str(path) for path in output_paths],
        "aggregate_results": combined,
    }
    aggregate_path = args.output_dir / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate_report, indent=2) + "\n")
    comparable_memory = aggregate_report["settings"][
        "memory_budget_comparable_across_engines"
    ]
    print(
        "Memory budget: "
        f"{args.gpu_memory_utilization * 100:.1f}% total-device utilization target | "
        f"fair cross-engine budget={'yes' if comparable_memory else 'no'}"
    )
    if not comparable_memory:
        print(
            "WARNING: engine-specific KV memory settings were supplied; do not "
            "make cross-engine VRAM-efficiency claims from this run."
        )
    else:
        print(
            "Peak device VRAM includes preallocated KV pools; treat it as budget "
            "usage, not memory efficiency. Compare KV-token capacity under this "
            "shared cap instead."
        )
    print(
        "total offered RPS | aggregate achieved RPS | SLO goodput RPS | output tok/s | "
        "TTFT p50/p95/p99 (ms) | TPOT p50/p95/p99 (ms) | "
        "ITL p50/p95/p99 (ms) | E2E p50/p95/p99 (ms) | failures"
    )
    print("-" * 150)
    for result in combined:
        goodput = result["goodput_requests_per_second"]
        goodput_text = "n/a" if goodput is None else f"{goodput:.3f}"
        print(
            f"{result['offered_request_rate']:.1f} | "
            f"{result['achieved_request_throughput']:.3f} | "
            f"{goodput_text} | "
            f"{result['output_token_throughput']:.2f} | "
            f"{triplet(result['ttft_seconds'])} | "
            f"{triplet(result['tpot_seconds'])} | "
            f"{triplet(result['itl_seconds'])} | "
            f"{triplet(result['e2e_seconds'])} | "
            f"{result['failed_requests']}"
        )
        print(
            f"  queue delay p50/p95/p99 "
            f"{triplet(result['queue_delay_seconds'])} ms | "
            f"engine TTFT after submit "
            f"{triplet(result['engine_ttft_seconds'])} ms"
        )
        realized_rate = result["realized_arrival_rate"]
        realized_text = "burst" if realized_rate is None else f"{realized_rate:.3f}"
        print(
            f"  realized aggregate arrivals {realized_text} RPS | "
            f"peak outstanding per GPU "
            f"{result['peak_outstanding_requests_per_gpu']}"
        )
        load_label = (
            "demand-limited; this row does not measure maximum capacity"
            if result["throughput_is_demand_limited"]
            else "did not keep up with offered load"
        )
        print(
            f"  delivered {result['offered_load_delivery_ratio'] * 100:.2f}% "
            f"of offered RPS | {load_label}"
        )
        if result.get("coalesced_token_callbacks", 0):
            print(
                "  stream coalescing: "
                f"{result['coalesced_token_callbacks']} multi-token callbacks | "
                f"{result['coalesced_generated_tokens']} tokens in coalesced "
                f"callbacks | {result['requests_with_coalesced_tokens']} "
                f"requests affected | max "
                f"{result['max_tokens_per_callback']} tokens/callback"
            )
            print(
                "  latency note: TTFT/E2E use observed callback delivery times; "
                "TPOT/ITL include zero-length gaps for tokens delivered together "
                "in one callback."
            )
        for shape in result["latency_by_request_shape"]:
            print(
                f"  shape {shape['input_tokens']}+{shape['output_tokens']} "
                f"(n={shape['request_count']}): TTFT {triplet(shape['ttft_seconds'])} ms | "
                f"TPOT {triplet(shape['tpot_seconds'])} ms | "
                f"ITL {triplet(shape['itl_seconds'])} ms | "
                f"E2E {triplet(shape['e2e_seconds'])} ms"
            )
        for gpu_id, telemetry in zip(args.gpu, result["per_gpu_telemetry"]):
            if not telemetry or not telemetry.get("sample_count"):
                continue
            utilization = telemetry["gpu_kernel_active_percent"]
            memory_activity = telemetry["memory_active_percent"]
            memory = telemetry["gpu_memory_used_mb"]
            power = telemetry["power_draw_watts"]
            print(
                f"  GPU {gpu_id}: kernel mean/p95/max "
                f"{utilization['mean']:.1f}/{utilization['p95']:.1f}/"
                f"{utilization['maximum']:.1f}% | memory activity mean/p95/max "
                f"{memory_activity['mean']:.1f}/{memory_activity['p95']:.1f}/"
                f"{memory_activity['maximum']:.1f}% | VRAM mean/max "
                f"{memory['mean']:.0f}/{memory['maximum']:.0f} MiB | "
                f"power mean/max {power['mean']:.1f}/{power['maximum']:.1f} W | "
                f"energy~{telemetry.get('estimated_energy_joules', float('nan')):.1f} J"
            )
    print(f"Aggregate raw results written to {aggregate_path}")


if __name__ == "__main__":
    main()
