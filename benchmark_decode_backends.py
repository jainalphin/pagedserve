"""Isolated, repeated GPT-2 end-to-end Torch-versus-Triton benchmark."""

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


def csv_ints(value):
    return tuple(int(item) for item in value.split(","))


def one_run(args, backend, batch_size, input_length, repetition, destination):
    output = destination / (
        f"{backend}-b{batch_size}-c{input_length}-run{repetition}.json"
    )
    command = [
        sys.executable,
        "comparison_benchmark.py",
        "--engine", "pagedserve",
        "--model-id", args.model_id,
        "--dtype", args.dtype,
        "--input-length", str(input_length),
        "--output-length", str(args.output_length),
        "--num-requests", str(batch_size),
        "--request-rate", "inf",
        "--max-batch-size", str(batch_size),
        "--warmup-batch-size", str(batch_size),
        "--pagedserve-strategy", "orca",
        "--decode-attention-backend", backend,
        "--kv-cache-dtype", args.kv_cache_dtype,
        "--seed", str(args.seed),
        "--json-output", str(output),
    ]
    if args.disable_cuda_graphs:
        command.append("--disable-cuda-graphs")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "."
    subprocess.run(command, check=True, cwd=Path(__file__).parent, env=environment)
    report = json.loads(output.read_text())
    scenario = report["results"][0]
    allocator = scenario["gpu_telemetry"]["torch_cuda_allocator"]
    return {
        "output_token_throughput": scenario["output_token_throughput"],
        "tpot_ms": scenario["tpot_seconds"]["median"] * 1000,
        "ttft_ms": scenario["ttft_seconds"]["median"] * 1000,
        "peak_gpu_memory_mib": allocator["peak_allocated_bytes"] / 2**20,
        "cuda_graphs": report["engine_metadata"].get("cuda_graphs"),
        "raw_report": str(output),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="openai-community/gpt2")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--kv-cache-dtype", choices=("model", "int8"), default="model")
    parser.add_argument("--batch-sizes", type=csv_ints, default=(1, 8, 32))
    parser.add_argument("--input-lengths", type=csv_ints, default=(128, 512))
    parser.add_argument("--output-length", type=int, default=32)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--disable-cuda-graphs",
        action="store_true",
        help="run the Triton eager ablation instead of CUDA-graph replay",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 3:
        parser.error("--runs must be at least 3")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_directory = args.output.parent / f"{args.output.stem}-raw"
    raw_directory.mkdir(parents=True, exist_ok=True)

    rows = []
    for batch_size in args.batch_sizes:
        for input_length in args.input_lengths:
            for backend in ("torch", "triton"):
                raw = [
                    one_run(
                        args, backend, batch_size, input_length, repetition, raw_directory
                    )
                    for repetition in range(args.runs)
                ]
                rows.append(
                    {
                        "backend": backend,
                        "dtype": args.dtype,
                        "kv_cache_dtype": args.kv_cache_dtype,
                        "batch_size": batch_size,
                        "input_length": input_length,
                        "output_length": args.output_length,
                        "runs": args.runs,
                        "median_output_token_throughput": statistics.median(
                            item["output_token_throughput"] for item in raw
                        ),
                        "median_tpot_ms": statistics.median(item["tpot_ms"] for item in raw),
                        "median_ttft_ms": statistics.median(item["ttft_ms"] for item in raw),
                        "median_peak_gpu_memory_mib": statistics.median(
                            item["peak_gpu_memory_mib"] for item in raw
                        ),
                        "raw_runs": raw,
                    }
                )
    report = {"settings": {**vars(args), "output": str(args.output)}, "results": rows}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("backend | batch | input | tok/s | TPOT ms | TTFT ms | peak GPU MiB")
    for row in rows:
        print(
            f"{row['backend']} | {row['batch_size']} | {row['input_length']} | "
            f"{row['median_output_token_throughput']:.2f} | "
            f"{row['median_tpot_ms']:.3f} | {row['median_ttft_ms']:.3f} | "
            f"{row['median_peak_gpu_memory_mib']:.1f}"
        )


if __name__ == "__main__":
    main()
