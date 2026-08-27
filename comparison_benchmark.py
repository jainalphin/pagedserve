"""Common open-loop benchmark for HF, PagedServe, and vLLM.

Run one engine per process so model memory and CUDA state cannot leak between
comparisons. Every engine receives the same deterministic token IDs and arrival
trace. Hugging Face is intentionally a sequential FCFS baseline; PagedServe and
vLLM use their native continuous-batching schedulers.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmark import positive_integer
from main import (
    DEFAULT_CUDA_KV_MEMORY_UTILIZATION,
    DEFAULT_CUDA_KV_SAFETY_MB,
    GPT2_MODEL,
    SUPPORTED_DECODE_ATTENTION_BACKENDS,
    SUPPORTED_DTYPES,
    SUPPORTED_KV_CACHE_DTYPES,
    build_scheduler,
)


SUPPORTED_ENGINES = ("hf", "pagedserve", "vllm")


@dataclass
class RequestRecord:
    request_index: int
    scheduled_arrival: float
    requested_output_length: int | None = None
    requested_input_length: int | None = None
    submitted_at: float | None = None
    token_times: list[float] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    error: str | None = None


class NvidiaSMIMonitor:
    """Sample whole-device telemetry while only the measured workload runs."""

    def __init__(self, interval_ms=200, gpu_id="0"):
        self.interval_ms = interval_ms
        self.gpu_id = gpu_id
        self.process = None

    def start(self):
        self.process = subprocess.Popen(
            [
                "nvidia-smi",
                f"--id={self.gpu_id}",
                "--query-gpu=utilization.gpu,utilization.memory,"
                "memory.used,memory.total,power.draw,power.limit",
                "--format=csv,noheader,nounits",
                f"--loop-ms={self.interval_ms}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def stop(self):
        if self.process is None:
            return None
        self.process.terminate()
        try:
            output, error = self.process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            output, error = self.process.communicate()

        samples = []
        for line in output.splitlines():
            try:
                samples.append(
                    [float(value.strip()) for value in line.split(",")]
                )
            except ValueError:
                continue
        if not samples:
            return {"sample_count": 0, "error": error.strip() or None}

        gpu_utilization = [sample[0] for sample in samples]
        memory_activity = [sample[1] for sample in samples]
        memory_used_mb = [sample[2] for sample in samples]
        memory_total_mb = [sample[3] for sample in samples]
        power_draw_watts = [sample[4] for sample in samples]
        power_limit_watts = [sample[5] for sample in samples]
        return {
            "sample_count": len(samples),
            "sample_interval_ms": self.interval_ms,
            "gpu_kernel_active_percent": summarize(gpu_utilization),
            "memory_active_percent": summarize(memory_activity),
            "gpu_memory_used_mb": summarize(memory_used_mb),
            "gpu_memory_total_mb": max(memory_total_mb),
            "power_draw_watts": summarize(power_draw_watts),
            "power_limit_watts": max(power_limit_watts),
        }


class NullMonitor:
    def start(self):
        return None

    def stop(self):
        return None


class TorchCUDAMemoryProfiler:
    def __init__(self):
        self.enabled = torch.cuda.is_available()
        self.start_allocated = None
        self.start_reserved = None

    def start(self):
        if not self.enabled:
            return
        synchronize_cuda()
        torch.cuda.reset_peak_memory_stats()
        self.start_allocated = torch.cuda.memory_allocated()
        self.start_reserved = torch.cuda.memory_reserved()

    def stop(self):
        if not self.enabled:
            return None
        synchronize_cuda()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "start_allocated_bytes": self.start_allocated,
            "start_reserved_bytes": self.start_reserved,
            "end_allocated_bytes": torch.cuda.memory_allocated(),
            "end_reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "device_free_bytes_at_end": free_bytes,
            "device_total_bytes": total_bytes,
        }


def physical_gpu_id():
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    return visible_devices.split(",", maxsplit=1)[0].strip() or "0"


def nvidia_smi_snapshot():
    if not torch.cuda.is_available():
        return None
    fields = (
        "memory.used,memory.total,utilization.gpu,utilization.memory,"
        "power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem,pstate"
    )
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                f"--id={physical_gpu_id()}",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": str(error)}
    values = [value.strip() for value in output.split(",")]
    names = fields.split(",")
    result = {}
    for name, value in zip(names, values):
        try:
            result[name] = float(value)
        except ValueError:
            result[name] = value
    return result


def create_monitor(args):
    if torch.cuda.is_available():
        return NvidiaSMIMonitor(
            args.telemetry_interval_ms,
            gpu_id=physical_gpu_id(),
        )
    return NullMonitor()


def percentile(values, percent):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize(values):
    if not values:
        return None
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "minimum": min(values),
        "maximum": max(values),
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def torch_dtype(dtype_name):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def module_memory_profile(model):
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    return {
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "parameter_bytes": sum(
            parameter.numel() * parameter.element_size()
            for parameter in parameters
        ),
        "buffer_count": sum(buffer.numel() for buffer in buffers),
        "buffer_bytes": sum(
            buffer.numel() * buffer.element_size() for buffer in buffers
        ),
        "dtype": str(parameters[0].dtype) if parameters else None,
        "device": str(parameters[0].device) if parameters else None,
    }


def config_profile(config):
    def first(*names):
        for name in names:
            value = getattr(config, name, None)
            if value is not None:
                return value
        return None

    return {
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "vocab_size": first("vocab_size"),
        "hidden_size": first("hidden_size", "n_embd"),
        "num_layers": first("num_hidden_layers", "n_layer"),
        "num_attention_heads": first("num_attention_heads", "n_head"),
        "num_key_value_heads": first(
            "num_key_value_heads",
            "num_attention_heads",
            "n_head",
        ),
        "intermediate_size": first("intermediate_size", "n_inner"),
        "max_position_embeddings": first(
            "max_position_embeddings",
            "n_positions",
            "n_ctx",
        ),
    }


def causal_lm_from_config(config):
    """Instantiate GPT-2 directly to avoid fragile lazy AutoModel resolution."""
    if getattr(config, "model_type", None) == "gpt2":
        from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

        return GPT2LMHeadModel(config)

    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_config(config)


def causal_lm_from_pretrained(model_id, config, dtype):
    if getattr(config, "model_type", None) == "gpt2":
        from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

        return GPT2LMHeadModel.from_pretrained(
            model_id,
            config=config,
            dtype=dtype,
        )

    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        dtype=dtype,
    )


def huggingface_cache_profile(model_id):
    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
        repository = next(
            repo
            for repo in cache.repos
            if repo.repo_type == "model" and repo.repo_id == model_id
        )
        revisions = []
        for revision in repository.revisions:
            revisions.append(
                {
                    "commit_hash": revision.commit_hash,
                    "size_on_disk_bytes": revision.size_on_disk,
                    "files": sorted(
                        (
                            {
                                "name": file.file_name,
                                "size_on_disk_bytes": file.size_on_disk,
                            }
                            for file in revision.files
                        ),
                        key=lambda item: item["name"],
                    ),
                }
            )
        return {
            "repository_size_on_disk_bytes": repository.size_on_disk,
            "revisions": revisions,
        }
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


def synchronize_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def arrival_offsets(
    request_rate,
    num_requests,
    pattern="fixed",
    seed=1234,
    duration_seconds=None,
):
    if duration_seconds is not None:
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if math.isinf(request_rate):
            raise ValueError("A burst rate cannot be combined with a duration")
        if pattern == "fixed":
            request_count = math.ceil(request_rate * duration_seconds)
            return [index / request_rate for index in range(request_count)]
        if pattern != "poisson":
            raise ValueError(f"Unknown arrival pattern: {pattern}")
        generator = random.Random(seed)
        offsets = [0.0]
        while True:
            next_offset = offsets[-1] + generator.expovariate(request_rate)
            if next_offset >= duration_seconds:
                return offsets
            offsets.append(next_offset)
    if math.isinf(request_rate):
        return [0.0] * num_requests
    if pattern == "poisson":
        generator = random.Random(seed)
        offsets = [0.0]
        for _ in range(1, num_requests):
            offsets.append(offsets[-1] + generator.expovariate(request_rate))
        return offsets
    if pattern != "fixed":
        raise ValueError(f"Unknown arrival pattern: {pattern}")
    return [request_index / request_rate for request_index in range(num_requests)]


def deterministic_prompts(tokenizer, input_length, num_requests, seed):
    special_ids = set(getattr(tokenizer, "all_special_ids", ()))
    vocab_size = len(tokenizer)
    prompts = []
    for request_index in range(num_requests):
        generator = random.Random(seed + request_index)
        prompt = []
        while len(prompt) < input_length:
            token_id = generator.randrange(vocab_size)
            if token_id not in special_ids:
                prompt.append(token_id)
        prompts.append(prompt)
    return prompts


def parse_request_shape(value):
    try:
        input_length, output_length, weight = value.split(":")
        parsed = (int(input_length), int(output_length), float(weight))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            "request shape must be INPUT_TOKENS:OUTPUT_TOKENS:WEIGHT"
        ) from error
    if parsed[0] <= 0 or parsed[1] <= 0 or parsed[2] <= 0:
        raise argparse.ArgumentTypeError("request shape values must be positive")
    return parsed


def request_shapes(args, num_requests=None):
    shapes = args.request_shape or [
        (args.input_length, args.output_length, 1.0)
    ]
    generator = random.Random(args.seed + 7919)
    selected = generator.choices(
        [(input_length, output_length) for input_length, output_length, _ in shapes],
        weights=[weight for _, _, weight in shapes],
        k=args.num_requests if num_requests is None else num_requests,
    )
    return (
        [input_length for input_length, _ in selected],
        [output_length for _, output_length in selected],
    )


def deterministic_mixed_prompts(tokenizer, input_lengths, seed):
    return [
        deterministic_prompts(tokenizer, input_length, 1, seed + index)[0]
        for index, input_length in enumerate(input_lengths)
    ]


def request_metrics(record):
    if record.error or not record.token_times:
        return None
    ttft = record.token_times[0] - record.scheduled_arrival
    submitted_at = (
        record.submitted_at
        if record.submitted_at is not None
        else record.scheduled_arrival
    )
    queue_delay = max(0.0, submitted_at - record.scheduled_arrival)
    engine_ttft = record.token_times[0] - submitted_at
    e2e = record.token_times[-1] - record.scheduled_arrival
    if len(record.token_times) > 1:
        intervals = [
            current - previous
            for previous, current in zip(record.token_times, record.token_times[1:])
        ]
        tpot = (record.token_times[-1] - record.token_times[0]) / len(intervals)
    else:
        intervals = []
        tpot = 0.0
    return {
        "ttft": ttft,
        "queue_delay": queue_delay,
        "engine_ttft": engine_ttft,
        "e2e": e2e,
        "tpot": tpot,
        "itls": intervals,
    }


def summarize_scenario(
    engine,
    request_rate,
    records,
    duration,
    output_length,
    telemetry,
    ttft_slo_ms,
    tpot_slo_ms,
    e2e_slo_ms,
):
    completed = []
    failed = []
    for record in records:
        metrics = request_metrics(record)
        expected_output_length = record.requested_output_length or output_length
        if (
            metrics is None
            or len(record.token_times) != expected_output_length
            or len(record.token_ids) != expected_output_length
        ):
            failed.append(
                {
                    "request_index": record.request_index,
                    "generated_tokens": len(record.token_times),
                    "error": record.error,
                }
            )
        else:
            completed.append(metrics)

    ttfts = [metrics["ttft"] for metrics in completed]
    queue_delays = [metrics["queue_delay"] for metrics in completed]
    engine_ttfts = [metrics["engine_ttft"] for metrics in completed]
    e2es = [metrics["e2e"] for metrics in completed]
    tpots = [metrics["tpot"] for metrics in completed]
    itls = [interval for metrics in completed for interval in metrics["itls"]]

    def meets_slo(metrics):
        checks = []
        if ttft_slo_ms is not None:
            checks.append(metrics["ttft"] * 1000 <= ttft_slo_ms)
        if tpot_slo_ms is not None:
            checks.append(metrics["tpot"] * 1000 <= tpot_slo_ms)
        if e2e_slo_ms is not None:
            checks.append(metrics["e2e"] * 1000 <= e2e_slo_ms)
        return all(checks)

    has_slo = any(
        value is not None
        for value in (ttft_slo_ms, tpot_slo_ms, e2e_slo_ms)
    )
    good_requests = (
        sum(meets_slo(metrics) for metrics in completed) if has_slo else None
    )
    generated_tokens = sum(len(record.token_times) for record in records)
    scheduled_arrivals = sorted(record.scheduled_arrival for record in records)
    arrival_intervals = [
        current - previous
        for previous, current in zip(scheduled_arrivals, scheduled_arrivals[1:])
    ]
    arrival_span = (
        scheduled_arrivals[-1] - scheduled_arrivals[0]
        if len(scheduled_arrivals) > 1
        else 0.0
    )
    realized_arrival_rate = (
        (len(scheduled_arrivals) - 1) / arrival_span if arrival_span > 0 else None
    )
    outstanding_events = []
    for record in records:
        outstanding_events.append((record.scheduled_arrival, 1))
        if record.token_times:
            outstanding_events.append((record.token_times[-1], -1))
    current_outstanding = 0
    peak_outstanding = 0
    for _, delta in sorted(outstanding_events, key=lambda event: (event[0], event[1])):
        current_outstanding += delta
        peak_outstanding = max(peak_outstanding, current_outstanding)
    if telemetry and telemetry.get("power_draw_watts"):
        mean_power = telemetry["power_draw_watts"]["mean"]
        estimated_energy = mean_power * duration
        telemetry["estimated_energy_joules"] = estimated_energy
        telemetry["estimated_joules_per_output_token"] = (
            estimated_energy / generated_tokens if generated_tokens else None
        )
        telemetry["estimated_joules_per_successful_request"] = (
            estimated_energy / len(completed) if completed else None
        )
    raw_requests = []
    for record in records:
        metrics = request_metrics(record)
        raw_requests.append(
            {
                "request_index": record.request_index,
                "scheduled_arrival_seconds": record.scheduled_arrival,
                "submitted_at_seconds": record.submitted_at,
                "input_tokens": record.requested_input_length,
                "requested_output_tokens": record.requested_output_length,
                "generated_tokens": len(record.token_times),
                "output_token_ids": record.token_ids,
                "ttft_seconds": metrics["ttft"] if metrics else None,
                "queue_delay_seconds": (
                    metrics["queue_delay"] if metrics else None
                ),
                "engine_ttft_seconds": (
                    metrics["engine_ttft"] if metrics else None
                ),
                "tpot_seconds": metrics["tpot"] if metrics else None,
                "e2e_seconds": metrics["e2e"] if metrics else None,
                "inter_token_seconds": metrics["itls"] if metrics else [],
                "error": record.error,
            }
        )

    return {
        "engine": engine,
        "offered_request_rate": "inf" if math.isinf(request_rate) else request_rate,
        "duration_seconds": duration,
        "realized_arrival_rate": realized_arrival_rate,
        "arrival_interval_seconds": summarize(arrival_intervals),
        "peak_outstanding_requests": peak_outstanding,
        "successful_requests": len(completed),
        "failed_requests": failed,
        "achieved_request_throughput": len(completed) / duration,
        "goodput_requests_per_second": (
            good_requests / duration if good_requests is not None else None
        ),
        "output_token_throughput": generated_tokens / duration,
        "generated_tokens": generated_tokens,
        "ttft_seconds": summarize(ttfts),
        "queue_delay_seconds": summarize(queue_delays),
        "engine_ttft_seconds": summarize(engine_ttfts),
        "tpot_seconds": summarize(tpots),
        "itl_seconds": summarize(itls),
        "e2e_seconds": summarize(e2es),
        "gpu_telemetry": telemetry,
        "raw_requests": raw_requests,
    }


def run_pagedserve_scenario(
    scheduler,
    prompts,
    output_lengths,
    request_rate,
    args,
):
    offsets = arrival_offsets(
        request_rate,
        len(prompts),
        pattern=args.arrival_pattern,
        seed=args.seed,
        duration_seconds=args.duration_seconds,
    )
    records = [
        RequestRecord(
            index,
            offset,
            requested_output_length=output_lengths[index],
            requested_input_length=len(prompts[index]),
        )
        for index, offset in enumerate(offsets)
    ]
    scheduler_ids = {}
    next_request = 0
    monitor = create_monitor(args)
    memory_profiler = TorchCUDAMemoryProfiler()
    scheduler.kv_manager.reset_usage_stats()
    synchronize_cuda()
    memory_profiler.start()
    monitor.start()
    benchmark_start = time.perf_counter()

    while next_request < len(offsets) or scheduler.waiting or scheduler.active:
        elapsed = time.perf_counter() - benchmark_start
        while next_request < len(offsets) and offsets[next_request] <= elapsed:
            scheduler_id = scheduler.add_token_request(
                prompts[next_request],
                max_new_tokens=output_lengths[next_request],
            )
            records[next_request].submitted_at = elapsed
            scheduler_ids[scheduler_id] = next_request
            next_request += 1

        if scheduler.waiting or scheduler.active:
            emitted = scheduler.step()
            synchronize_cuda()
            token_time = time.perf_counter() - benchmark_start
            for scheduler_id, token_id in emitted.items():
                record = records[scheduler_ids[scheduler_id]]
                record.token_times.append(token_time)
                record.token_ids.append(token_id)
        elif next_request < len(offsets):
            sleep_for = offsets[next_request] - (time.perf_counter() - benchmark_start)
            if sleep_for > 0:
                time.sleep(min(sleep_for, 0.001))

    synchronize_cuda()
    duration = time.perf_counter() - benchmark_start
    telemetry = monitor.stop() or {}
    telemetry["torch_cuda_allocator"] = memory_profiler.stop()
    telemetry["paged_kv_cache"] = scheduler.kv_manager.usage_summary()
    return records, duration, telemetry


def hf_generate_one(model, prompt, output_length, record, benchmark_start):
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True)
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        past_key_values = output.past_key_values
        synchronize_cuda()
        record.token_times.append(time.perf_counter() - benchmark_start)
        record.token_ids.append(int(next_token.item()))

        for _ in range(output_length - 1):
            output = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            past_key_values = output.past_key_values
            synchronize_cuda()
            record.token_times.append(time.perf_counter() - benchmark_start)
            record.token_ids.append(int(next_token.item()))


def run_hf_scenario(model, prompts, output_lengths, request_rate, args):
    offsets = arrival_offsets(
        request_rate,
        len(prompts),
        pattern=args.arrival_pattern,
        seed=args.seed,
        duration_seconds=args.duration_seconds,
    )
    records = [
        RequestRecord(
            index,
            offset,
            requested_output_length=output_lengths[index],
            requested_input_length=len(prompts[index]),
        )
        for index, offset in enumerate(offsets)
    ]
    monitor = create_monitor(args)
    memory_profiler = TorchCUDAMemoryProfiler()
    synchronize_cuda()
    memory_profiler.start()
    monitor.start()
    benchmark_start = time.perf_counter()

    for prompt, output_length, record in zip(prompts, output_lengths, records):
        wait_for = record.scheduled_arrival - (time.perf_counter() - benchmark_start)
        if wait_for > 0:
            time.sleep(wait_for)
        try:
            record.submitted_at = time.perf_counter() - benchmark_start
            hf_generate_one(model, prompt, output_length, record, benchmark_start)
        except Exception as error:
            record.error = f"{type(error).__name__}: {error}"

    synchronize_cuda()
    duration = time.perf_counter() - benchmark_start
    telemetry = monitor.stop() or {}
    telemetry["torch_cuda_allocator"] = memory_profiler.stop()
    return records, duration, telemetry


async def consume_vllm_request(
    engine,
    sampling_params,
    prompt,
    record,
    benchmark_start,
):
    wait_for = record.scheduled_arrival - (time.perf_counter() - benchmark_start)
    if wait_for > 0:
        await asyncio.sleep(wait_for)
    try:
        record.submitted_at = time.perf_counter() - benchmark_start
        async for output in engine.generate(
            request_id=f"benchmark-{record.request_index}-{benchmark_start}",
            prompt={"prompt_token_ids": prompt},
            sampling_params=sampling_params,
        ):
            if not output.outputs:
                continue
            new_token_ids = output.outputs[0].token_ids
            if new_token_ids:
                token_time = time.perf_counter() - benchmark_start
                record.token_times.extend([token_time] * len(new_token_ids))
                record.token_ids.extend(int(token_id) for token_id in new_token_ids)
    except Exception as error:
        record.error = f"{type(error).__name__}: {error}"


async def run_vllm_scenario(
    engine,
    prompts,
    output_lengths,
    request_rate,
    args,
):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    offsets = arrival_offsets(
        request_rate,
        len(prompts),
        pattern=args.arrival_pattern,
        seed=args.seed,
        duration_seconds=args.duration_seconds,
    )
    records = [
        RequestRecord(
            index,
            offset,
            requested_output_length=output_lengths[index],
            requested_input_length=len(prompts[index]),
        )
        for index, offset in enumerate(offsets)
    ]
    monitor = create_monitor(args)
    memory_profiler = TorchCUDAMemoryProfiler()
    synchronize_cuda()
    memory_profiler.start()
    monitor.start()
    benchmark_start = time.perf_counter()
    await asyncio.gather(
        *[
            consume_vllm_request(
                engine,
                SamplingParams(
                    temperature=0.0,
                    max_tokens=output_length,
                    ignore_eos=True,
                    output_kind=RequestOutputKind.DELTA,
                ),
                prompt,
                record,
                benchmark_start,
            )
            for prompt, output_length, record in zip(
                prompts,
                output_lengths,
                records,
            )
        ]
    )
    synchronize_cuda()
    duration = time.perf_counter() - benchmark_start
    telemetry = monitor.stop() or {}
    telemetry["torch_cuda_allocator"] = memory_profiler.stop()
    return records, duration, telemetry


def warmup_pagedserve(scheduler, prompt, output_length):
    request_id = scheduler.add_token_request(
        prompt,
        max_new_tokens=min(output_length, 8),
    )
    while request_id not in scheduler.finished:
        scheduler.step()
    synchronize_cuda()


def warmup_hf(model, prompt, output_length):
    record = RequestRecord(0, 0.0)
    hf_generate_one(model, prompt, min(output_length, 8), record, time.perf_counter())
    synchronize_cuda()


async def warmup_vllm(engine, prompt, output_length):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    params = SamplingParams(
        temperature=0.0,
        max_tokens=min(output_length, 8),
        ignore_eos=True,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )
    async for _ in engine.generate(
        request_id="benchmark-warmup",
        prompt={"prompt_token_ids": prompt},
        sampling_params=params,
    ):
        pass
    synchronize_cuda()


def system_metadata():
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
        git_dirty = None
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        metadata.update(
            {
                "device": "cuda",
                "cuda_device": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "gpu_memory_mb": properties.total_memory / (1024 * 1024),
            }
        )
    else:
        metadata.update(
            {
                "device": "cpu",
                "cuda_device": None,
                "compute_capability": None,
                "gpu_memory_mb": None,
            }
        )
    return metadata


def parse_request_rate(value):
    if value.lower() in ("inf", "infinity", "burst"):
        return math.inf
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("request rate must be positive or 'inf'")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare HF, PagedServe, and vLLM with one common load generator"
    )
    parser.add_argument("--engine", choices=SUPPORTED_ENGINES, required=True)
    parser.add_argument("--model-id", default="openai-community/gpt2")
    parser.add_argument("--dtype", choices=SUPPORTED_DTYPES, default="float32")
    parser.add_argument("--input-length", type=positive_integer, required=True)
    parser.add_argument("--output-length", type=positive_integer, required=True)
    parser.add_argument("--num-requests", type=positive_integer, default=100)
    parser.add_argument(
        "--duration-seconds",
        type=float,
        help="generate arrivals for this duration; overrides --num-requests",
    )
    parser.add_argument(
        "--arrival-pattern",
        choices=("fixed", "poisson"),
        default="fixed",
        help="fixed intervals for controlled tests or Poisson arrivals for production-like load",
    )
    parser.add_argument(
        "--request-shape",
        type=parse_request_shape,
        action="append",
        help="repeat INPUT_TOKENS:OUTPUT_TOKENS:WEIGHT for a mixed workload",
    )
    parser.add_argument(
        "--request-rate",
        type=parse_request_rate,
        action="append",
        required=True,
        help="offered requests/second; repeat for a sweep or use inf for a burst",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-model-len", type=positive_integer, default=1024)
    parser.add_argument("--max-batch-size", type=positive_integer, default=64)
    parser.add_argument(
        "--kv-cache-memory-mb",
        type=positive_integer,
        help="explicit PagedServe KV budget; omit for safe automatic CUDA sizing",
    )
    parser.add_argument(
        "--kv-cache-memory-utilization",
        type=float,
        default=DEFAULT_CUDA_KV_MEMORY_UTILIZATION,
    )
    parser.add_argument(
        "--kv-cache-safety-mb",
        type=positive_integer,
        default=DEFAULT_CUDA_KV_SAFETY_MB,
    )
    parser.add_argument("--prefill-chunk-size", type=positive_integer, default=128)
    parser.add_argument(
        "--pagedserve-strategy",
        choices=("orca", "sarathi"),
        default="sarathi",
    )
    parser.add_argument(
        "--decode-attention-backend",
        choices=SUPPORTED_DECODE_ATTENTION_BACKENDS,
        default="torch",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        choices=SUPPORTED_KV_CACHE_DTYPES,
        default="model",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--telemetry-interval-ms", type=positive_integer, default=200)
    parser.add_argument("--ttft-slo-ms", type=float)
    parser.add_argument("--tpot-slo-ms", type=float)
    parser.add_argument("--e2e-slo-ms", type=float)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args()


def validate_args(args):
    if args.engine == "vllm" and not torch.cuda.is_available():
        raise RuntimeError("The vLLM comparison requires a supported GPU environment")
    if not torch.cuda.is_available() and args.dtype != "float32":
        raise ValueError("CPU comparison runs must use float32")
    if args.decode_attention_backend == "triton":
        if args.engine != "pagedserve":
            raise ValueError("Triton decode attention applies only to PagedServe")
        if not torch.cuda.is_available():
            raise ValueError("Triton decode attention requires CUDA")
    shapes = args.request_shape or [(args.input_length, args.output_length, 1.0)]
    if any(input_length + output_length > args.max_model_len for input_length, output_length, _ in shapes):
        raise ValueError("a request shape exceeds max_model_len")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    if not 0 < args.kv_cache_memory_utilization <= 1:
        raise ValueError("kv_cache_memory_utilization must be in (0, 1]")
    if args.duration_seconds is not None:
        if args.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if any(math.isinf(rate) for rate in args.request_rate):
            raise ValueError("duration_seconds cannot be combined with burst traffic")


def base_report(args, input_lengths, output_lengths):
    return {
        "profile_schema_version": 3,
        "system": system_metadata(),
        "gpu_lifecycle": {
            "before_engine_initialization": nvidia_smi_snapshot(),
        },
        "settings": {
            "engine": args.engine,
            "model_id": args.model_id,
            "dtype": args.dtype,
            "input_length": args.input_length,
            "output_length": args.output_length,
            "total_sequence_length": args.input_length + args.output_length,
            "actual_total_sequence_length_tokens": summarize(
                [
                    input_length + output_length
                    for input_length, output_length in zip(
                        input_lengths,
                        output_lengths,
                    )
                ]
            ),
            "num_requests": len(input_lengths),
            "configured_num_requests": args.num_requests,
            "request_count_mode": (
                "duration" if args.duration_seconds is not None else "fixed_count"
            ),
            "arrival_duration_seconds": args.duration_seconds,
            "total_prompt_tokens": sum(input_lengths),
            "requested_output_tokens": sum(output_lengths),
            "actual_input_length_tokens": summarize(input_lengths),
            "actual_output_length_tokens": summarize(output_lengths),
            "request_shapes": [
                {
                    "input_tokens": input_length,
                    "output_tokens": output_length,
                    "weight": weight,
                }
                for input_length, output_length, weight in (
                    args.request_shape
                    or [(args.input_length, args.output_length, 1.0)]
                )
            ],
            "request_rates": [
                "inf" if math.isinf(rate) else rate for rate in args.request_rate
            ],
            "arrival_pattern": f"{args.arrival_pattern}_open_loop",
            "seed": args.seed,
            "max_model_len": args.max_model_len,
            "max_batch_size": args.max_batch_size,
            "kv_cache_memory_mb": args.kv_cache_memory_mb,
            "kv_cache_memory_utilization": args.kv_cache_memory_utilization,
            "kv_cache_safety_mb": args.kv_cache_safety_mb,
            "prefill_chunk_size": args.prefill_chunk_size,
            "pagedserve_strategy": args.pagedserve_strategy,
            "decode_attention_backend": args.decode_attention_backend,
            "kv_cache_dtype": args.kv_cache_dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "vllm_enforce_eager": args.vllm_enforce_eager,
            "ttft_slo_ms": args.ttft_slo_ms,
            "tpot_slo_ms": args.tpot_slo_ms,
            "e2e_slo_ms": args.e2e_slo_ms,
        },
        "results": [],
    }


def add_summary(report, args, request_rate, records, duration, telemetry):
    report["results"].append(
        summarize_scenario(
            engine=args.engine,
            request_rate=request_rate,
            records=records,
            duration=duration,
            output_length=args.output_length,
            telemetry=telemetry,
            ttft_slo_ms=args.ttft_slo_ms,
            tpot_slo_ms=args.tpot_slo_ms,
            e2e_slo_ms=args.e2e_slo_ms,
        )
    )


def run_pagedserve(args, tokenizer, prompts, output_lengths, report):
    initialization_start = time.perf_counter()
    scheduler = build_scheduler(
        GPT2_MODEL,
        gpt2_model_id=args.model_id,
        scheduling_strategy=args.pagedserve_strategy,
        prefill_chunk_size=args.prefill_chunk_size,
        max_batch_size=args.max_batch_size,
        kv_cache_memory_mb=args.kv_cache_memory_mb,
        kv_cache_memory_utilization=args.kv_cache_memory_utilization,
        kv_cache_safety_mb=args.kv_cache_safety_mb,
        execution_dtype=args.dtype,
        decode_attention_backend=args.decode_attention_backend,
        kv_cache_dtype=args.kv_cache_dtype,
    )
    synchronize_cuda()
    initialization_seconds = time.perf_counter() - initialization_start
    report["gpu_lifecycle"]["after_engine_initialization"] = nvidia_smi_snapshot()
    scheduler.eos_token_id = None
    warmup_start = time.perf_counter()
    warmup_pagedserve(scheduler, prompts[0], output_lengths[0])
    warmup_seconds = time.perf_counter() - warmup_start
    report["gpu_lifecycle"]["after_warmup"] = nvidia_smi_snapshot()
    report["engine_metadata"] = {
        "policy": "continuous_batching",
        "decode_attention_backend": args.decode_attention_backend,
        "kv_cache_dtype": args.kv_cache_dtype,
        "model_dtype": str(next(scheduler.model_engine.parameters()).dtype),
        "initialization_seconds": initialization_seconds,
        "warmup_seconds": warmup_seconds,
        "module_memory": module_memory_profile(scheduler.model_engine),
        "model_parameter_bytes": scheduler.kv_manager.model_parameter_bytes,
        "model_parameter_count": scheduler.kv_manager.model_parameter_count,
        "model_buffer_bytes": scheduler.kv_manager.model_buffer_bytes,
        "kv_cache_memory_bytes": scheduler.kv_manager.total_memory,
        "kv_cache_memory_source": scheduler.kv_manager.memory_budget_source,
        "kv_cache_blocks": scheduler.kv_manager.total_available_blocks,
        "kv_cache_block_size_tokens": scheduler.kv_manager.block_size,
        "kv_cache_bytes_per_token": (
            scheduler.kv_manager.bytes_per_block
            // scheduler.kv_manager.block_size
        ),
        "kv_cache_capacity_tokens": (
            scheduler.kv_manager.total_available_blocks
            * scheduler.kv_manager.block_size
        ),
        "kv_cache_capacity_max_length_requests": (
            scheduler.kv_manager.total_available_blocks
            // math.ceil(args.max_model_len / scheduler.kv_manager.block_size)
        ),
        "cuda_memory_snapshot": scheduler.kv_manager.cuda_memory_snapshot,
        "cuda_allocator_snapshots": (
            scheduler.kv_manager.cuda_allocator_snapshots
        ),
    }
    for request_rate in args.request_rate:
        records, duration, telemetry = run_pagedserve_scenario(
            scheduler,
            prompts,
            output_lengths,
            request_rate,
            args,
        )
        add_summary(report, args, request_rate, records, duration, telemetry)


def run_hf(args, tokenizer, model_config, prompts, output_lengths, report):
    import transformers

    initialization_start = time.perf_counter()
    model = causal_lm_from_pretrained(
        args.model_id,
        model_config,
        torch_dtype(args.dtype),
    ).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    synchronize_cuda()
    initialization_seconds = time.perf_counter() - initialization_start
    report["gpu_lifecycle"]["after_engine_initialization"] = nvidia_smi_snapshot()
    warmup_start = time.perf_counter()
    warmup_hf(model, prompts[0], output_lengths[0])
    warmup_seconds = time.perf_counter() - warmup_start
    report["gpu_lifecycle"]["after_warmup"] = nvidia_smi_snapshot()
    report["engine_metadata"] = {
        "policy": "sequential_fcfs_no_continuous_batching",
        "transformers_version": transformers.__version__,
        "model_dtype": str(next(model.parameters()).dtype),
        "initialization_seconds": initialization_seconds,
        "warmup_seconds": warmup_seconds,
        "module_memory": module_memory_profile(model),
    }
    for request_rate in args.request_rate:
        records, duration, telemetry = run_hf_scenario(
            model,
            prompts,
            output_lengths,
            request_rate,
            args,
        )
        add_summary(report, args, request_rate, records, duration, telemetry)


async def run_vllm(args, prompts, output_lengths, report):
    import vllm
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    initialization_start = time.perf_counter()
    engine_args = AsyncEngineArgs(
        model=args.model_id,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.vllm_enforce_eager,
        enable_prefix_caching=False,
        disable_log_stats=True,
        seed=args.seed,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    initialization_seconds = time.perf_counter() - initialization_start
    report["gpu_lifecycle"]["after_engine_initialization"] = nvidia_smi_snapshot()
    cache_config = getattr(getattr(engine, "vllm_config", None), "cache_config", None)
    cache_dtype = getattr(cache_config, "cache_dtype", None)
    report["engine_metadata"] = {
        "policy": "vllm_async_continuous_batching",
        "vllm_version": vllm.__version__,
        "initialization_seconds": initialization_seconds,
        "vllm_cache_config": {
            "block_size_tokens": getattr(cache_config, "block_size", None),
            "kv_cache_memory_bytes": getattr(
                cache_config,
                "kv_cache_memory_bytes",
                None,
            ),
            "kv_cache_size_tokens": getattr(
                cache_config,
                "kv_cache_size_tokens",
                None,
            ),
            "kv_cache_max_concurrency": getattr(
                cache_config,
                "kv_cache_max_concurrency",
                None,
            ),
            "num_gpu_blocks": getattr(cache_config, "num_gpu_blocks", None),
            "kv_cache_dtype": str(cache_dtype) if cache_dtype is not None else None,
        },
    }
    try:
        warmup_start = time.perf_counter()
        await warmup_vllm(engine, prompts[0], output_lengths[0])
        report["engine_metadata"]["warmup_seconds"] = (
            time.perf_counter() - warmup_start
        )
        report["gpu_lifecycle"]["after_warmup"] = nvidia_smi_snapshot()
        for request_rate in args.request_rate:
            records, duration, telemetry = await run_vllm_scenario(
                engine,
                prompts,
                output_lengths,
                request_rate,
                args,
            )
            add_summary(report, args, request_rate, records, duration, telemetry)
    finally:
        engine.shutdown()


def print_report(report):
    system = report["system"]
    print(
        f"Platform: {system['platform']} | PyTorch {system['pytorch_version']} | "
        f"CUDA {system['cuda_runtime_version']} | device {system['cuda_device']}"
    )
    print(
        f"Model: {report['settings']['model_id']} | "
        f"dtype {report['settings']['dtype']} | "
        f"vLLM {report['engine_metadata'].get('vllm_version', 'n/a')}"
    )
    model_metadata = report["model_metadata"]
    model_config = model_metadata["config"]
    parameter_count = model_metadata["theoretical_parameter_count"]
    print(
        f"Architecture: {model_config['model_type']} | "
        f"parameters {parameter_count if parameter_count is not None else 'n/a'} | "
        f"runtime weight bytes "
        f"{model_metadata['theoretical_runtime_weight_bytes']} | "
        f"layers/heads/KV-heads "
        f"{model_config['num_layers']}/{model_config['num_attention_heads']}/"
        f"{model_config['num_key_value_heads']} | hidden "
        f"{model_config['hidden_size']} | context "
        f"{model_config['max_position_embeddings']}"
    )
    engine_metadata = report["engine_metadata"]
    print(
        f"Engine initialization: {engine_metadata['initialization_seconds']:.3f} s | "
        f"warmup: {engine_metadata.get('warmup_seconds', float('nan')):.3f} s"
    )
    if "kv_cache_memory_bytes" in engine_metadata:
        print(
            f"PagedServe memory: model parameters "
            f"{engine_metadata['model_parameter_bytes'] / (1024 ** 2):.1f} MiB | "
            f"KV cache {engine_metadata['kv_cache_memory_bytes'] / (1024 ** 2):.1f} MiB "
            f"({engine_metadata['kv_cache_memory_source']}) | "
            f"KV capacity {engine_metadata['kv_cache_capacity_tokens']:,} tokens / "
            f"{engine_metadata['kv_cache_capacity_max_length_requests']:,} "
            f"max-length requests"
        )
    print(
        "engine | offered RPS | achieved RPS | goodput RPS | output tok/s | "
        "TTFT p50/p95/p99 (ms) | TPOT p50/p95/p99 (ms) | "
        "E2E p50/p95/p99 (ms) | failures"
    )
    print("-" * 145)

    def latency_triplet(summary):
        if summary is None:
            return "n/a"
        return (
            f"{summary['median'] * 1000:.2f}/"
            f"{summary['p95'] * 1000:.2f}/"
            f"{summary['p99'] * 1000:.2f}"
        )

    for result in report["results"]:
        ttft = result["ttft_seconds"]
        tpot = result["tpot_seconds"]
        e2e = result["e2e_seconds"]
        goodput = result["goodput_requests_per_second"]
        goodput_text = "n/a" if goodput is None else f"{goodput:.3f}"
        print(
            f"{result['engine']} | {result['offered_request_rate']} | "
            f"{result['achieved_request_throughput']:.3f} | "
            f"{goodput_text} | "
            f"{result['output_token_throughput']:.2f} | "
            f"{latency_triplet(ttft)} | "
            f"{latency_triplet(tpot)} | "
            f"{latency_triplet(e2e)} | "
            f"{len(result['failed_requests'])}"
        )
        print(
            f"  queue delay p50/p95/p99: "
            f"{latency_triplet(result['queue_delay_seconds'])} ms | "
            f"engine TTFT after submit: "
            f"{latency_triplet(result['engine_ttft_seconds'])} ms"
        )
        realized_rate = result["realized_arrival_rate"]
        realized_text = "burst" if realized_rate is None else f"{realized_rate:.3f}"
        print(
            f"  realized arrivals: {realized_text} RPS | "
            f"peak outstanding requests: {result['peak_outstanding_requests']}"
        )

    for result in report["results"]:
        telemetry = result["gpu_telemetry"]
        if not telemetry or not telemetry.get("sample_count"):
            continue
        utilization = telemetry["gpu_kernel_active_percent"]
        memory_activity = telemetry["memory_active_percent"]
        used_memory = telemetry["gpu_memory_used_mb"]
        total_memory = telemetry["gpu_memory_total_mb"]
        print(
            f"GPU telemetry ({result['offered_request_rate']} RPS, "
            f"{telemetry['sample_count']} samples): "
            f"kernel mean/p95/max={utilization['mean']:.1f}/"
            f"{utilization['p95']:.1f}/{utilization['maximum']:.1f}% | "
            f"memory activity mean/p95/max={memory_activity['mean']:.1f}/"
            f"{memory_activity['p95']:.1f}/{memory_activity['maximum']:.1f}% | "
            f"VRAM mean/max={used_memory['mean']:.0f}/"
            f"{used_memory['maximum']:.0f} MiB of {total_memory:.0f} MiB | "
            f"power mean/max={telemetry['power_draw_watts']['mean']:.1f}/"
            f"{telemetry['power_draw_watts']['maximum']:.1f} W | "
            f"energy~{telemetry.get('estimated_energy_joules', float('nan')):.1f} J"
        )
        kv_usage = telemetry.get("paged_kv_cache")
        if kv_usage:
            print(
                f"  Paged KV peak: {kv_usage['peak_allocated_blocks']:,}/"
                f"{kv_usage['total_blocks']:,} blocks, "
                f"{kv_usage['peak_allocated_tokens']:,} tokens, "
                f"{kv_usage['peak_allocated_bytes'] / (1024 ** 2):.1f} MiB, "
                f"{kv_usage['peak_utilization_percent']:.2f}%"
            )
        allocator = telemetry.get("torch_cuda_allocator")
        if allocator:
            print(
                f"  Torch CUDA peak allocated/reserved: "
                f"{allocator['peak_allocated_bytes'] / (1024 ** 2):.1f}/"
                f"{allocator['peak_reserved_bytes'] / (1024 ** 2):.1f} MiB"
            )


def main():
    args = parse_args()
    validate_args(args)

    from transformers import AutoConfig, AutoTokenizer

    tokenizer_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer_load_seconds = time.perf_counter() - tokenizer_start
    config_start = time.perf_counter()
    model_config = AutoConfig.from_pretrained(args.model_id)
    config_load_seconds = time.perf_counter() - config_start
    theoretical_parameter_count = None
    parameter_count_error = None
    try:
        with torch.device("meta"):
            meta_model = causal_lm_from_config(model_config)
        theoretical_parameter_count = sum(
            parameter.numel() for parameter in meta_model.parameters()
        )
        del meta_model
    except Exception as error:
        parameter_count_error = f"{type(error).__name__}: {error}"
    if args.duration_seconds is None:
        generated_request_count = args.num_requests
    else:
        generated_request_count = max(
            len(
                arrival_offsets(
                    request_rate,
                    args.num_requests,
                    pattern=args.arrival_pattern,
                    seed=args.seed,
                    duration_seconds=args.duration_seconds,
                )
            )
            for request_rate in args.request_rate
        )
    input_lengths, output_lengths = request_shapes(
        args, num_requests=generated_request_count
    )
    prompts = deterministic_mixed_prompts(tokenizer, input_lengths, args.seed)
    report = base_report(args, input_lengths, output_lengths)
    prompt_digest = hashlib.sha256(
        json.dumps(prompts, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    report["model_metadata"] = {
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_commit": tokenizer.init_kwargs.get("_commit_hash"),
        "model_config_commit": getattr(model_config, "_commit_hash", None),
        "vocab_size": len(tokenizer),
        "prompt_token_ids_sha256": prompt_digest,
        "tokenizer_load_seconds": tokenizer_load_seconds,
        "config_load_seconds": config_load_seconds,
        "config": config_profile(model_config),
        "theoretical_parameter_count": theoretical_parameter_count,
        "theoretical_runtime_weight_bytes": (
            theoretical_parameter_count
            * torch.empty(0, dtype=torch_dtype(args.dtype)).element_size()
            if theoretical_parameter_count is not None
            else None
        ),
        "parameter_count_error": parameter_count_error,
    }
    if args.engine == "pagedserve":
        run_pagedserve(args, tokenizer, prompts, output_lengths, report)
    elif args.engine == "hf":
        run_hf(
            args,
            tokenizer,
            model_config,
            prompts,
            output_lengths,
            report,
        )
    else:
        asyncio.run(run_vllm(args, prompts, output_lengths, report))

    report["model_metadata"]["huggingface_cache_after_run"] = (
        huggingface_cache_profile(args.model_id)
    )
    print_report(report)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Raw results written to {args.json_output}")


if __name__ == "__main__":
    main()
