import math
from types import SimpleNamespace

from dual_gpu_capacity_benchmark import build_worker_command, combine_rate


def raw_request(index, input_tokens=128, output_tokens=2):
    return {
        "request_index": index,
        "input_tokens": input_tokens,
        "requested_output_tokens": output_tokens,
        "generated_tokens": output_tokens,
        "ttft_seconds": 0.1,
        "queue_delay_seconds": 0.01,
        "engine_ttft_seconds": 0.09,
        "tpot_seconds": 0.2,
        "e2e_seconds": 0.3,
        "inter_token_seconds": [0.2] * (output_tokens - 1),
        "error": None,
    }


def worker_result(duration, request_count, token_count, shape=(128, 2)):
    return {
        "duration_seconds": duration,
        "successful_requests": request_count,
        "slo_good_requests": request_count,
        "generated_tokens": token_count,
        "achieved_request_throughput": request_count / duration,
        "output_token_throughput": token_count / duration,
        "failed_requests": [],
        "realized_arrival_rate": 10.0,
        "peak_outstanding_requests": 3,
        "gpu_telemetry": {},
        "raw_requests": [
            raw_request(index, shape[0], shape[1])
            for index in range(request_count)
        ],
    }


def test_combined_throughput_uses_slowest_replica_makespan():
    first = worker_result(duration=10.0, request_count=10, token_count=20)
    second = worker_result(duration=12.0, request_count=10, token_count=20)
    combined = combine_rate([first, second], total_offered_rate=2.0)

    assert combined["aggregate_measurement_duration_seconds"] == 12.0
    assert math.isclose(combined["achieved_request_throughput"], 20 / 12)
    assert math.isclose(combined["output_token_throughput"], 40 / 12)
    assert math.isclose(
        combined["replica_rate_sum_request_throughput"],
        10 / 10 + 10 / 12,
    )
    assert math.isclose(combined["goodput_requests_per_second"], 20 / 12)


def test_combined_results_keep_request_shapes_separate():
    first = worker_result(10.0, 1, 2, shape=(128, 2))
    second = worker_result(10.0, 1, 3, shape=(900, 3))
    combined = combine_rate([first, second], total_offered_rate=1.0)

    assert [
        (shape["input_tokens"], shape["output_tokens"])
        for shape in combined["latency_by_request_shape"]
    ] == [(128, 2), (900, 3)]


def test_worker_uses_one_shared_memory_target_and_full_batch_warmup():
    args = SimpleNamespace(
        gpu=["0", "1"],
        engine="pagedserve",
        model_id="openai-community/gpt2",
        dtype="float16",
        input_length=128,
        output_length=32,
        num_requests_per_replica=120,
        max_batch_size=64,
        seed=1234,
        arrival_pattern="poisson",
        request_rate=[120.0],
        duration_seconds=600.0,
        request_shape=None,
        pagedserve_strategy="orca",
        decode_attention_backend="triton",
        kv_cache_memory_utilization=None,
        kv_cache_safety_mb=3072,
        kv_cache_memory_mb=None,
        gpu_memory_utilization=0.8,
        ttft_slo_ms=None,
        tpot_slo_ms=None,
        e2e_slo_ms=None,
    )
    command = build_worker_command(args, 0, "worker.json")

    utilization_index = command.index("--kv-cache-memory-utilization")
    warmup_index = command.index("--warmup-batch-size")
    assert command[utilization_index + 1] == "0.8"
    assert command[warmup_index + 1] == "64"

