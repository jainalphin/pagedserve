import asyncio
import math
import time
from types import SimpleNamespace

import pytest
import torch

from comparison_benchmark import (
    RequestRecord,
    arrival_offsets,
    causal_lm_from_config,
    consume_vllm_request,
    create_monitor,
    deterministic_prompts,
    request_metrics,
    request_shapes,
    summarize_scenario,
    warmup_pagedserve,
)


class DummyTokenizer:
    all_special_ids = [0, 1]

    def __len__(self):
        return 32


def test_pagedserve_warmup_uses_requested_batch_shape():
    class FakeScheduler:
        def __init__(self):
            self.added = []
            self.finished = {}

        def add_token_request(self, prompt, max_new_tokens):
            request_id = len(self.added)
            self.added.append((prompt, max_new_tokens))
            return request_id

        def step(self):
            self.finished.update(
                (request_id, object()) for request_id in range(len(self.added))
            )

    scheduler = FakeScheduler()
    warmup_pagedserve(
        scheduler,
        [[1] * 4, [2] * 8, [3] * 16],
        [4, 12, 20],
        batch_size=2,
    )
    assert scheduler.added == [([1] * 4, 4), ([2] * 8, 8)]


def test_gpt2_model_instantiation_bypasses_auto_model_lookup():
    from transformers import GPT2Config
    from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=32,
        n_positions=16,
        n_embd=16,
        n_layer=1,
        n_head=2,
    )
    with torch.device("meta"):
        model = causal_lm_from_config(config)
    assert isinstance(model, GPT2LMHeadModel)


def test_deterministic_prompt_tokens_are_exact_and_repeatable():
    tokenizer = DummyTokenizer()
    first = deterministic_prompts(tokenizer, 9, 4, seed=123)
    second = deterministic_prompts(tokenizer, 9, 4, seed=123)
    assert first == second
    assert all(len(prompt) == 9 for prompt in first)
    assert all(token not in tokenizer.all_special_ids for prompt in first for token in prompt)
    assert len({tuple(prompt) for prompt in first}) == 4


def test_arrival_offsets_support_fixed_rate_and_burst():
    assert arrival_offsets(2.0, 4) == [0.0, 0.5, 1.0, 1.5]
    assert arrival_offsets(math.inf, 4) == [0.0, 0.0, 0.0, 0.0]


def test_poisson_arrivals_are_deterministic_and_not_fixed_interval():
    first = arrival_offsets(10.0, 5, pattern="poisson", seed=7)
    second = arrival_offsets(10.0, 5, pattern="poisson", seed=7)
    assert first == second
    assert first[0] == 0
    assert first == sorted(first)
    assert first != arrival_offsets(10.0, 5, pattern="fixed", seed=7)


def test_duration_driven_arrivals_override_request_count():
    assert arrival_offsets(
        2.0,
        999,
        pattern="fixed",
        duration_seconds=1.1,
    ) == [0.0, 0.5, 1.0]
    poisson = arrival_offsets(
        10.0,
        1,
        pattern="poisson",
        seed=7,
        duration_seconds=1.0,
    )
    assert len(poisson) > 1
    assert poisson == sorted(poisson)
    assert all(0 <= offset < 1.0 for offset in poisson)
    with pytest.raises(ValueError, match="burst"):
        arrival_offsets(math.inf, 1, duration_seconds=1.0)


def test_weighted_request_shapes_are_repeatable_and_valid():
    args = SimpleNamespace(
        request_shape=[(128, 32, 50), (900, 64, 5)],
        input_length=1,
        output_length=1,
        num_requests=100,
        seed=1234,
    )
    first = request_shapes(args)
    second = request_shapes(args)
    assert first == second
    assert len(first[0]) == 100
    assert set(zip(*first)) <= {(128, 32), (900, 64)}


def test_common_request_metrics_and_goodput():
    records = [
        RequestRecord(
            request_index=0,
            scheduled_arrival=0.0,
            token_times=[0.1, 0.2, 0.3],
            token_ids=[4, 5, 6],
        ),
        RequestRecord(
            request_index=1,
            scheduled_arrival=0.5,
            token_times=[0.7, 0.9, 1.1],
            token_ids=[7, 8, 9],
        ),
    ]
    first = request_metrics(records[0])
    assert math.isclose(first["ttft"], 0.1)
    assert first["queue_delay"] == 0
    assert math.isclose(first["engine_ttft"], 0.1)
    assert math.isclose(first["tpot"], 0.1)
    assert math.isclose(first["e2e"], 0.3)

    summary = summarize_scenario(
        engine="test",
        request_rate=2.0,
        records=records,
        duration=1.1,
        output_length=3,
        telemetry=None,
        ttft_slo_ms=150,
        tpot_slo_ms=150,
        e2e_slo_ms=500,
    )
    assert summary["successful_requests"] == 2
    assert not summary["failed_requests"]
    assert math.isclose(summary["goodput_requests_per_second"], 1 / 1.1)
    assert summary["generated_tokens"] == 6

    no_slo_summary = summarize_scenario(
        engine="test",
        request_rate=2.0,
        records=records,
        duration=1.1,
        output_length=3,
        telemetry=None,
        ttft_slo_ms=None,
        tpot_slo_ms=None,
        e2e_slo_ms=None,
    )
    assert no_slo_summary["goodput_requests_per_second"] is None


def test_request_metrics_separate_queue_delay_from_engine_ttft():
    record = RequestRecord(
        request_index=0,
        scheduled_arrival=1.0,
        submitted_at=1.25,
        token_times=[1.35, 1.45],
        token_ids=[4, 5],
    )
    metrics = request_metrics(record)
    assert math.isclose(metrics["queue_delay"], 0.25)
    assert math.isclose(metrics["engine_ttft"], 0.10)
    assert math.isclose(metrics["ttft"], 0.35)


def test_vllm_delta_stream_records_single_token_callbacks():
    class FakeVLLMEngine:
        async def generate(self, **kwargs):
            assert kwargs["prompt"] == {"prompt_token_ids": [10, 11]}
            for token_ids in ([20], [21], [22]):
                yield SimpleNamespace(
                    outputs=[SimpleNamespace(token_ids=token_ids)]
                )

    record = RequestRecord(request_index=0, scheduled_arrival=0.0)
    asyncio.run(
        consume_vllm_request(
            FakeVLLMEngine(),
            sampling_params=object(),
            prompt=[10, 11],
            record=record,
            benchmark_start=time.perf_counter(),
        )
    )
    assert record.error is None
    assert record.token_ids == [20, 21, 22]
    assert len(record.token_times) == 3


def test_vllm_delta_stream_rejects_ambiguous_multi_token_callback():
    class FakeVLLMEngine:
        async def generate(self, **kwargs):
            yield SimpleNamespace(
                outputs=[SimpleNamespace(token_ids=[20, 21])]
            )

    record = RequestRecord(request_index=0, scheduled_arrival=0.0)
    asyncio.run(
        consume_vllm_request(
            FakeVLLMEngine(),
            sampling_params=object(),
            prompt=[10, 11],
            record=record,
            benchmark_start=time.perf_counter(),
        )
    )
    assert "timestamps would be ambiguous" in record.error
    assert not record.token_ids
    assert not record.token_times


def test_summary_reports_each_request_shape_separately():
    records = [
        RequestRecord(
            request_index=0,
            scheduled_arrival=0.0,
            requested_input_length=128,
            requested_output_length=2,
            token_times=[0.1, 0.2],
            token_ids=[1, 2],
        ),
        RequestRecord(
            request_index=1,
            scheduled_arrival=0.0,
            requested_input_length=900,
            requested_output_length=3,
            token_times=[0.3, 0.5, 0.7],
            token_ids=[3, 4, 5],
        ),
    ]
    summary = summarize_scenario(
        engine="test",
        request_rate=1.0,
        records=records,
        duration=1.0,
        output_length=2,
        telemetry=None,
        ttft_slo_ms=None,
        tpot_slo_ms=None,
        e2e_slo_ms=None,
    )
    shapes = summary["latency_by_request_shape"]
    assert [(item["input_tokens"], item["output_tokens"]) for item in shapes] == [
        (128, 2),
        (900, 3),
    ]
    assert [item["request_count"] for item in shapes] == [1, 1]
    assert math.isclose(shapes[0]["e2e_seconds"]["p95"], 0.2)
    assert math.isclose(shapes[1]["tpot_seconds"]["p95"], 0.2)


def test_failed_partial_request_does_not_inflate_successful_token_throughput():
    complete = RequestRecord(
        request_index=0,
        scheduled_arrival=0.0,
        requested_output_length=2,
        token_times=[0.1, 0.2],
        token_ids=[1, 2],
    )
    partial = RequestRecord(
        request_index=1,
        scheduled_arrival=0.0,
        requested_output_length=2,
        token_times=[0.1],
        token_ids=[3],
        error="failed",
    )
    summary = summarize_scenario(
        engine="test",
        request_rate=2.0,
        records=[complete, partial],
        duration=1.0,
        output_length=2,
        telemetry=None,
        ttft_slo_ms=None,
        tpot_slo_ms=None,
        e2e_slo_ms=None,
    )
    assert summary["generated_tokens"] == 2
    assert summary["all_generated_tokens_including_failed_requests"] == 3
    assert summary["output_token_throughput"] == 2


def test_gpu_monitor_targets_only_the_first_cuda_visible_device(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    monkeypatch.setattr("comparison_benchmark.torch.cuda.is_available", lambda: True)
    monitor = create_monitor(SimpleNamespace(telemetry_interval_ms=200))
    assert monitor.gpu_id == "1"


def test_summary_estimates_energy_from_sampled_mean_power():
    record = RequestRecord(
        request_index=0,
        scheduled_arrival=0.0,
        token_times=[0.1, 0.2],
        token_ids=[4, 5],
    )
    telemetry = {"power_draw_watts": {"mean": 50.0}}
    summary = summarize_scenario(
        engine="test",
        request_rate=1.0,
        records=[record],
        duration=2.0,
        output_length=2,
        telemetry=telemetry,
        ttft_slo_ms=None,
        tpot_slo_ms=None,
        e2e_slo_ms=None,
    )
    measured = summary["gpu_telemetry"]
    assert measured["estimated_energy_joules"] == 100.0
    assert measured["estimated_joules_per_output_token"] == 50.0
    assert measured["estimated_joules_per_successful_request"] == 100.0
