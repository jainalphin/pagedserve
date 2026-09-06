import pytest
import torch

from main import REFERENCE_MODEL, build_scheduler, calculate_cuda_kv_cache_budget
from src.model.kv_manager import KVCacheManager


def test_cuda_kv_budget_respects_utilization_target():
    assert calculate_cuda_kv_cache_budget(
        free_memory=800,
        total_memory=1000,
        memory_utilization=0.8,
        safety_memory=100,
    ) == 600


def test_cuda_kv_budget_respects_activation_safety_margin():
    assert calculate_cuda_kv_cache_budget(
        free_memory=800,
        total_memory=1000,
        memory_utilization=0.9,
        safety_memory=150,
    ) == 650


@pytest.mark.parametrize(
    "kwargs",
    (
        {"free_memory": 800, "total_memory": 1000, "memory_utilization": 0},
        {"free_memory": 800, "total_memory": 1000, "memory_utilization": 1.1},
        {"free_memory": 0, "total_memory": 1000},
        {"free_memory": 1100, "total_memory": 1000},
        {"free_memory": 800, "total_memory": 1000, "safety_memory": -1},
    ),
)
def test_cuda_kv_budget_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        calculate_cuda_kv_cache_budget(**kwargs)


def test_cuda_kv_budget_rejects_exhausted_memory():
    with pytest.raises(RuntimeError):
        calculate_cuda_kv_cache_budget(
            free_memory=100,
            total_memory=1000,
            safety_memory=200,
        )


def test_kv_cache_usage_summary_tracks_peak_and_capacity():
    bytes_per_block = 2 * 1 * 1 * 2 * 2 * 4
    manager = KVCacheManager(
        block_size=2,
        total_memory=bytes_per_block * 4,
        tensor_dtype=torch.float32,
        device="cpu",
        num_layers=1,
        num_kv_heads=1,
        head_dim=2,
    )
    manager.reset_usage_stats()
    first = manager.allocate_block()
    second = manager.allocate_block()
    manager.free_block(first)
    manager.free_block(second)

    summary = manager.usage_summary()
    assert summary["total_blocks"] == 4
    assert summary["free_blocks_at_end"] == 4
    assert summary["peak_allocated_blocks"] == 2
    assert summary["peak_allocated_tokens"] == 4
    assert summary["peak_allocated_bytes"] == bytes_per_block * 2
    assert summary["peak_utilization_percent"] == 50


def test_scheduler_records_memory_profile_metadata():
    scheduler = build_scheduler(REFERENCE_MODEL)
    manager = scheduler.kv_manager
    assert manager.model_parameter_count > 0
    assert manager.model_parameter_bytes > 0
    assert manager.model_buffer_bytes >= 0
    snapshots = manager.cuda_allocator_snapshots
    assert set(snapshots) == {
        "before_model",
        "after_model",
        "after_kv_cache",
    }
    if torch.cuda.is_available():
        for snapshot in snapshots.values():
            assert snapshot["total_bytes"] > 0
            assert snapshot["free_bytes"] >= 0
            assert snapshot["device_used_bytes"] >= 0
            assert snapshot["torch_allocated_bytes"] >= 0
            assert snapshot["torch_reserved_bytes"] >= 0
    else:
        assert all(snapshot is None for snapshot in snapshots.values())
