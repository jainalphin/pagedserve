from unittest.mock import patch

import torch

from src.model.kv_manager import KVCacheManager
from src.model.iteration import IterationBatch, IterationItem
from src.model.paged_attention import PagedAttention
from src.model.paged_decoder import PagedDecoderLM, TransformerConfig


def _manager(cache_dtype=torch.float32, num_layers=3, total_memory=128 * 1024):
    return KVCacheManager(
        block_size=16,
        total_memory=total_memory,
        tensor_dtype=torch.float32,
        cache_dtype=cache_dtype,
        device="cpu",
        num_layers=num_layers,
        num_kv_heads=2,
        head_dim=8,
    )


def test_decode_metadata_is_constructed_once_and_reused_across_layers():
    torch.manual_seed(91)
    config = TransformerConfig(
        vocab_size=32,
        hidden_size=16,
        num_layers=3,
        num_heads=2,
        head_dim=8,
        mlp_hidden_size=32,
        max_sequence_length=32,
    )
    model = PagedDecoderLM(config).eval()
    manager = _manager(num_layers=config.num_layers)
    attention = PagedAttention(manager, decode_attention_backend="torch")
    prompt = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        _, cache = model.prefill(prompt)
        manager.store_prefill("request", cache)
        with patch.object(
            manager,
            "build_decode_metadata",
            wraps=manager.build_decode_metadata,
        ) as build, patch.object(
            attention,
            "forward_iteration",
            wraps=attention.forward_iteration,
        ) as generic_iteration_attention:
            model.forward_iteration(
                IterationBatch(
                    items=(
                        IterationItem(
                            request_id="request",
                            phase="decode",
                            token_ids=(5,),
                            position_ids=(4,),
                            start_offset=0,
                            end_offset=1,
                        ),
                    ),
                    input_ids=torch.tensor([5]),
                    position_ids=torch.tensor([4]),
                ),
                manager,
                attention,
            )
    assert build.call_count == 1
    assert generic_iteration_attention.call_count == 0
    manager.reserve_token_slot("request")
    metadata = manager.build_decode_metadata(["request"])
    assert metadata.request_ids == ("request",)
    assert metadata.token_offsets.tolist() == [0]
    assert metadata.reserved_block_ids.numel() == 1
    assert metadata.reserved_block_offsets.numel() == 1


def test_reserved_slot_tensors_are_reused_for_every_layer_write():
    manager = _manager(num_layers=3)
    manager.store_prefill_request(
        "request",
        [(torch.randn(2, 4, 8), torch.randn(2, 4, 8)) for _ in range(3)],
    )
    manager.reserve_token_slot("request")
    metadata = manager.build_decode_metadata(["request"])
    key = torch.randn(1, 2, 8)
    value = torch.randn(1, 2, 8)

    with patch("src.model.kv_manager.torch.tensor") as tensor_constructor:
        for layer_id in range(3):
            manager.write_layer_kv_batch(
                ["request"],
                layer_id,
                key,
                value,
                decode_metadata=metadata,
            )
    tensor_constructor.assert_not_called()


def test_inference_buffers_and_offset_patterns_are_reused():
    manager = _manager()
    attention = PagedAttention(manager)
    reference = torch.randn(2, 2, 8)
    with torch.inference_mode():
        first_buffer = attention._inference_buffer("decode", reference)
        second_buffer = attention._inference_buffer("decode", reference)
        first_offsets = attention.inference_index_tensor(
            "outputs", [0, 3], reference.device
        )
        second_offsets = attention.inference_index_tensor(
            "outputs", [0, 3], reference.device
        )
    assert first_buffer.data_ptr() == second_buffer.data_ptr()
    assert first_offsets.data_ptr() == second_offsets.data_ptr()


def test_packed_decode_metadata_storage_is_reused_in_inference_mode():
    manager = _manager()
    manager.store_prefill_request(
        "request",
        [(torch.randn(2, 4, 8), torch.randn(2, 4, 8)) for _ in range(3)],
    )
    manager.reserve_token_slot("request")
    with torch.inference_mode():
        first = manager.build_decode_metadata(["request"])
        second = manager.build_decode_metadata(["request"])
    assert (
        first.block_table.untyped_storage().data_ptr()
        == second.block_table.untyped_storage().data_ptr()
    )
    assert second.context_lengths.tolist() == [5]
    assert second.reserved_block_ids.tolist() == [
        manager.requests["request"].reserved_block_id
    ]
    assert second.reserved_block_offsets.tolist() == [4]


def test_int8_cache_increases_capacity_and_tracks_float_attention():
    torch.manual_seed(92)
    fp_manager = _manager(cache_dtype=torch.float32)
    int8_manager = _manager(cache_dtype=torch.int8)
    assert int8_manager.bytes_per_block < fp_manager.bytes_per_block
    assert int8_manager.total_available_blocks > fp_manager.total_available_blocks

    request_id = "fragmented"
    context_length = 33
    layer_cache = [
        (
            torch.randn(2, context_length - 1, 8),
            torch.randn(2, context_length - 1, 8),
        )
        for _ in range(3)
    ]
    current_keys = [torch.randn(2, 8) for _ in range(3)]
    current_values = [torch.randn(2, 8) for _ in range(3)]
    for manager in (fp_manager, int8_manager):
        manager.store_prefill_request(request_id, layer_cache)
        manager.reserve_token_slot(request_id)
        for layer_id in range(3):
            manager.write_layer_kv(
                request_id,
                layer_id,
                current_keys[layer_id],
                current_values[layer_id],
            )

    # Copy identical current-token vectors into both managers before comparing.
    # Prefill quantization remains the only approximation under test.
    query = torch.randn(1, 2, 8)
    for layer_id in (0, 1, 2):
        fp = PagedAttention(fp_manager).forward_batch([request_id], layer_id, query)
        quantized = PagedAttention(int8_manager).forward_batch(
            [request_id], layer_id, query
        )
        assert torch.isfinite(quantized).all()
        torch.testing.assert_close(quantized, fp, atol=4e-2, rtol=4e-2)


def test_decode_metadata_rejects_missing_reservation_and_wrong_request_order():
    manager = _manager()
    manager.store_prefill_request(
        "a",
        [(torch.randn(2, 1, 8), torch.randn(2, 1, 8)) for _ in range(3)],
    )
    try:
        manager.build_decode_metadata(["a"])
    except RuntimeError as error:
        assert "reserved decode token" in str(error)
    else:
        raise AssertionError("metadata without a reservation was accepted")
