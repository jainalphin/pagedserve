import pytest
import torch
from torch.nn import functional as F


CUDA_TRITON = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA and Triton are required"
)
CONTEXT_LENGTHS = (1, 15, 16, 17, 31, 32, 33, 128, 512, 1024)
BATCH_SIZES = (1, 8, 32)
TEST_LAYERS = (0, 1, 2)  # first, second, and last


def _fixture(dtype):
    block_size = 16
    max_batch = max(BATCH_SIZES)
    max_blocks = max(CONTEXT_LENGTHS) // block_size
    required = max_batch * max_blocks
    physical_blocks = required + 17
    generator = torch.Generator(device="cuda").manual_seed(20260825)
    key_pool = torch.randn(
        3, physical_blocks, 1, block_size, 32,
        dtype=dtype, device="cuda", generator=generator,
    )
    value_pool = torch.randn(
        key_pool.shape,
        dtype=dtype, device="cuda", generator=generator,
    )
    fragmented = torch.randperm(
        physical_blocks, device="cuda", generator=generator
    )[:required].reshape(max_batch, max_blocks).to(torch.int32)
    # Assert this is genuinely neither a contiguous nor logical page mapping.
    assert not torch.equal(
        fragmented[0], torch.arange(max_blocks, device="cuda", dtype=torch.int32)
    )
    return key_pool, value_pool, fragmented


def _torch_gather_sdpa(queries, key_pool, value_pool, table, layer_id, context):
    keys = key_pool[layer_id, table.to(torch.long)].permute(0, 2, 1, 3, 4)
    values = value_pool[layer_id, table.to(torch.long)].permute(0, 2, 1, 3, 4)
    batch, heads, head_dim = queries.shape
    keys = keys.reshape(batch, heads, -1, head_dim)[:, :, :context]
    values = values.reshape(batch, heads, -1, head_dim)[:, :, :context]
    return F.scaled_dot_product_attention(
        queries.unsqueeze(2), keys, values, dropout_p=0.0
    ).squeeze(2)


@CUDA_TRITON
@pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
def test_torch_vs_triton_full_correctness_matrix(dtype):
    pytest.importorskip("triton")
    from src.kernels.triton_paged_attention import paged_decode_attention_triton

    key_pool, value_pool, fragmented = _fixture(dtype)
    dummy_scales = torch.empty(1, device="cuda", dtype=torch.float32)
    generator = torch.Generator(device="cuda").manual_seed(17)
    atol, rtol = ((3e-2, 3e-2) if dtype == torch.float16 else (3e-4, 3e-4))

    for batch_size in BATCH_SIZES:
        queries = torch.randn(
            batch_size, 1, 32, dtype=dtype, device="cuda", generator=generator
        )
        for context_length in CONTEXT_LENGTHS:
            logical_blocks = (context_length + 15) // 16
            table = fragmented[:batch_size, :logical_blocks]
            lengths = torch.full(
                (batch_size,), context_length, dtype=torch.int32, device="cuda"
            )
            for layer_id in TEST_LAYERS:
                expected = _torch_gather_sdpa(
                    queries, key_pool, value_pool, table, layer_id, context_length
                )
                output_buffer = torch.empty_like(queries)
                actual = paged_decode_attention_triton(
                    queries,
                    key_pool,
                    value_pool,
                    dummy_scales,
                    dummy_scales,
                    table,
                    lengths,
                    layer_id,
                    output=output_buffer,
                    maximum_context_length=context_length,
                    validate_inputs=False,
                )
                assert actual.data_ptr() == output_buffer.data_ptr()
                assert torch.isfinite(actual).all(), (
                    dtype, batch_size, context_length, layer_id
                )
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@CUDA_TRITON
@pytest.mark.parametrize(
    "corruption",
    ("zero_context", "oversized_context", "negative_page", "large_page"),
)
def test_invalid_decode_metadata_is_rejected(corruption):
    pytest.importorskip("triton")
    from src.kernels.triton_paged_attention import paged_decode_attention_triton

    query = torch.randn(1, 1, 32, device="cuda")
    key_pool = torch.randn(3, 4, 1, 16, 32, device="cuda")
    value_pool = torch.randn_like(key_pool)
    scales = torch.empty(1, device="cuda")
    table = torch.tensor([[2]], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([16], dtype=torch.int32, device="cuda")
    if corruption == "zero_context":
        lengths[0] = 0
    elif corruption == "oversized_context":
        lengths[0] = 17
    elif corruption == "negative_page":
        table[0, 0] = -1
    else:
        table[0, 0] = key_pool.shape[1]
    with pytest.raises(ValueError):
        paged_decode_attention_triton(
            query, key_pool, value_pool, scales, scales, table, lengths, 0
        )


@CUDA_TRITON
@pytest.mark.parametrize("nonfinite", (float("nan"), float("inf"), -float("inf")))
def test_nonfinite_queries_are_rejected(nonfinite):
    pytest.importorskip("triton")
    from src.kernels.triton_paged_attention import paged_decode_attention_triton

    query = torch.randn(1, 1, 32, device="cuda")
    query[0, 0, 0] = nonfinite
    pool = torch.randn(3, 1, 1, 16, 32, device="cuda")
    scales = torch.empty(1, device="cuda")
    with pytest.raises(ValueError, match="NaN or Inf"):
        paged_decode_attention_triton(
            query,
            pool,
            pool,
            scales,
            scales,
            torch.tensor([[0]], dtype=torch.int32, device="cuda"),
            torch.tensor([1], dtype=torch.int32, device="cuda"),
            0,
        )


@CUDA_TRITON
def test_nonfinite_used_kv_and_int8_scales_are_rejected():
    pytest.importorskip("triton")
    from src.kernels.triton_paged_attention import paged_decode_attention_triton

    query = torch.randn(1, 1, 32, device="cuda")
    table = torch.tensor([[0]], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([1], dtype=torch.int32, device="cuda")
    scales = torch.empty(1, device="cuda")
    float_pool = torch.randn(1, 1, 1, 16, 32, device="cuda")
    float_pool[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        paged_decode_attention_triton(
            query, float_pool, float_pool, scales, scales, table, lengths, 0
        )

    int8_pool = torch.zeros(1, 1, 1, 16, 32, dtype=torch.int8, device="cuda")
    int8_scales = torch.ones(1, 1, 1, 16, dtype=torch.float32, device="cuda")
    int8_scales[0, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="NaN or Inf"):
        paged_decode_attention_triton(
            query,
            int8_pool,
            int8_pool,
            int8_scales,
            int8_scales,
            table,
            lengths,
            0,
        )


@CUDA_TRITON
def test_int8_kv_is_dequantized_inside_triton_kernel():
    pytest.importorskip("triton")
    from src.kernels.triton_paged_attention import paged_decode_attention_triton

    torch.manual_seed(101)
    float_keys = torch.randn(3, 7, 2, 16, 64, device="cuda")
    float_values = torch.randn_like(float_keys)

    def quantize(values):
        scales = values.abs().amax(dim=-1) / 127
        scales = torch.where(scales > 0, scales, torch.ones_like(scales))
        integers = torch.round(values / scales.unsqueeze(-1)).clamp(-127, 127)
        return integers.to(torch.int8), scales.float()

    key_pool, key_scales = quantize(float_keys)
    value_pool, value_scales = quantize(float_values)
    table = torch.tensor([[5, 1, 6], [2, 4, 0]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([33, 33], device="cuda", dtype=torch.int32)
    queries = torch.randn(2, 2, 64, device="cuda", dtype=torch.float16)
    dequantized_keys = key_pool.float() * key_scales.unsqueeze(-1)
    dequantized_values = value_pool.float() * value_scales.unsqueeze(-1)
    expected = _torch_gather_sdpa(
        queries.float(), dequantized_keys, dequantized_values, table, 2, 33
    ).half()
    actual = paged_decode_attention_triton(
        queries,
        key_pool,
        value_pool,
        key_scales,
        value_scales,
        table,
        lengths,
        2,
        maximum_context_length=33,
        validate_inputs=False,
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
