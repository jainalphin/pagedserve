import pytest
import torch
from torch.nn import functional as F


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA and Triton are required")
@pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
@pytest.mark.parametrize("rows", (1, 8, 32))
def test_fused_residual_layernorm_matches_torch(dtype, rows):
    pytest.importorskip("triton")
    from src.kernels.triton_residual_layernorm import fused_residual_layer_norm

    generator = torch.Generator(device="cuda").manual_seed(515)
    residual = torch.randn(rows, 768, device="cuda", dtype=dtype, generator=generator)
    update = torch.randn(rows, 768, device="cuda", dtype=dtype, generator=generator)
    weight = torch.randn(768, device="cuda", dtype=dtype, generator=generator)
    bias = torch.randn(768, device="cuda", dtype=dtype, generator=generator)
    expected_residual = residual + update
    expected_normalized = F.layer_norm(
        expected_residual,
        (768,),
        weight,
        bias,
        1e-5,
    )
    actual_residual, actual_normalized = fused_residual_layer_norm(
        residual,
        update,
        weight,
        bias,
        1e-5,
    )
    atol, rtol = ((3e-3, 3e-3) if dtype == torch.float16 else (2e-5, 2e-5))
    torch.testing.assert_close(actual_residual, expected_residual, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_normalized, expected_normalized, atol=atol, rtol=rtol)
