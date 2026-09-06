import pytest
import torch

from src.model.gpt2 import convert_gpt2_model
from src.model.kv_manager import KVCacheManager
from src.model.paged_attention import PagedAttention


transformers = pytest.importorskip("transformers")


@pytest.mark.parametrize("num_layers", [2, 6])
def test_converted_gpt2_matches_huggingface_prefill_and_decode(num_layers):
    torch.manual_seed(7)
    source_config = transformers.GPT2Config(
        vocab_size=41,
        n_positions=16,
        n_ctx=16,
        n_embd=16,
        n_layer=num_layers,
        n_head=2,
        n_inner=32,
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=0,
        eos_token_id=1,
    )
    source_model = transformers.GPT2LMHeadModel(source_config).eval()
    model = convert_gpt2_model(source_model)
    for layer in model.layers:
        assert layer.self_attn.qkv_linear.in_features == source_config.n_embd
        assert layer.self_attn.qkv_linear.out_features == 3 * source_config.n_embd

    prompt = torch.tensor([[4, 8, 15, 16]], dtype=torch.long)
    with torch.inference_mode():
        expected_prefill = source_model(prompt).logits
        actual_prefill, layer_cache = model.prefill(prompt)

    torch.testing.assert_close(actual_prefill, expected_prefill, atol=1e-5, rtol=1e-5)
    assert model.output_layer.weight is model.embedding_table.weight

    kv_manager = KVCacheManager(
        block_size=4,
        total_memory=1024 * 1024,
        tensor_dtype=next(model.parameters()).dtype,
        device=torch.device("cpu"),
        num_layers=model.config.num_layers,
        num_kv_heads=model.config.num_heads,
        head_dim=model.config.head_dim,
    )
    paged_attention = PagedAttention(kv_manager)
    kv_manager.store_prefill("request", layer_cache)

    next_token = torch.tensor([[23]], dtype=torch.long)
    with torch.inference_mode():
        expected_decode = source_model(torch.cat((prompt, next_token), dim=1)).logits[
            :, -1:
        ]
        actual_decode = model.decode_batch(
            next_token,
            ["request"],
            kv_manager,
            paged_attention,
        )

    torch.testing.assert_close(actual_decode, expected_decode, atol=1e-5, rtol=1e-5)
