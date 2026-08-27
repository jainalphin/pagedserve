import copy

import pytest
import torch

from src.model.iteration import IterationBatch, IterationItem
from src.model.kv_manager import KVCacheManager
from src.model.paged_attention import PagedAttention
from src.model.paged_decoder import PagedDecoderLM, TransformerConfig


def _iteration(request_id, token_id, position):
    return IterationBatch(
        items=(
            IterationItem(
                request_id=request_id,
                phase="decode",
                token_ids=(token_id,),
                position_ids=(position,),
                start_offset=0,
                end_offset=1,
            ),
        ),
        input_ids=torch.tensor([token_id], dtype=torch.long, device="cuda"),
        position_ids=torch.tensor([position], dtype=torch.long, device="cuda"),
    )


def _runtime(model):
    manager = KVCacheManager(
        block_size=16,
        total_memory=2 * 1024 * 1024,
        tensor_dtype=torch.float16,
        device=torch.device("cuda"),
        num_layers=model.config.num_layers,
        num_kv_heads=model.config.num_heads,
        head_dim=model.config.head_dim,
    )
    return manager, PagedAttention(manager, decode_attention_backend="triton")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA and Triton are required")
def test_cuda_graph_decode_matches_eager_and_replays():
    pytest.importorskip("triton")
    torch.manual_seed(2026)
    config = TransformerConfig(
        vocab_size=64,
        hidden_size=64,
        num_layers=2,
        num_heads=2,
        head_dim=32,
        mlp_hidden_size=128,
        max_sequence_length=32,
    )
    graph_model = PagedDecoderLM(config, enable_cuda_graphs=True).cuda().half().eval()
    eager_model = copy.deepcopy(graph_model)
    eager_model.enable_cuda_graphs = False
    graph_manager, graph_attention = _runtime(graph_model)
    eager_manager, eager_attention = _runtime(eager_model)
    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device="cuda")

    with torch.inference_mode():
        _, graph_cache = graph_model.prefill(prompt)
        _, eager_cache = eager_model.prefill(prompt)
        graph_manager.store_prefill("graph", graph_cache)
        eager_manager.store_prefill("eager", eager_cache)

        for token_id, position in ((5, 4), (6, 5)):
            graph_logits = graph_model.forward_iteration(
                _iteration("graph", token_id, position),
                graph_manager,
                graph_attention,
            )
            eager_logits = eager_model.forward_iteration(
                _iteration("eager", token_id, position),
                eager_manager,
                eager_attention,
            )
            torch.testing.assert_close(
                graph_logits,
                eager_logits,
                atol=3e-2,
                rtol=3e-2,
            )

    summary = graph_model.cuda_graph_summary()
    assert summary["captured_graphs"] == 1
    assert summary["graph_replays"] == 2
    assert summary["capture_failures"] == []
    assert eager_model.cuda_graph_summary()["captured_graphs"] == 0
