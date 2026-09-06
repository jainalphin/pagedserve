import math

import pytest
import torch

from src.model.kv_manager import KVCacheManager
from src.model.iteration import IterationBatch, IterationItem
from src.model.paged_attention import PagedAttention
from src.model.paged_decoder import (
    PagedDecoderLM,
    TransformerConfig,
)
from src.scheduler.orca_scheduler import (
    SARATHI_STRATEGY,
    ContinuousBatchScheduler,
)


def test_store_prefill():
    num_layers = 2
    num_kv_heads = 2
    head_dim = 3
    block_size = 2
    prompt_length = 5

    manager = KVCacheManager(
        block_size=block_size,
        total_memory=4096,
        tensor_dtype=torch.float32,
        device="cpu",
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    cache = []
    original_layers = []

    for layer_id in range(num_layers):
        key = torch.arange(
            num_kv_heads * prompt_length * head_dim,
            dtype=torch.float32,
        ).reshape(1, num_kv_heads, prompt_length, head_dim)

        # Make each layer and K/V distinguishable.
        key = key + layer_id * 1_000
        value = key + 10_000

        cache.append((key, value))
        original_layers.append((key.clone(), value.clone()))

    request_id = "test-request"
    initial_free_blocks = len(manager.free_blocks)

    manager.store_prefill(request_id, cache)

    request = manager.requests[request_id]
    expected_blocks = math.ceil(prompt_length / block_size)

    assert request.sequence_length == prompt_length
    assert len(request.block_ids) == expected_blocks
    assert len(manager.free_blocks) == initial_free_blocks - expected_blocks

    assert request.reserved_position is None
    assert request.reserved_block_id is None
    assert request.reserved_block_offset is None
    assert request.written_layer_ids == set()

    for token_index in range(prompt_length):
        logical_block = token_index // block_size
        block_offset = token_index % block_size
        physical_block = request.block_ids[logical_block]

        for layer_id in range(num_layers):
            expected_key = original_layers[layer_id][0][0, :, token_index, :]
            expected_value = original_layers[layer_id][1][0, :, token_index, :]

            stored_key = manager.key_pool[
                layer_id, physical_block, :, block_offset, :
            ]
            stored_value = manager.value_pool[
                layer_id, physical_block, :, block_offset, :
            ]

            torch.testing.assert_close(stored_key, expected_key)
            torch.testing.assert_close(stored_value, expected_value)

    print("store_prefill test passed")

def test_single_request_paged_attention():
    torch.manual_seed(10)

    num_layers = 2
    num_heads = 2
    head_dim = 3
    block_size = 2
    prompt_length = 5
    layer_id = 0
    request_id = "paged-attention-test"

    manager = KVCacheManager(
        block_size=block_size,
        total_memory=4096,
        tensor_dtype=torch.float32,
        device="cpu",
        num_layers=num_layers,
        num_kv_heads=num_heads,
        head_dim=head_dim,
    )

    # Change this line if your constructor has different arguments.
    paged_attention = PagedAttention(
        kv_manager=manager,
    )

    # Create a synthetic prefill cache.
    prefill_cache = []
    original_keys = []
    original_values = []

    for current_layer in range(num_layers):
        keys = torch.randn(
            1,
            num_heads,
            prompt_length,
            head_dim,
        )
        values = torch.randn(
            1,
            num_heads,
            prompt_length,
            head_dim,
        )

        prefill_cache.append((keys, values))
        original_keys.append(keys)
        original_values.append(values)

    manager.store_prefill(request_id, prefill_cache)

    # Simulate one new decode token for layer 0.
    query = torch.randn(num_heads, head_dim)
    new_key = torch.randn(num_heads, head_dim)
    new_value = torch.randn(num_heads, head_dim)

    manager.reserve_token_slot(request_id)
    manager.write_layer_kv(
        request_id,
        layer_id,
        new_key,
        new_value,
    )

    # Calculate using the paged implementation.
    paged_scores = paged_attention.attention_score(
        request_id,
        layer_id,
        query,
    )

    paged_output = paged_attention.compute_weighted_value_sum(
        request_id,
        layer_id,
        paged_scores,
    )

    # Construct the equivalent contiguous K/V tensors.
    contiguous_keys = torch.cat(
        [
            original_keys[layer_id][0],
            new_key.unsqueeze(1),
        ],
        dim=1,
    )

    contiguous_values = torch.cat(
        [
            original_values[layer_id][0],
            new_value.unsqueeze(1),
        ],
        dim=1,
    )

    scale = head_dim ** -0.5

    reference_scores = torch.matmul(
        query.unsqueeze(1),
        contiguous_keys.transpose(-2, -1),
    ).squeeze(1) * scale

    reference_probabilities = torch.softmax(
        reference_scores,
        dim=-1,
    )

    reference_output = torch.matmul(
        reference_probabilities.unsqueeze(1),
        contiguous_values,
    ).squeeze(1)

    torch.testing.assert_close(
        paged_scores,
        reference_scores,
        rtol=1e-5,
        atol=1e-6,
    )

    torch.testing.assert_close(
        paged_output,
        reference_output,
        rtol=1e-5,
        atol=1e-6,
    )

    assert paged_scores.shape == (
        num_heads,
        prompt_length + 1,
    )
    assert paged_output.shape == (
        num_heads,
        head_dim,
    )

    print("single-request PagedAttention test passed")


def test_paged_attention_matches_contiguous_attention():
    torch.manual_seed(42)

    num_tokens = 5
    num_heads = 2
    head_dim = 4
    layer_id = 0
    request_id = "test-request"

    manager = KVCacheManager(
        block_size=2,
        total_memory=1024,
        tensor_dtype=torch.float32,
        device="cpu",
        num_layers=1,
        num_kv_heads=num_heads,
        head_dim=head_dim,
    )

    paged_attention = PagedAttention(
        kv_manager=manager,
    )

    all_keys = torch.randn(num_tokens, num_heads, head_dim)
    all_values = torch.randn(num_tokens, num_heads, head_dim)
    query = torch.randn(num_heads, head_dim)

    # Store and commit previous tokens.
    for token_index in range(num_tokens - 1):
        manager.reserve_token_slot(request_id)

        manager.write_layer_kv(
            request_id=request_id,
            layer_id=layer_id,
            new_key=all_keys[token_index],
            new_value=all_values[token_index],
        )

        manager.commit_token(request_id)

    # Write the current token but leave it uncommitted during attention.
    manager.reserve_token_slot(request_id)

    manager.write_layer_kv(
        request_id=request_id,
        layer_id=layer_id,
        new_key=all_keys[-1],
        new_value=all_values[-1],
    )

    paged_output = paged_attention.forward(
        request_id=request_id,
        layer_id=layer_id,
        query=query,
    )

    # Ordinary contiguous-attention baseline.
    contiguous_keys = all_keys.permute(1, 0, 2)
    contiguous_values = all_values.permute(1, 0, 2)

    scores = torch.matmul(
        query.unsqueeze(1),
        contiguous_keys.transpose(-2, -1),
    ).squeeze(1)

    scores = scores * (head_dim ** -0.5)
    probabilities = torch.softmax(scores, dim=-1)

    expected_output = torch.matmul(
        probabilities.unsqueeze(1),
        contiguous_values,
    ).squeeze(1)

    assert paged_output.shape == (num_heads, head_dim)
    assert len(manager.requests[request_id].block_ids) == 3
    assert manager.requests[request_id].sequence_length == 4

    torch.testing.assert_close(
        paged_output,
        expected_output,
        rtol=1e-5,
        atol=1e-6,
    )

    manager.commit_token(request_id)
    assert manager.requests[request_id].sequence_length == 5


def test_batched_decode_attention_matches_scalar_with_variable_contexts():
    torch.manual_seed(43)
    num_heads = 2
    head_dim = 4
    manager = KVCacheManager(
        block_size=2,
        total_memory=16 * 1024,
        tensor_dtype=torch.float32,
        device="cpu",
        num_layers=1,
        num_kv_heads=num_heads,
        head_dim=head_dim,
    )
    attention = PagedAttention(manager)
    request_ids = ["short", "medium", "long"]
    context_lengths = [1, 3, 5]
    queries = torch.randn(len(request_ids), num_heads, head_dim)

    for request_id, context_length in zip(request_ids, context_lengths):
        prompt_length = context_length - 1
        if prompt_length:
            manager.store_prefill_request(
                request_id,
                [
                    (
                        torch.randn(num_heads, prompt_length, head_dim),
                        torch.randn(num_heads, prompt_length, head_dim),
                    )
                ],
            )
        manager.reserve_token_slot(request_id)
        manager.write_layer_kv(
            request_id,
            0,
            torch.randn(num_heads, head_dim),
            torch.randn(num_heads, head_dim),
        )

    expected = torch.stack(
        [
            attention.forward(request_id, 0, query)
            for request_id, query in zip(request_ids, queries)
        ]
    )
    actual = attention.forward_batch(request_ids, 0, queries)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_paged_decode_matches_torch_across_kv_blocks():
    pytest.importorskip("triton")
    torch.manual_seed(45)
    device = torch.device("cuda")
    num_heads = 2
    head_dim = 64
    num_layers = 12
    manager = KVCacheManager(
        block_size=16,
        total_memory=4 * 1024 * 1024,
        tensor_dtype=torch.float32,
        device=device,
        num_layers=num_layers,
        num_kv_heads=num_heads,
        head_dim=head_dim,
    )
    request_ids = ["one-token", "partial-block", "three-blocks"]
    context_lengths = [1, 19, 35]
    queries = torch.randn(
        len(request_ids),
        num_heads,
        head_dim,
        device=device,
    )

    for request_id, context_length in zip(request_ids, context_lengths):
        prompt_length = context_length - 1
        if prompt_length:
            layer_cache = [
                (
                    torch.randn(
                        num_heads, prompt_length, head_dim, device=device
                    ),
                    torch.randn(
                        num_heads, prompt_length, head_dim, device=device
                    ),
                )
                for _ in range(num_layers)
            ]
            manager.store_prefill_request(
                request_id,
                layer_cache,
            )
        manager.reserve_token_slot(request_id)
        for layer_id in range(num_layers):
            manager.write_layer_kv(
                request_id,
                layer_id,
                torch.randn(num_heads, head_dim, device=device),
                torch.randn(num_heads, head_dim, device=device),
            )

    for tested_layer in (0, num_layers - 1):
        expected = PagedAttention(
            manager,
            decode_attention_backend="torch",
        ).forward_batch(
            request_ids,
            tested_layer,
            queries,
        )
        actual = PagedAttention(
            manager,
            decode_attention_backend="triton",
        ).forward_batch(
            request_ids,
            tested_layer,
            queries,
        )
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


def test_batched_fresh_prefill_matches_individual_causal_attention():
    torch.manual_seed(44)
    batch_size = 3
    token_count = 5
    num_heads = 2
    head_dim = 4
    manager = KVCacheManager(
        block_size=2,
        total_memory=16 * 1024,
        tensor_dtype=torch.float32,
        device="cpu",
        num_layers=1,
        num_kv_heads=num_heads,
        head_dim=head_dim,
    )
    attention = PagedAttention(manager)
    queries = torch.randn(batch_size, token_count, num_heads, head_dim)
    keys = torch.randn_like(queries)
    values = torch.randn_like(queries)

    expected = torch.stack(
        [
            attention.causal_prefill(query, key, value)
            for query, key, value in zip(queries, keys, values)
        ]
    )
    actual = attention.causal_prefill_batch(queries, keys, values)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_prefill():
    config = TransformerConfig(
        vocab_size=100,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        head_dim=16,
        mlp_hidden_size=256,
        max_sequence_length=32,
    )

    model = PagedDecoderLM(config).eval()

    batch_size = 2
    prompt_length = 5

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (batch_size, prompt_length),
    )

    with torch.inference_mode():
        logits, kv_caches = model.prefill(input_ids)

    assert len(model.layers) == config.num_layers
    assert len(kv_caches) == config.num_layers

    assert logits.shape == (
        batch_size,
        prompt_length,
        config.vocab_size,
    )

    expected_kv_shape = (
        batch_size,
        config.num_heads,
        prompt_length,
        config.head_dim,
    )

    for keys, values in kv_caches:
        assert keys.shape == expected_kv_shape
        assert values.shape == expected_kv_shape

    print("Test passed")


def test_prefill_then_one_decode():
    config = TransformerConfig(
        vocab_size=100,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        head_dim=16,
        mlp_hidden_size=256,
        max_sequence_length=32,
    )

    model = PagedDecoderLM(config).eval()

    manager = KVCacheManager(
        block_size=4,
        total_memory=1024 * 1024,
        tensor_dtype=next(model.parameters()).dtype,
        device="cpu",
        num_layers=config.num_layers,
        num_kv_heads=config.num_heads,
        head_dim=config.head_dim,
    )

    paged_attn_manager = PagedAttention(
        kv_manager=manager,
    )

    request_id = "request-1"
    input_ids = torch.tensor([[10, 20, 30]], dtype=torch.long)

    with torch.inference_mode():
        prefill_logits, layer_kv_caches = model.prefill(input_ids)

    manager.store_prefill(
        request_id,
        layer_kv_caches,
    )

    prompt_length = input_ids.shape[1]

    assert manager.requests[request_id].sequence_length == prompt_length

    next_token = prefill_logits[:, -1, :].argmax(
        dim=-1,
        keepdim=True,
    )

    with torch.inference_mode():
        decode_logits = model.decode_batch(
            input_ids=next_token,
            request_ids=[request_id],
            kv_manager=manager,
            paged_attn_manager=paged_attn_manager
        )

    request_info = manager.requests[request_id]

    assert decode_logits.shape == (
        1,
        1,
        config.vocab_size,
    )

    assert request_info.sequence_length == prompt_length + 1
    assert request_info.reserved_position is None
    assert request_info.reserved_block_id is None
    assert request_info.reserved_block_offset is None
    assert request_info.written_layer_ids == set()

    print("Prefill + decode test passed")


def test_multi_request_prefill_then_one_decode():
    config = TransformerConfig(
        vocab_size=100,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        head_dim=16,
        mlp_hidden_size=256,
        max_sequence_length=32,
    )

    model = PagedDecoderLM(config).eval()

    manager = KVCacheManager(
        block_size=4,
        total_memory=1024 * 1024,
        tensor_dtype=next(model.parameters()).dtype,
        device="cpu",
        num_layers=config.num_layers,
        num_kv_heads=config.num_heads,
        head_dim=config.head_dim,
    )

    paged_attn_manager = PagedAttention(
        kv_manager=manager,
    )

    request_ids = ["request-A", "request-B"]

    # Equal-length prompts allow one batched prefill without padding.
    input_ids = torch.tensor(
        [
            [10, 20, 30, 40],
            [50, 60, 70, 80],
        ],
        dtype=torch.long,
    )

    batch_size, prompt_length = input_ids.shape

    with torch.inference_mode():
        prefill_logits, layer_kv_caches = model.prefill(input_ids)

    assert prefill_logits.shape == (
        batch_size,
        prompt_length,
        config.vocab_size,
    )

    assert len(layer_kv_caches) == config.num_layers

    # Scatter each prefill batch row into its own request block table.
    for batch_index, request_id in enumerate(request_ids):
        manager.store_prefill(
            request_id=request_id,
            kv_cache=layer_kv_caches,
            batch_index=batch_index,
        )

    for request_id in request_ids:
        request_info = manager.requests[request_id]

        assert request_info.sequence_length == prompt_length
        assert request_info.reserved_position is None
        assert request_info.reserved_block_id is None
        assert request_info.reserved_block_offset is None
        assert request_info.written_layer_ids == set()

    # Ensure each request owns different physical blocks.
    request_a_blocks = set(manager.requests["request-A"].block_ids)
    request_b_blocks = set(manager.requests["request-B"].block_ids)

    assert request_a_blocks.isdisjoint(request_b_blocks)

    # Verify that each scattered request matches its source batch row.
    for layer_id in range(config.num_layers):
        for batch_index, request_id in enumerate(request_ids):
            gathered_key, gathered_value = manager.gather_layer(
                request_id,
                layer_id,
            )

            expected_key = layer_kv_caches[layer_id][0][
                batch_index : batch_index + 1
            ]
            expected_value = layer_kv_caches[layer_id][1][
                batch_index : batch_index + 1
            ]

            torch.testing.assert_close(gathered_key, expected_key)
            torch.testing.assert_close(gathered_value, expected_value)

    # Each row produces one pending token for decode.
    next_tokens = prefill_logits[:, -1, :].argmax(
        dim=-1,
        keepdim=True,
    )

    assert next_tokens.shape == (batch_size, 1)

    with torch.inference_mode():
        decode_logits = model.decode_batch(
            input_ids=next_tokens,
            request_ids=request_ids,
            kv_manager=manager,
            paged_attn_manager=paged_attn_manager,
        )

    assert decode_logits.shape == (
        batch_size,
        1,
        config.vocab_size,
    )

    for request_id in request_ids:
        request_info = manager.requests[request_id]

        assert request_info.sequence_length == prompt_length + 1
        assert request_info.reserved_position is None
        assert request_info.reserved_block_id is None
        assert request_info.reserved_block_offset is None
        assert request_info.written_layer_ids == set()

        # Prompt length 4 fills one block; decode allocates a second.
        assert len(request_info.block_ids) == 2

    print("Batched prefill + multi-request decode test passed")


def test_flattened_orca_iteration_mixes_prefill_and_decode():
    torch.manual_seed(7)
    config = TransformerConfig(
        vocab_size=100,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        head_dim=16,
        mlp_hidden_size=256,
        max_sequence_length=32,
    )
    model = PagedDecoderLM(config).eval()
    manager = KVCacheManager(
        block_size=4,
        total_memory=1024 * 1024,
        tensor_dtype=next(model.parameters()).dtype,
        device="cpu",
        num_layers=config.num_layers,
        num_kv_heads=config.num_heads,
        head_dim=config.head_dim,
    )
    paged_attention = PagedAttention(
        kv_manager=manager,
    )

    request_a_prefill = IterationBatch(
        items=(
            IterationItem(
                request_id="A",
                phase="prefill",
                token_ids=(10, 20, 30),
                position_ids=(0, 1, 2),
                start_offset=0,
                end_offset=3,
            ),
        ),
        input_ids=torch.tensor([10, 20, 30], dtype=torch.long),
        position_ids=torch.tensor([0, 1, 2], dtype=torch.long),
    )

    with torch.inference_mode():
        request_a_logits = model.forward_iteration(
            request_a_prefill,
            manager,
            paged_attention,
        )

    request_a_token = request_a_logits.argmax(dim=-1).item()
    mixed_iteration = IterationBatch(
        items=(
            IterationItem(
                request_id="A",
                phase="decode",
                token_ids=(request_a_token,),
                position_ids=(3,),
                start_offset=0,
                end_offset=1,
            ),
            IterationItem(
                request_id="B",
                phase="prefill",
                token_ids=(40, 50),
                position_ids=(0, 1),
                start_offset=1,
                end_offset=3,
            ),
        ),
        input_ids=torch.tensor([request_a_token, 40, 50], dtype=torch.long),
        position_ids=torch.tensor([3, 0, 1], dtype=torch.long),
    )

    with torch.inference_mode():
        logits = model.forward_iteration(
            mixed_iteration,
            manager,
            paged_attention,
        )

    assert logits.shape == (2, config.vocab_size)
    assert manager.requests["A"].sequence_length == 4
    assert manager.requests["B"].sequence_length == 2
    for request_id in ("A", "B"):
        request_info = manager.requests[request_id]
        assert request_info.reserved_position is None
        assert request_info.written_layer_ids == set()


def test_flattened_prefill_matches_dense_prefill():
    torch.manual_seed(9)
    config = TransformerConfig(
        vocab_size=100,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        head_dim=16,
        mlp_hidden_size=256,
        max_sequence_length=32,
    )
    model = PagedDecoderLM(config).eval()
    manager = KVCacheManager(
        block_size=4,
        total_memory=1024 * 1024,
        tensor_dtype=next(model.parameters()).dtype,
        device="cpu",
        num_layers=config.num_layers,
        num_kv_heads=config.num_heads,
        head_dim=config.head_dim,
    )
    paged_attention = PagedAttention(
        kv_manager=manager,
    )
    dense_input = torch.tensor([[12, 34, 56]], dtype=torch.long)

    with torch.inference_mode():
        dense_logits, dense_cache = model.prefill(dense_input)

    iteration = IterationBatch(
        items=(
            IterationItem(
                request_id="flattened",
                phase="prefill",
                token_ids=(12, 34, 56),
                position_ids=(0, 1, 2),
                start_offset=0,
                end_offset=3,
            ),
        ),
        input_ids=dense_input[0],
        position_ids=torch.tensor([0, 1, 2], dtype=torch.long),
    )
    with torch.inference_mode():
        iteration_logits = model.forward_iteration(
            iteration,
            manager,
            paged_attention,
        )

    torch.testing.assert_close(iteration_logits[0], dense_logits[0, -1])
    for layer_id, (expected_keys, expected_values) in enumerate(dense_cache):
        stored_keys, stored_values = manager.gather_layer("flattened", layer_id)
        torch.testing.assert_close(stored_keys, expected_keys)
        torch.testing.assert_close(stored_values, expected_values)


def test_chunked_prefill_matches_dense_prefill():
    torch.manual_seed(10)
    config = TransformerConfig(
        vocab_size=100,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        head_dim=16,
        mlp_hidden_size=256,
        max_sequence_length=32,
    )
    model = PagedDecoderLM(config).eval()
    manager = KVCacheManager(
        block_size=4,
        total_memory=1024 * 1024,
        tensor_dtype=next(model.parameters()).dtype,
        device="cpu",
        num_layers=config.num_layers,
        num_kv_heads=config.num_heads,
        head_dim=config.head_dim,
    )
    paged_attention = PagedAttention(manager)
    prompt = torch.tensor([[12, 34, 56, 78, 21]], dtype=torch.long)

    with torch.inference_mode():
        dense_logits, dense_cache = model.prefill(prompt)

    chunk_logits = None
    for start, end in ((0, 2), (2, 4), (4, 5)):
        chunk = IterationBatch(
            items=(
                IterationItem(
                    request_id="chunked",
                    phase="prefill",
                    token_ids=tuple(prompt[0, start:end].tolist()),
                    position_ids=tuple(range(start, end)),
                    start_offset=0,
                    end_offset=end - start,
                    produces_output=end == prompt.shape[1],
                ),
            ),
            input_ids=prompt[0, start:end],
            position_ids=torch.arange(start, end, dtype=torch.long),
        )
        with torch.inference_mode():
            chunk_logits = model.forward_iteration(
                chunk,
                manager,
                paged_attention,
            )
        assert manager.requests["chunked"].sequence_length == end
        expected_logit_rows = 1 if end == prompt.shape[1] else 0
        assert chunk_logits.shape == (expected_logit_rows, config.vocab_size)

    torch.testing.assert_close(
        chunk_logits[0],
        dense_logits[0, -1],
        atol=1e-5,
        rtol=1e-5,
    )
    for layer_id, (expected_keys, expected_values) in enumerate(dense_cache):
        stored_keys, stored_values = manager.gather_layer("chunked", layer_id)
        torch.testing.assert_close(stored_keys, expected_keys, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            stored_values,
            expected_values,
            atol=1e-5,
            rtol=1e-5,
        )


class IntegerTokenizer:
    eos_token_id = None

    def encode(self, prompt):
        return [int(token) for token in prompt.split()]

    def decode(self, token_ids):
        return " ".join(str(token_id) for token_id in token_ids)


def test_orca_scheduler_batches_different_prompt_lengths():
    torch.manual_seed(11)
    config = TransformerConfig(
        vocab_size=100,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        head_dim=16,
        mlp_hidden_size=256,
        max_sequence_length=32,
    )
    model = PagedDecoderLM(config).eval()
    manager = KVCacheManager(
        block_size=4,
        total_memory=1024 * 1024,
        tensor_dtype=next(model.parameters()).dtype,
        device="cpu",
        num_layers=config.num_layers,
        num_kv_heads=config.num_heads,
        head_dim=config.head_dim,
    )
    paged_attention = PagedAttention(
        kv_manager=manager,
    )
    scheduler = ContinuousBatchScheduler(
        model_engine=model,
        max_batch_size=2,
        tokenizer=IntegerTokenizer(),
        kv_manager=manager,
        paged_attn_manager=paged_attention,
    )

    request_a = scheduler.add_request("10 20 30", max_new_tokens=2)
    request_b = scheduler.add_request("40 50", max_new_tokens=2)

    first_iteration_tokens = scheduler.step()
    assert set(first_iteration_tokens) == {request_a, request_b}
    assert manager.requests[request_a].sequence_length == 3
    assert manager.requests[request_b].sequence_length == 2
    assert set(scheduler.active) == {request_a, request_b}

    second_iteration_tokens = scheduler.step()
    assert set(second_iteration_tokens) == {request_a, request_b}
    assert not scheduler.active
    assert not scheduler.waiting
    assert set(scheduler.finished) == {request_a, request_b}
    assert request_a not in manager.requests
    assert request_b not in manager.requests
    assert scheduler.reserved_blocks == 0


def test_sarathi_scheduler_chunks_prefill_and_prioritizes_decodes():
    torch.manual_seed(12)
    config = TransformerConfig(
        vocab_size=100,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        head_dim=16,
        mlp_hidden_size=256,
        max_sequence_length=32,
    )
    model = PagedDecoderLM(config).eval()
    manager = KVCacheManager(
        block_size=4,
        total_memory=1024 * 1024,
        tensor_dtype=next(model.parameters()).dtype,
        device="cpu",
        num_layers=config.num_layers,
        num_kv_heads=config.num_heads,
        head_dim=config.head_dim,
    )
    scheduler = ContinuousBatchScheduler(
        model_engine=model,
        max_batch_size=2,
        tokenizer=IntegerTokenizer(),
        kv_manager=manager,
        paged_attn_manager=PagedAttention(manager),
        scheduling_strategy=SARATHI_STRATEGY,
        prefill_chunk_size=2,
    )

    request_a = scheduler.add_request("10 20 30 40 50", max_new_tokens=2)
    assert scheduler.step() == {}
    assert scheduler.waiting[0].prefill_cursor == 2
    assert manager.requests[request_a].sequence_length == 2

    assert scheduler.step() == {}
    assert scheduler.waiting[0].prefill_cursor == 4
    assert manager.requests[request_a].sequence_length == 4

    first_tokens = scheduler.step()
    assert set(first_tokens) == {request_a}
    assert request_a in scheduler.active
    assert not scheduler.waiting

    request_b = scheduler.add_request("60 70 80 90", max_new_tokens=2)
    mixed_tokens = scheduler.step()
    assert set(mixed_tokens) == {request_a}
    assert request_a in scheduler.finished
    assert scheduler.waiting[0].request_id == request_b
    assert scheduler.waiting[0].prefill_cursor == 2
    assert manager.requests[request_b].sequence_length == 2

    final_prefill_tokens = scheduler.step()
    assert set(final_prefill_tokens) == {request_b}
    assert request_b in scheduler.active
    assert not scheduler.waiting

    scheduler.run_until_complete()
    assert set(scheduler.finished) == {request_a, request_b}
    assert scheduler.reserved_blocks == 0
