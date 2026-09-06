"""GPT-2 adapter for the PagedServe model runtime."""

from typing import Any

import torch

from src.model.paged_decoder import PagedDecoderLM, TransformerConfig


DEFAULT_GPT2_MODEL_ID = "openai-community/gpt2"
DEFAULT_DISTILGPT2_MODEL_ID = "distilbert/distilgpt2"


def _copy_linear_from_conv1d(target, source):
    """Copy a Hugging Face GPT-2 Conv1D projection into nn.Linear."""
    target.weight.copy_(source.weight.transpose(0, 1))
    target.bias.copy_(source.bias)


def convert_gpt2_model(huggingface_model: Any) -> PagedDecoderLM:
    """Convert a GPT2LMHeadModel to PagedServe's paged decoder implementation."""
    source_config = huggingface_model.config
    if source_config.n_embd % source_config.n_head != 0:
        raise ValueError("GPT-2 embedding size must be divisible by its head count")
    if getattr(source_config, "add_cross_attention", False):
        raise ValueError("GPT-2 models with cross-attention are not supported")
    if source_config.activation_function not in ("gelu_new", "gelu_pytorch_tanh"):
        raise ValueError(
            "Only GPT-2 checkpoints using the tanh GELU activation are supported"
        )

    config = TransformerConfig(
        vocab_size=source_config.vocab_size,
        hidden_size=source_config.n_embd,
        num_layers=source_config.n_layer,
        num_heads=source_config.n_head,
        head_dim=source_config.n_embd // source_config.n_head,
        mlp_hidden_size=source_config.n_inner or 4 * source_config.n_embd,
        max_sequence_length=source_config.max_position_embeddings,
        activation_function="gelu_tanh",
        layer_norm_epsilon=source_config.layer_norm_epsilon,
        tie_word_embeddings=source_config.tie_word_embeddings,
        lm_head_bias=False,
    )
    model = PagedDecoderLM(config)
    source = huggingface_model.transformer

    with torch.no_grad():
        model.embedding_table.weight.copy_(source.wte.weight)
        model.position_embeddings.weight.copy_(source.wpe.weight)

        for target_layer, source_layer in zip(model.layers, source.h):
            target_layer.input_layernorm.load_state_dict(source_layer.ln_1.state_dict())
            target_layer.post_attention_layernorm.load_state_dict(
                source_layer.ln_2.state_dict()
            )

            _copy_linear_from_conv1d(
                target_layer.self_attn.qkv_linear,
                source_layer.attn.c_attn,
            )

            _copy_linear_from_conv1d(
                target_layer.self_attn.output_linear,
                source_layer.attn.c_proj,
            )
            _copy_linear_from_conv1d(
                target_layer.mlp_up_proj,
                source_layer.mlp.c_fc,
            )
            _copy_linear_from_conv1d(
                target_layer.mlp_down_proj,
                source_layer.mlp.c_proj,
            )

        model.final_layernorm.load_state_dict(source.ln_f.state_dict())
        if not config.tie_word_embeddings:
            model.output_layer.weight.copy_(huggingface_model.lm_head.weight)

    return model.eval()


def load_gpt2_pretrained(model_id: str = DEFAULT_GPT2_MODEL_ID):
    """Download a GPT-2 checkpoint and return a PagedServe model and tokenizer."""
    try:
        from transformers import AutoTokenizer, GPT2LMHeadModel
    except ImportError as error:
        raise ImportError(
            "GPT-2 support requires the 'transformers' package. Run ./env.sh first."
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    huggingface_model = GPT2LMHeadModel.from_pretrained(model_id).eval()
    return convert_gpt2_model(huggingface_model), tokenizer
