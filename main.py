import argparse
import importlib.util

import torch

from src.model.gpt2 import (
    DEFAULT_DISTILGPT2_MODEL_ID,
    DEFAULT_GPT2_MODEL_ID,
    load_gpt2_pretrained,
)
from src.model.kv_manager import KVCacheManager
from src.model.paged_attention import PagedAttention
from src.model.paged_decoder import PagedDecoderLM, TransformerConfig
from src.model.tokenizer import ByteTokenizer
from src.scheduler.orca_scheduler import (
    ORCA_STRATEGY,
    SUPPORTED_SCHEDULING_STRATEGIES,
    ContinuousBatchScheduler,
)


REFERENCE_MODEL = "reference"
DISTILGPT2_MODEL = "distilgpt2"
GPT2_MODEL = "gpt2"
SUPPORTED_MODELS = (REFERENCE_MODEL, DISTILGPT2_MODEL, GPT2_MODEL)
SUPPORTED_DTYPES = ("float32", "float16", "bfloat16")
SUPPORTED_DECODE_ATTENTION_BACKENDS = ("torch", "triton")
SUPPORTED_KV_CACHE_DTYPES = ("model", "int8")
MIB = 1024 * 1024
DEFAULT_CUDA_KV_MEMORY_UTILIZATION = 0.90
DEFAULT_CUDA_KV_SAFETY_MB = 3072


def cuda_memory_snapshot(device):
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    free_memory, total_memory = torch.cuda.mem_get_info(device)
    return {
        "free_bytes": free_memory,
        "total_bytes": total_memory,
        "device_used_bytes": total_memory - free_memory,
        "torch_allocated_bytes": torch.cuda.memory_allocated(device),
        "torch_reserved_bytes": torch.cuda.memory_reserved(device),
    }


def calculate_cuda_kv_cache_budget(
    free_memory,
    total_memory,
    memory_utilization=DEFAULT_CUDA_KV_MEMORY_UTILIZATION,
    safety_memory=DEFAULT_CUDA_KV_SAFETY_MB * MIB,
):
    """Choose a KV budget without treating all currently free VRAM as safe."""
    if not 0 < memory_utilization <= 1:
        raise ValueError("KV-cache memory utilization must be in (0, 1]")
    if safety_memory < 0:
        raise ValueError("KV-cache safety memory cannot be negative")
    if not 0 < free_memory <= total_memory:
        raise ValueError("CUDA memory information is invalid")

    currently_used = total_memory - free_memory
    budget_to_utilization_target = (
        int(total_memory * memory_utilization) - currently_used
    )
    budget_after_safety_margin = free_memory - safety_memory
    budget = min(budget_to_utilization_target, budget_after_safety_margin)
    if budget <= 0:
        raise RuntimeError(
            "No VRAM remains for the KV cache after the utilization target and "
            "safety margin"
        )
    return budget


def _resolve_execution_dtype(execution_dtype, device):
    if execution_dtype not in SUPPORTED_DTYPES:
        choices = ", ".join(SUPPORTED_DTYPES)
        raise ValueError(f"Unknown dtype '{execution_dtype}'. Choose one of: {choices}")

    resolved = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[execution_dtype]
    if device.type == "cpu" and resolved == torch.float16:
        raise ValueError("float16 execution requires a CUDA device")
    if (
        device.type == "cuda"
        and resolved == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("This CUDA device does not support bfloat16 execution")
    return resolved


def _load_model(
    model_name,
    device,
    execution_dtype,
    gpt2_model_id,
    distilgpt2_model_id,
):
    if model_name == REFERENCE_MODEL:
        tokenizer = ByteTokenizer()
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            head_dim=16,
            mlp_hidden_size=256,
            max_sequence_length=128,
        )
        model = PagedDecoderLM(config)
        default_kv_cache_memory = 32 * 1024 * 1024
    elif model_name in (DISTILGPT2_MODEL, GPT2_MODEL):
        model_id = (
            distilgpt2_model_id
            if model_name == DISTILGPT2_MODEL
            else gpt2_model_id
        )
        model, tokenizer = load_gpt2_pretrained(model_id)
        config = model.config
        default_kv_cache_memory = (
            1024 * 1024 * 1024 if device.type == "cuda" else 256 * 1024 * 1024
        )
    else:
        choices = ", ".join(SUPPORTED_MODELS)
        raise ValueError(f"Unknown model '{model_name}'. Choose one of: {choices}")

    model = model.to(device=device, dtype=execution_dtype).eval()
    return model, tokenizer, config, default_kv_cache_memory


def build_scheduler(
    model_name=REFERENCE_MODEL,
    gpt2_model_id=DEFAULT_GPT2_MODEL_ID,
    distilgpt2_model_id=DEFAULT_DISTILGPT2_MODEL_ID,
    scheduling_strategy=ORCA_STRATEGY,
    prefill_chunk_size=16,
    max_batch_size=None,
    kv_cache_memory_mb=None,
    kv_cache_memory_utilization=DEFAULT_CUDA_KV_MEMORY_UTILIZATION,
    kv_cache_safety_mb=DEFAULT_CUDA_KV_SAFETY_MB,
    execution_dtype="float32",
    decode_attention_backend="torch",
    kv_cache_dtype="model",
    enable_cuda_graphs=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if decode_attention_backend not in SUPPORTED_DECODE_ATTENTION_BACKENDS:
        choices = ", ".join(SUPPORTED_DECODE_ATTENTION_BACKENDS)
        raise ValueError(
            f"Unknown decode attention backend '{decode_attention_backend}'. "
            f"Choose one of: {choices}"
        )
    if decode_attention_backend == "triton" and device.type != "cuda":
        raise ValueError("The Triton decode attention backend requires CUDA")
    if (
        decode_attention_backend == "triton"
        and importlib.util.find_spec("triton") is None
    ):
        raise ImportError("The Triton decode attention backend requires triton")
    if kv_cache_dtype not in SUPPORTED_KV_CACHE_DTYPES:
        choices = ", ".join(SUPPORTED_KV_CACHE_DTYPES)
        raise ValueError(f"Unknown KV cache dtype '{kv_cache_dtype}'. Choose one of: {choices}")

    resolved_dtype = _resolve_execution_dtype(execution_dtype, device)
    before_model_memory = cuda_memory_snapshot(device)
    model, tokenizer, config, default_kv_cache_memory = _load_model(
        model_name,
        device,
        resolved_dtype,
        gpt2_model_id,
        distilgpt2_model_id,
    )
    model.enable_cuda_graphs = enable_cuda_graphs
    after_model_memory = cuda_memory_snapshot(device)
    if max_batch_size is None:
        max_batch_size = 32 if device.type == "cuda" else 4
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    memory_budget_source = "model_default"
    cuda_budget_snapshot = None
    if (
        kv_cache_memory_mb is None
        and device.type == "cuda"
        and model_name in (DISTILGPT2_MODEL, GPT2_MODEL)
    ):
        torch.cuda.synchronize(device)
        free_memory, total_memory = torch.cuda.mem_get_info(device)
        kv_cache_memory = calculate_cuda_kv_cache_budget(
            free_memory=free_memory,
            total_memory=total_memory,
            memory_utilization=kv_cache_memory_utilization,
            safety_memory=kv_cache_safety_mb * MIB,
        )
        memory_budget_source = "cuda_auto"
        cuda_budget_snapshot = {
            "free_bytes_before_kv": free_memory,
            "total_bytes": total_memory,
            "target_utilization": kv_cache_memory_utilization,
            "safety_bytes": kv_cache_safety_mb * MIB,
        }
    elif kv_cache_memory_mb is None:
        kv_cache_memory = default_kv_cache_memory
    else:
        if kv_cache_memory_mb <= 0:
            raise ValueError("kv_cache_memory_mb must be positive")
        kv_cache_memory = kv_cache_memory_mb * MIB
        memory_budget_source = "explicit"

    kv_manager = KVCacheManager(
        block_size=16,
        total_memory=kv_cache_memory,
        tensor_dtype=next(model.parameters()).dtype,
        device=device,
        num_layers=config.num_layers,
        num_kv_heads=config.num_heads,
        head_dim=config.head_dim,
        cache_dtype=(torch.int8 if kv_cache_dtype == "int8" else resolved_dtype),
    )
    after_kv_memory = cuda_memory_snapshot(device)
    kv_manager.memory_budget_source = memory_budget_source
    kv_manager.cuda_memory_snapshot = cuda_budget_snapshot
    kv_manager.model_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    kv_manager.model_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    kv_manager.model_buffer_bytes = sum(
        buffer.numel() * buffer.element_size() for buffer in model.buffers()
    )
    kv_manager.cuda_allocator_snapshots = {
        "before_model": before_model_memory,
        "after_model": after_model_memory,
        "after_kv_cache": after_kv_memory,
    }
    paged_attention = PagedAttention(
        kv_manager=kv_manager,
        decode_attention_backend=decode_attention_backend,
    )
    return ContinuousBatchScheduler(
        model_engine=model,
        max_batch_size=max_batch_size,
        tokenizer=tokenizer,
        kv_manager=kv_manager,
        paged_attn_manager=paged_attention,
        scheduling_strategy=scheduling_strategy,
        prefill_chunk_size=prefill_chunk_size,
    )


def main():
    parser = argparse.ArgumentParser(description="Run the PagedServe inference engine")
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default=REFERENCE_MODEL,
        help="model backend to run",
    )
    parser.add_argument(
        "--strategy",
        choices=SUPPORTED_SCHEDULING_STRATEGIES,
        default=ORCA_STRATEGY,
        help="request scheduling strategy",
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=16,
        help="prompt tokens per Sarathi prefill chunk",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        help="maximum requests per iteration (default: 32 on CUDA, 4 on CPU)",
    )
    parser.add_argument(
        "--kv-cache-memory-mb",
        type=int,
        help="explicit KV-cache budget in MiB; CUDA GPT models default to auto",
    )
    parser.add_argument(
        "--kv-cache-memory-utilization",
        type=float,
        default=DEFAULT_CUDA_KV_MEMORY_UTILIZATION,
        help="target total GPU memory utilization for automatic KV sizing",
    )
    parser.add_argument(
        "--kv-cache-safety-mb",
        type=int,
        default=DEFAULT_CUDA_KV_SAFETY_MB,
        help="VRAM retained for activations and temporary attention buffers",
    )
    parser.add_argument(
        "--dtype",
        choices=SUPPORTED_DTYPES,
        default="float32",
        help="execution dtype; float16/bfloat16 are explicit optimized runs",
    )
    parser.add_argument(
        "--decode-attention-backend",
        choices=SUPPORTED_DECODE_ATTENTION_BACKENDS,
        default="torch",
        help="single-token decode attention implementation",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        choices=SUPPORTED_KV_CACHE_DTYPES,
        default="model",
        help="KV storage dtype; int8 uses per-token/per-head symmetric scales",
    )
    parser.add_argument(
        "--disable-cuda-graphs",
        action="store_true",
        help="disable steady-state Triton decode CUDA-graph replay",
    )
    args = parser.parse_args()

    scheduler = build_scheduler(
        args.model,
        scheduling_strategy=args.strategy,
        prefill_chunk_size=args.prefill_chunk_size,
        max_batch_size=args.max_batch_size,
        kv_cache_memory_mb=args.kv_cache_memory_mb,
        kv_cache_memory_utilization=args.kv_cache_memory_utilization,
        kv_cache_safety_mb=args.kv_cache_safety_mb,
        execution_dtype=args.dtype,
        decode_attention_backend=args.decode_attention_backend,
        kv_cache_dtype=args.kv_cache_dtype,
        enable_cuda_graphs=not args.disable_cuda_graphs,
    )
    first_request = scheduler.add_request("Paged attention", max_new_tokens=8)
    second_request = scheduler.add_request("Orca", max_new_tokens=8)
    results = scheduler.run_until_complete()

    print(f"request {first_request}: {results[first_request]!r}")
    print(f"request {second_request}: {results[second_request]!r}")


if __name__ == "__main__":
    main()
