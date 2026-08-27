"""Profile complete PagedServe GPT-2 decode iterations, not only attention."""

import argparse
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from main import GPT2_MODEL, build_scheduler


def install_module_scope(module, label, handles, active_scopes):
    """Label eager module calls in the trace without changing serving code."""

    def enter(current_module, _inputs):
        scope = record_function(label)
        scope.__enter__()
        active_scopes[id(current_module)] = scope

    def exit_scope(current_module, _inputs, output):
        active_scopes.pop(id(current_module)).__exit__(None, None, None)
        return output

    handles.append(module.register_forward_pre_hook(enter))
    handles.append(module.register_forward_hook(exit_scope))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="openai-community/gpt2")
    parser.add_argument("--backend", choices=("torch", "triton"), default="triton")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--warmup-decodes", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Full-model decode profiling requires CUDA")
    if args.batch_size <= 0 or args.context_length <= 0:
        parser.error("batch size and context length must be positive")
    if args.warmup_decodes < 0 or args.iterations <= 0:
        parser.error("warmup decodes cannot be negative and iterations must be positive")

    scheduler = build_scheduler(
        GPT2_MODEL,
        gpt2_model_id=args.model_id,
        scheduling_strategy="orca",
        max_batch_size=args.batch_size,
        execution_dtype=args.dtype,
        decode_attention_backend=args.backend,
    )
    scheduler.eos_token_id = None
    model_config = scheduler.model_engine.config
    generated_tokens = args.warmup_decodes + args.iterations + 1
    if args.context_length + generated_tokens > model_config.max_sequence_length:
        parser.error("context plus warmup/profile tokens exceeds the model context window")

    # Exact token IDs make Torch and Triton traces reproducible and avoid timing
    # tokenization. One prefill step moves every request into decode state.
    prompts = [
        [2 + ((request_id * args.context_length + index) % (model_config.vocab_size - 2))
         for index in range(args.context_length)]
        for request_id in range(args.batch_size)
    ]
    for prompt in prompts:
        scheduler.add_token_request(prompt, max_new_tokens=generated_tokens)
    scheduler.step()
    for _ in range(args.warmup_decodes):
        scheduler.step()
    torch.cuda.synchronize()

    original_metadata_builder = scheduler.kv_manager.build_decode_metadata
    original_paged_attention = scheduler.paged_attn_manager.forward_batch

    def profiled_metadata(*metadata_args, **metadata_kwargs):
        with record_function("decode_metadata_construction"):
            return original_metadata_builder(*metadata_args, **metadata_kwargs)

    def profiled_attention(*attention_args, **attention_kwargs):
        with record_function(f"{args.backend}_paged_decode_attention"):
            return original_paged_attention(*attention_args, **attention_kwargs)

    scheduler.kv_manager.build_decode_metadata = profiled_metadata
    scheduler.paged_attn_manager.forward_batch = profiled_attention
    handles = []
    active_scopes = {}
    for layer in scheduler.model_engine.layers:
        install_module_scope(
            layer.self_attn.qkv_linear,
            "packed_qkv_projection",
            handles,
            active_scopes,
        )
        install_module_scope(
            layer.self_attn.output_linear,
            "attention_output_projection",
            handles,
            active_scopes,
        )
        install_module_scope(
            layer.mlp_up_proj,
            "mlp_up_projection",
            handles,
            active_scopes,
        )
        install_module_scope(
            layer.mlp_down_proj,
            "mlp_down_projection",
            handles,
            active_scopes,
        )

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,
    ) as profiler:
        for _ in range(args.iterations):
            with record_function("scheduler_plus_full_gpt2_decode_iteration"):
                scheduler.step()
    torch.cuda.synchronize()

    for handle in handles:
        handle.remove()

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(args.trace))
    print(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=40))
    print(profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=40))
    print(f"Chrome trace: {args.trace}")


if __name__ == "__main__":
    main()
