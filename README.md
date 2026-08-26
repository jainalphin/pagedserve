# PagedServe

A PyTorch LLM inference engine with continuous batching and a paged KV cache.

PagedServe combines new prompts and active generation requests in each model iteration. The KV cache is stored in reusable fixed-size blocks, reducing wasted memory and allowing completed requests to release memory immediately.

## Paged decode attention

During single-token decode, each request owns a logical sequence of 16-token KV
pages. Its block table maps logical page `j` to an arbitrary physical page; pages
do not need to be adjacent. The Torch backend gathers those pages into padded,
contiguous K/V tensors and calls SDPA. The Triton backend launches one program per
request and attention head, follows the table inside the kernel, and reads K/V
directly from non-contiguous physical pages. No context-sized K/V tensor is
materialized.

The block table and context-length tensors describe an iteration, not a model
layer. PagedServe builds them once after reserving decode slots and passes the same
tensors through layers 0 through `L-1`.

For each page, the Triton kernel updates online-softmax state. Given previous
state `(m, l, a)` and page scores `s_i` with values `v_i`:

```text
m' = max(m, max_i(s_i))
p_i = exp(s_i - m')
alpha = exp(m - m')
l' = alpha l + sum_i(p_i)
a' = alpha a + sum_i(p_i v_i)
output = a' / l'
```

The query, scores, running maximum, normalization sum, and value accumulator use
FP32 inside the kernel. This prevents long reductions from accumulating in FP16
and keeps rescaling stable when scores have a large dynamic range; the final
output is cast back to the query dtype.

## Supported models

| Model | Weights | Tokenizer |
| --- | --- | --- |
| Reference Transformer | Locally initialized | UTF-8 byte tokenizer |
| DistilGPT-2 | `distilbert/distilgpt2` | Hugging Face GPT-2 tokenizer |
| GPT-2 | `openai-community/gpt2` | Hugging Face GPT-2 tokenizer |


## Installation

PagedServe requires Python 3.12.

```bash
git clone https://github.com/jainalphin/pagedserve.git
cd pagedserve
./env.sh
source .venv/bin/activate
```

The first run of each pretrained model downloads and caches its weights from Hugging Face. CUDA is used automatically when available; otherwise, the model runs on CPU.

Triton is optional and requires an NVIDIA CUDA environment:

```bash
pip install -r requirements-triton.txt
PYTHONPATH=. python main.py \
  --model gpt2 --dtype float16 \
  --decode-attention-backend triton
```

Supported Triton query/output dtypes are FP16, BF16, and FP32. Floating-point KV
storage must match the query dtype. Head dimensions up to 256 and power-of-two KV
page sizes are supported; the engine uses 16-token pages. Triton decode is
CUDA-only, single-device self-attention decode, not prefill or training.

## Run the web interface

```bash
PYTHONPATH=. python -m streamlit run app.py
```

Select Reference Transformer, DistilGPT-2, or GPT-2 in the interface.

## Run from the command line

Reference Transformer:

```bash
PYTHONPATH=. python main.py
```

GPT-2:

```bash
PYTHONPATH=. python main.py --model gpt2
```

On CUDA, GPT-2 and DistilGPT-2 automatically size the paged KV cache after model
loading. The default targets 90% total device utilization while retaining 3 GiB
for activations and temporary attention buffers. An explicit
`--kv-cache-memory-mb` overrides automatic sizing. The startup benchmark reports
model bytes, KV bytes, token capacity, and maximum-length request capacity; free
VRAM is never assumed to be safely usable in full.

DistilGPT-2:

```bash
PYTHONPATH=. python main.py --model distilgpt2
```

## Scheduling strategies

Orca-style iteration scheduling is the default. A simplified Sarathi-style strategy is also available:

```bash
PYTHONPATH=. python main.py \
  --model distilgpt2 \
  --strategy sarathi \
  --prefill-chunk-size 128
```

The Sarathi strategy admits at most one prompt chunk per iteration and fills the remaining request slots with active decodes. Later chunks attend to earlier chunks through the paged KV cache, and the scheduler emits the first generated token only after the final prompt chunk. This follows the chunked-prefill and decode-maximal batching policy from [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal).

This is a single-device educational implementation, not the complete optimized Sarathi-Serve runtime. Chunking is intended to bound interruptions to active decodes when long prompts arrive; it can increase prompt TTFT and reduce closed-batch throughput, especially on CPU.

## Run in Notebook

Install the project in a notebook cell:

```python
!git clone https://github.com/jainalphin/pagedserve.git
%cd pagedserve
%pip install -r requirements.txt
```

Restart the notebook kernel after installation, then run GPT-2:

```python
from main import GPT2_MODEL, build_scheduler

scheduler = build_scheduler(GPT2_MODEL)
request_id = scheduler.add_request(
    "Once upon a time",
    max_new_tokens=30,
)

results = scheduler.run_until_complete()
print(results[request_id])
```

Multiple requests can be processed with continuous batching:

```python
first = scheduler.add_request("Artificial intelligence is", max_new_tokens=20)
second = scheduler.add_request("The future of computing", max_new_tokens=20)

results = scheduler.run_until_complete()
print(results[first])
print(results[second])
```

## Run benchmarks

### Correctness and kernel microbenchmark

The CUDA correctness gate compares Torch gather+SDPA with Triton for FP16 and
FP32; contexts 1, 15, 16, 17, 31, 32, 33, 128, 512, and 1024; batches 1, 8, and
32; randomized non-contiguous physical pages; and layers 0, 1, and last. It also
checks finite outputs, rejects NaN/Inf queries, and rejects zero/oversized lengths
and out-of-range page IDs.

```bash
PYTHONPATH=. python -m pytest -q testing/test_triton_correctness_matrix.py

PYTHONPATH=. python benchmark_paged_attention.py \
  --dtype float16 --warmup 25 --iterations 100 \
  --json-output artifacts/paged-attention-fp16.json

PYTHONPATH=. python benchmark_paged_attention.py \
  --dtype float32 --warmup 25 --iterations 100 \
  --json-output artifacts/paged-attention-fp32.json
```

The microbenchmark reports latency in microseconds and peak temporary allocated
GPU memory for every batch × context cell. Both paths share randomized block
tables, and every measured cell is correctness-checked first.

### Identical GPT-2 end-to-end comparison

This driver runs each configuration in a fresh process at least three times. Both
backends receive the same model, token-ID prompts, greedy decoding, burst arrival
trace, scheduler, batch size, input length, and output length. Each published row
is the median of isolated runs.

```bash
PYTHONPATH=. python benchmark_decode_backends.py \
  --dtype float16 \
  --batch-sizes 1,8,32 \
  --input-lengths 128,512 \
  --output-length 32 \
  --runs 3 \
  --output artifacts/gpt2-decode-backends.json
```

| Backend | Batch | Input | Output tok/s (median) | TPOT ms (median) | TTFT ms (median) | Peak GPU MiB (median) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Torch | 1/8/32 | 128/512 | Not measured here | Not measured here | Not measured here | Not measured here |
| Triton | 1/8/32 | 128/512 | Not measured here | Not measured here | Not measured here | Not measured here |

This checkout was developed in a CPU-only environment, so CUDA numbers and a
trace are deliberately not fabricated. The commands write all raw runs and the
median table; publish the generated JSON alongside filled results, including
cells where Triton loses.

### Profiling evidence

Capture one PyTorch Profiler Chrome trace with named regions for metadata
construction, Torch KV-gather allocations plus SDPA, and Triton's direct reads:

```bash
PYTHONPATH=. python profile_paged_attention.py \
  --batch-size 8 --context-length 512 --iterations 10 \
  --trace artifacts/paged-attention-trace.json
```

Open it in Perfetto or `chrome://tracing`.
`torch_kv_gather_allocations_plus_sdpa` exposes context-sized allocation/copy
kernels, `triton_direct_noncontiguous_paged_reads` contains the fused kernel, and
`metadata_construction_once_per_iteration` exposes metadata CPU/device-copy cost.
The trace also provides launch duration and the dominant kernel bottlenecks.

### INT8 KV-cache extension

INT8 is the only advanced extension. Each K and V vector uses symmetric
per-token/per-head quantization, `q = round(clamp(x / scale, -127, 127))`, with an
FP32 scale. Triton loads INT8 values from the paged pool and applies the scale in
registers before the dot product or weighted-value reduction. Torch dequantizes
after gather as the reference path.

```bash
PYTHONPATH=. python main.py \
  --model gpt2 --dtype float16 \
  --decode-attention-backend triton \
  --kv-cache-dtype int8

PYTHONPATH=. python evaluate_int8_kv.py \
  --tokens 128 --prefix-tokens 32 --iterations 100 \
  --output artifacts/int8-kv-evaluation.json
```

The evaluation records usable token capacity under the same byte budget, FP16
versus INT8 Triton decode latency, mean/maximum logit error, and deterministic
proxy-perplexity delta. The included text is an engineering proxy, not a named
corpus benchmark.

Benchmark every supported model:

```bash
PYTHONPATH=. python benchmark.py
```

Benchmark one model with custom settings:

```bash
PYTHONPATH=. python benchmark.py --model gpt2 --batch-size 4 --max-new-tokens 32 --runs 5
```

Use FP32 for the baseline. FP16 is an explicit, separately reported optimization:

```bash
PYTHONPATH=. python benchmark.py \
  --model gpt2 \
  --dtype float16 \
  --max-batch-size 32 \
  --batch-size 32
```

Compare both schedulers, or measure prefill-induced decode stalls:

```bash
PYTHONPATH=. python benchmark.py \
  --model distilgpt2 \
  --strategy orca \
  --strategy sarathi \
  --prefill-chunk-size 128

PYTHONPATH=. python strategy_benchmark.py --model distilgpt2
```

The script reports model and system metadata, loading time, median and p95 generation latency, time to first token, token throughput, exact token counts, and peak CUDA memory. Pass `--json-output PATH` to retain every raw run. New entries in `SUPPORTED_MODELS` are benchmarked automatically.

See [BENCHMARKS.md](BENCHMARKS.md) for measured Apple M4 CPU results for all included models and the exact reproduction command.

## Compare Hugging Face, PagedServe, and vLLM

`comparison_benchmark.py` uses one common open-loop load generator for all three
engines. It creates identical deterministic token-ID prompts and arrival times,
requests an exact output-token count with greedy decoding, and records raw per-request
TTFT, TPOT, ITL, end-to-end latency, achieved RPS, output throughput, failures, GPU
activity, memory, and power. Each engine runs in a separate process.

- `hf`: sequential FCFS Hugging Face baseline without continuous batching.
- `pagedserve`: this repository with Orca or Sarathi scheduling.
- `vllm`: vLLM's asynchronous continuous-batching engine.

The quick profile tests 128-token and 900-token prompts. The extreme profile tests
16, 128, 512, and 900 input tokens at offered rates from 1 to 32 RPS plus an
all-at-once burst. GPT-2 has a 1,024-token context limit, so the 900-input/64-output
case is close to its architectural maximum without exceeding it.

vLLM recommends a compatible environment because its wheel bundles compiled
CUDA/PyTorch components. Install it into the notebook kernel's exact interpreter:

```python
import sys

%pip install -q uv
!uv pip install --python {sys.executable} vllm --torch-backend=auto
!{sys.executable} -c "import vllm; print(vllm.__version__)"
```

Then clone this repository and run the correctness gate before benchmarking:

```bash
PYTHONPATH=. python -m pytest -q -p no:cacheprovider
```

Run vLLM alone first as a smoke test. This uses the same exact prompts, arrival
trace, decoding settings, metrics, and telemetry format as the other engines:

```bash
PYTHONPATH=. python comparison_benchmark.py \
  --engine vllm \
  --dtype float16 \
  --input-length 128 \
  --output-length 32 \
  --num-requests 30 \
  --request-rate 1 \
  --request-rate 4 \
  --request-rate 16 \
  --request-rate inf \
  --max-batch-size 64 \
  --gpu-memory-utilization 0.8 \
  --json-output /tmp/vllm_kaggle_smoke.json
```

The vLLM backend uses `AsyncLLM` rather than the synchronous convenience API, so
requests can arrive while earlier requests are decoding. It disables prefix
caching and uses greedy decoding with EOS ignored to generate exactly the requested
number of tokens. Do not add `--vllm-enforce-eager` for the measured run unless
CUDA graph initialization fails; eager and graph results are different tracks.

Start with the FP16 quick matrix on NVIDIA GPUs:

```bash
PYTHONPATH=. python run_comparison_matrix.py \
  --profile quick \
  --dtype float16
```

FP32 is an optional numerical control and must remain a separate track rather than
being combined with FP16 results.

## Two-GPU capacity

Two T4s do not provide one unified memory pool. For GPT-2 throughput, use one
independent replica per GPU and split incoming requests evenly. The following
commands run both replicas concurrently and report aggregate RPS plus per-GPU
utilization. Do not set `CUDA_VISIBLE_DEVICES=0` around these commands.

Short-context PagedServe capacity around the 50–60 RPS target:

```bash
PYTHONPATH=. python dual_gpu_capacity_benchmark.py \
  --engine pagedserve \
  --pagedserve-strategy orca \
  --dtype float16 \
  --input-length 128 \
  --output-length 32 \
  --max-batch-size 128 \
  --request-rate 20 \
  --request-rate 30 \
  --request-rate 40 \
  --request-rate 50 \
  --request-rate 60 \
  --request-rate 70 \
  --request-rate 80
```

Run the identical curve for vLLM:

```bash
PYTHONPATH=. python dual_gpu_capacity_benchmark.py \
  --engine vllm \
  --dtype float16 \
  --input-length 128 \
  --output-length 32 \
  --max-batch-size 128 \
  --request-rate 20 \
  --request-rate 30 \
  --request-rate 40 \
  --request-rate 50 \
  --request-rate 60 \
  --request-rate 70 \
  --request-rate 80 \
  --output-dir /tmp/vllm-dual-capacity
```

Run the simple sequential Hugging Face baseline with the same two-replica layout:

```bash
PYTHONPATH=. python dual_gpu_capacity_benchmark.py \
  --engine hf \
  --dtype float16 \
  --input-length 128 \
  --output-length 32 \
  --request-rate 20 \
  --request-rate 30 \
  --request-rate 40 \
  --request-rate 50 \
  --request-rate 60 \
  --request-rate 70 \
  --request-rate 80 \
  --output-dir /tmp/hf-dual-capacity
```

This HF control runs one request at a time on each GPU. It measures two replicated
plain Transformers workers, not a dynamically batched Hugging Face serving stack.

## Run and temporarily publish the full suite

`run_all_benchmarks.sh` runs both GPT-2 and DistilGPT-2 by default across HF,
PagedServe Orca, PagedServe Sarathi, and vLLM. It records environment metadata,
resolved packages, tests, commands, console summaries, raw temporary results, and
SHA-256 checksums. `REPORT.md` is the single readable run summary; the JSON files
under each case are the machine-readable evidence behind it.
With `--push`, it commits those artifacts to a new timestamped branch such as
`benchmark-results/20260818T120000Z`; it never pushes benchmark data to `main`.

Each worker profile records:

- hardware, driver, CUDA, PyTorch, Python, git commit, and the exact command;
- model architecture, parameter count, configured context window, runtime weight
  and buffer bytes, checkpoint files and disk bytes, dtype, and load/warm-up time;
- PagedServe KV bytes, bytes per token/block, total blocks/tokens, maximum-length
  request capacity, memory-budget source, and peak live KV pages per load point;
- vLLM's reported KV-cache bytes, blocks, token capacity, and concurrency when the
  installed vLLM version exposes those initialized cache-config fields;
- CUDA allocator start/end/peak allocated and reserved bytes, plus device-wide
  VRAM before model load, after initialization, after warm-up, and during traffic;
- offered/achieved RPS, successful and failed requests, output tokens/s, SLO
  goodput, client queue delay, engine TTFT after submission, and total
  TTFT/TPOT/ITL/E2E distributions including p50, p95, and p99;
- sampled GPU kernel activity, memory-controller activity, VRAM, power draw, power
  limit, and estimated energy per run, output token, and successful request.

The device-wide `nvidia-smi` memory reading is the cross-engine comparison value.
PyTorch allocator counters are supplementary because an engine may own memory
outside PyTorch's caching allocator. Power-derived energy is an estimate from the
configured sampling interval, not a hardware energy-counter measurement. This
suite is serving-level profiling; it does not claim per-CUDA-kernel timing from
Nsight Systems or Nsight Compute.

Store a fine-grained GitHub token with repository Contents read/write permission as
a private Kaggle secret named `GH_TOKEN`, then expose it without printing it:

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["GH_TOKEN"] = UserSecretsClient().get_secret("GH_TOKEN")
```

Run the full dual-T4 suite and push only after every case succeeds:

```bash
bash run_all_benchmarks.sh \
  --requests-per-replica 120 \
  --push
```

For a production-style capacity test, use mixed request sizes, independent
Poisson arrivals on each replica, rates above the expected 50–60 RPS operating
point, and three cold engine repetitions:

```bash
bash run_all_benchmarks.sh \
  --production-only \
  --production-rate 30 \
  --production-rate 50 \
  --production-rate 60 \
  --production-rate 80 \
  --production-rate 100 \
  --production-rate 120 \
  --requests-per-replica 500 \
  --repetitions 3 \
  --ttft-slo-ms YOUR_TTFT_LIMIT \
  --tpot-slo-ms YOUR_TPOT_LIMIT \
  --e2e-slo-ms YOUR_E2E_LIMIT \
  --push
```

The built-in production-like mix is 50% 128-input/32-output, 30% 384/64,
15% 768/96, and 5% 900/64. These are explicit starting assumptions, not claims
about your users. Replace them with an anonymized production length/arrival trace
before making a final deployment decision.

This track measures the inference workers and two-replica load split. It still
does not include an HTTP gateway, TLS, JSON serialization, request tokenization,
network transit, autoscaling, or multi-host failures. Those require a deployed
end-to-end load test; the report labels this track `production-mix`, not a measured
production service.

Add GPT-2-family checkpoints with repeated `--model` arguments. Supplying any
`--model` replaces the defaults:

```bash
bash run_all_benchmarks.sh \
  --model openai-community/gpt2 \
  --model distilbert/distilgpt2 \
  --model openai-community/gpt2-medium \
  --requests-per-replica 120 \
  --push
```

The current PagedServe converter supports GPT2LMHeadModel-family checkpoints. For
another architecture, benchmark only the compatible HF and vLLM paths:

```bash
bash run_all_benchmarks.sh \
  --model MODEL_ID \
  --engine hf \
  --engine vllm \
  --push
```

At completion the script prints `BENCHMARK_RESULTS_BRANCH=...`. Fetch that branch
for review; after its verified summaries are copied into `BENCHMARKS.md`, delete the
temporary remote branch.

Repeat with `--input-length 900 --output-length 64` for near-maximum GPT-2
context. Add your actual `--ttft-slo-ms`, `--tpot-slo-ms`, and `--e2e-slo-ms`
requirements to measure SLO-qualified goodput. Memory capacity and sustainable RPS
are different: adding KV pages prevents admission failures but cannot increase the
T4's compute throughput.

The matrix writes temporary raw results and logs under `benchmarks/comparison/`,
which Git ignores. Copy only verified environment details and summary tables into
`BENCHMARKS.md`, the single tracked benchmark record. Do not call one rate
"maximum sustainable RPS" without a latency objective; pass `--ttft-slo-ms`,
`--tpot-slo-ms`, and/or `--e2e-slo-ms` to calculate SLO-qualified goodput.
