# Triton paged-decode results

Measured on 2026-08-26 on one NVIDIA Tesla T4 (compute capability 7.5,
15,360 MiB), Python 3.12.13, PyTorch 2.11.0+cu130, CUDA 13.0, and GPT-2
(`openai-community/gpt2`, 124,439,808 parameters). The report recorded source
commit `aa13ef41432342e335cd6affb2bb79405ecb8628`; the worktree was marked dirty
because benchmark artifacts were generated inside it.

The evidence archive SHA-256 is
`d2a476b82289fee735925368f6fc7476a399d701767c18a0f642df95a0f64152`.
Every CUDA correctness test passed: **50 passed in 19.54 seconds**.

## Kernel microbenchmark

Each cell is `Torch gather+SDPA / Triton` latency in microseconds. Measurements
use 25 warm-up launches and 100 timed iterations, 12 heads, head dimension 64,
16-token pages, and randomized non-contiguous physical blocks. Every cell was
numerically checked before timing.

### FP16 latency (µs)

| Batch | 1 | 15 | 16 | 17 | 31 | 32 | 33 | 128 | 512 | 1024 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 286.79 / 108.52 | 290.34 / 113.17 | 284.25 / 128.66 | 175.68 / 56.62 | 173.42 / 58.12 | 167.90 / 57.89 | 180.88 / 60.29 | 189.56 / 59.87 | 178.45 / 110.80 | 267.75 / 217.07 |
| 8 | 130.77 / 57.48 | 130.72 / 54.38 | 126.18 / 54.40 | 182.48 / 54.99 | 163.49 / 54.50 | 165.89 / 54.81 | 163.94 / 55.36 | 161.00 / 55.66 | 405.45 / 186.95 | 776.34 / 366.95 |
| 32 | 131.44 / 60.73 | 136.13 / 54.77 | 127.61 / 55.17 | 163.14 / 55.40 | 167.96 / 54.84 | 160.02 / 55.35 | 164.39 / 55.13 | 321.15 / 100.89 | 1219.81 / 420.42 | 2436.67 / 748.52 |

Triton won all 30 FP16 cells. Speedup ranged from 1.23× to 3.32×, with a
2.90× median.

### FP32 latency (µs)

| Batch | 1 | 15 | 16 | 17 | 31 | 32 | 33 | 128 | 512 | 1024 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 138.20 / 56.42 | 131.51 / 55.64 | 133.43 / 61.19 | 167.14 / 55.27 | 173.08 / 56.63 | 166.29 / 58.43 | 166.67 / 56.73 | 161.62 / 55.58 | 300.97 / 100.91 | 566.66 / 164.17 |
| 8 | 157.79 / 56.16 | 135.03 / 56.93 | 142.47 / 55.11 | 166.43 / 56.11 | 165.85 / 57.83 | 162.87 / 55.57 | 170.22 / 56.81 | 237.93 / 54.85 | 873.16 / 110.13 | 1551.04 / 196.18 |
| 32 | 137.38 / 56.61 | 132.85 / 55.87 | 131.13 / 55.67 | 214.36 / 56.80 | 220.84 / 56.54 | 220.98 / 56.39 | 281.38 / 56.37 | 694.82 / 105.45 | 2665.78 / 402.73 | 5331.52 / 786.44 |

Triton won all 30 FP32 cells. Speedup ranged from 2.18× to 7.93×, with a
2.97× median.

### Temporary GPU allocation

At batch 32 and context 1,024, FP16 Torch required 144 MiB of temporary
allocation versus 0.0469 MiB for Triton. FP32 required 288 MiB versus
0.0938 MiB. Across the matrix the measured Torch-to-Triton temporary-memory
ratio ranged from 33× to 3,072×. Triton's reported allocation is its output;
it does not materialize gathered context-sized K/V tensors.

## End-to-end GPT-2

Each row is the median of three fresh-process runs. Both backends used identical
FP16 model weights, deterministic token-ID prompts, greedy decoding, Orca
scheduling, burst arrivals, 32 output tokens, and an automatically sized paged
KV pool.

| Backend | Batch | Input | Output tok/s | TPOT ms | TTFT ms | Peak GPU MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Torch | 1 | 128 | 76.51 | 13.192 | 9.292 | 11742.8 |
| Triton | 1 | 128 | 100.54 | 9.976 | 9.014 | 11742.8 |
| Torch | 1 | 512 | 71.11 | 14.004 | 16.026 | 11770.3 |
| Triton | 1 | 512 | 86.99 | 11.345 | 16.158 | 11770.3 |
| Torch | 8 | 128 | 557.84 | 13.980 | 25.523 | 11787.0 |
| Triton | 8 | 128 | 678.86 | 11.327 | 25.759 | 11787.0 |
| Torch | 8 | 512 | 403.33 | 17.861 | 83.627 | 11943.5 |
| Triton | 8 | 512 | 546.79 | 12.392 | 84.032 | 11943.5 |
| Torch | 32 | 128 | 1363.55 | 21.658 | 81.141 | 11943.5 |
| Triton | 32 | 128 | 1882.43 | 14.895 | 83.345 | 11943.5 |
| Torch | 32 | 512 | 651.57 | 41.378 | 289.802 | 12573.7 |
| Triton | 32 | 512 | 1284.72 | 16.308 | 291.271 | 12573.7 |

Triton improved output throughput in every configuration by 1.22×–1.97× and
reduced TPOT by 19.0%–60.6%. TTFT was nearly unchanged: between 3.0% lower and
2.7% higher. End-to-end peak allocated memory was unchanged because the fixed,
preallocated 11.2 GiB KV pool dominated the allocator peak; the kernel-level
temporary-memory result above isolates the gather saving.

## Profiler evidence

The profiled workload was FP16, batch 8, context 512, for 10 calls per backend.
The trace showed:

- Torch `aten::index` allocated 120 MiB cumulatively across 20 K/V gathers and
  launched vectorized gather plus device-to-device copy kernels.
- The Torch SDPA CUTLASS kernel averaged 163.08 µs, in addition to gather,
  masking, and copy work.
- Triton's `_paged_decode_attention_kernel` averaged 246.24 µs as one direct
  paged-read kernel, with no context-sized K/V allocation.
- Metadata was constructed once in the trace, not once per layer. Its profiled
  CPU range was 4.23 ms; profiler stack collection makes this a diagnostic
  measurement rather than a steady-state microbenchmark number.

The Chrome trace SHA-256 is
`896568e915881355a7b51539419f57f1216c13017ee7f09c4a5109b506b3bffa`.

## INT8 KV cache

Under the same 256 MiB budget, FP16 stored 7,280 tokens at 36,864 bytes/token;
INT8 plus FP32 scales stored 13,696 tokens at 19,584 bytes/token, a **1.88×**
capacity improvement.

INT8 did not win latency in this workload: FP16 Triton decode attention took
66.32 µs and INT8 took 68.68 µs, a 3.6% regression. On the deterministic proxy
text, mean/max absolute logit error was 1.0566/32.75 and perplexity changed from
2.08985 to 2.09425 (+0.00440, +0.21%). The small proxy-perplexity change is
encouraging, but the maximum logit error warrants evaluation on a standard,
larger corpus before making a production-quality claim.

## Ten-minute production-style comparison

> **Historical result notice:** this comparison predates benchmark schema 4.
> PagedServe and vLLM did not share a single total-device memory-utilization
> target, and aggregate throughput used the former per-replica rate-sum method.
> Do not use this table for cross-engine VRAM efficiency or capacity-superiority
> claims; re-run it with the current benchmark first.

Three independent trials per engine used two Tesla T4 replicas, FP16 GPT-2,
Poisson arrivals at 60 total offered RPS, and 18,000 requests per replica. Each
trial therefore contained 36,000 requests and approximately ten minutes of
arrivals. The request mix was 50% 128-input/32-output, 30% 384/64, 15% 768/96,
and 5% 900/64. The table reports the median of the three trial-level statistics.

| Engine | Achieved req/s | Output tok/s | TTFT p50/p95/p99 ms | TPOT p50/p95/p99 ms | E2E p50/p95/p99 ms | Failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PagedServe Torch | 20.13 | 1063.86 | 582822.83 / 1122609.75 / 1184949.67 | 238.80 / 248.36 / 249.68 | 595657.66 / 1135046.22 / 1198030.65 | 0 |
| PagedServe Triton | 59.93 | 3153.70 | 35.50 / 54.72 / 68.74 | 21.30 / 26.02 / 28.97 | 1080.82 / 2180.34 / 2456.97 | 0 |
| vLLM | 60.02 | 3158.51 | 26.20 / 44.03 / 54.28 | 4.79 / 6.44 / 7.57 | 262.24 / 519.90 / 605.77 | 0 |

Torch was overloaded: it accepted the full arrival trace but drained at only
20.13 requests/s, leaving roughly 12,000 outstanding requests per GPU at peak.
Its minute-scale TTFT and E2E percentiles are queue-saturation behavior, not a
steady-state kernel-latency comparison.

Triton sustained the offered load and improved output throughput over Torch by
2.96× while reducing p95 TPOT by 89.5%. At this offered rate it delivered 99.85%
of vLLM's output throughput, but vLLM retained materially better latency: Triton
p95 TTFT was 24.3% higher, p95 TPOT was 4.04× vLLM's, and p95 E2E was 4.19×
vLLM's.

Using `(Triton - Torch) / (vLLM - Torch)`, Triton closed 99.77% of the observed
output-throughput gap at 60 RPS. This is a demand-limited result: Triton and vLLM
both processed essentially all offered requests, so it does **not** show that
their maximum capacities are equal. A production-mix rate sweep above 60 RPS is
required to locate each engine's saturation point and calculate capacity-gap
closure.

The comparison archive and every internal result checksum validated successfully.
Archive SHA-256:
`d84e392b273000a03ee7e45cfb83c0c3dae32f070fd4f4270a76492e4484b004`.

## Reproduce

Use the correctness, microbenchmark, end-to-end, profiler, and INT8 commands in
the main [README](README.md). The benchmark programs enforce at least 100 timed
kernel iterations and at least three end-to-end repetitions.
