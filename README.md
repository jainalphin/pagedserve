# PagedServe

## Installation

```bash
git clone https://github.com/jainalphin/pagedserve.git
cd pagedserve
./env.sh
source .venv/bin/activate
```

Benchmark dependencies:

```bash
pip install -r requirements-triton.txt
pip install -r requirements-vllm.txt
```

## Run

```bash
PYTHONPATH=. python main.py --model gpt2

PYTHONPATH=. python main.py \
  --model gpt2 \
  --dtype float16 \
  --decode-attention-backend triton
```

## Benchmark Summary

Run: `20260830T065517Z`, 2x Tesla T4, GPT-2 FP16, 3 trials.

| Metric | PagedServe Triton | vLLM |
| --- | ---: | ---: |
| Best SLO goodput | 117.0 RPS | 109.3 RPS |
| Stable offered load | 120 RPS | 80 RPS |
| Raw overloaded throughput | 146 RPS | 150 RPS |
| Raw output throughput | 7.7k tok/s | 7.9k tok/s |

| Offered RPS | PagedServe SLO goodput | vLLM SLO goodput |
| ---: | ---: | ---: |
| 60 | 58.8 | 58.8 |
| 80 | 79.3 | 77.2 |
| 100 | 99.8 | 94.4 |
| 120 | 117.0 | 109.3 |
| 140 | 101.1 | 104.0 |
| 160 | 0.6 | 0.0 |
| 180 | 0.3 | 0.0 |
| 200 | 0.2 | 0.0 |

## Run Benchmarks

```bash
PYTHONPATH=. python benchmark_decode_backends.py \
  --dtype float16 \
  --batch-sizes 1,8,32 \
  --input-lengths 128,512 \
  --output-length 32 \
  --runs 3

bash run_all_benchmarks.sh \
  --model openai-community/gpt2 \
  --production-only \
  --production-rate 60 \
  --production-rate 80 \
  --production-rate 100 \
  --production-rate 120 \
  --production-rate 140 \
  --production-rate 160 \
  --duration-seconds 600 \
  --repetitions 3
```
