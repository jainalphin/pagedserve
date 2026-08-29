#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_ROOT="benchmark-results/$RUN_ID"
RESULT_BRANCH=""
PUSH_RESULTS=false
RUN_TESTS=true
RUN_SHORT=true
RUN_LONG=true
RUN_PRODUCTION=false
REPETITIONS=3
REQUESTS_PER_REPLICA=120
DURATION_SECONDS=""
MAX_BATCH_SIZE=128
KV_CACHE_SAFETY_MB=3072
GPU_MEMORY_UTILIZATION=0.8
DECODE_ATTENTION_BACKEND="torch"
TTFT_SLO_MS=""
TPOT_SLO_MS=""
E2E_SLO_MS=""
MODELS=()
ENGINES=()
SHORT_RATES=()
LONG_RATES=()
PRODUCTION_RATES=()

usage() {
  sed -n '1,120p' <<'USAGE'
Run the complete dual-GPU benchmark suite and optionally push results to a
temporary GitHub branch.

Usage:
  bash run_all_benchmarks.sh [options]

Options:
  --model MODEL_ID              Repeat for multiple checkpoints.
  --engine ENGINE               hf, pagedserve-orca, pagedserve-sarathi, vllm.
                                Repeat to select multiple engines.
  --requests-per-replica N      Requests per GPU and offered rate (default: 120).
  --duration-seconds SECONDS    Generate arrivals for this duration per rate;
                                overrides the request count in timed sweeps.
  --max-batch-size N            Maximum sequences per worker (default: 128).
  --kv-cache-safety-mb N        VRAM retained outside PagedServe KV cache
                                (default: 3072).
  --gpu-memory-utilization F    Shared total-device memory target for PagedServe
                                and vLLM (default: 0.8).
  --decode-attention-backend B  PagedServe decode backend: torch or triton.
  --short-rate RPS              Repeat to replace short-context rates.
  --long-rate RPS               Repeat to replace long-context rates.
  --production-rate RPS         Repeat to replace mixed production-like rates.
  --repetitions N               Repeat every case with a new deterministic seed
                                (default: 3; use 1 only for smoke tests).
  --ttft-slo-ms MS              Optional per-request TTFT objective.
  --tpot-slo-ms MS              Optional per-token latency objective.
  --e2e-slo-ms MS               Optional per-request E2E objective.
  --short-only                  Skip the 900+64 workload.
  --long-only                   Skip the 128+32 workload.
  --production-only             Run only Poisson mixed-shape traffic.
  --include-production          Add Poisson mixed-shape traffic to fixed cases.
  --skip-tests                  Do not run pytest before benchmarking.
  --push                        Commit results to a new temporary remote branch.
  --branch NAME                 Override the temporary result branch name.
  -h, --help                    Show this help.

Defaults:
  Models: openai-community/gpt2 and distilbert/distilgpt2
  Engines: HF, PagedServe Orca, PagedServe Sarathi, and vLLM
  Short rates: 30, 40, 50, 60, 70, 80 RPS
  Long rates: 4, 8, 12, 16, 20, 30 RPS
  Production-like rates: 30, 50, 60, 80, 100, 120, 140, 160 RPS

Production-like mix: 50% 128+32, 30% 384+64, 15% 768+96, and
5% 900+64 tokens with independent Poisson arrivals on each replica.

Only GPT2LMHeadModel-family checkpoints can currently be compared across every
engine. For another architecture, select only --engine hf and --engine vllm.
USAGE
}

require_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      require_value "$@"
      MODELS+=("$2")
      shift 2
      ;;
    --engine)
      require_value "$@"
      case "$2" in
        hf|pagedserve-orca|pagedserve-sarathi|vllm) ;;
        *) echo "Unknown engine: $2" >&2; exit 2 ;;
      esac
      ENGINES+=("$2")
      shift 2
      ;;
    --requests-per-replica)
      require_value "$@"
      REQUESTS_PER_REPLICA="$2"
      shift 2
      ;;
    --duration-seconds)
      require_value "$@"
      DURATION_SECONDS="$2"
      shift 2
      ;;
    --max-batch-size)
      require_value "$@"
      MAX_BATCH_SIZE="$2"
      shift 2
      ;;
    --kv-cache-safety-mb)
      require_value "$@"
      KV_CACHE_SAFETY_MB="$2"
      shift 2
      ;;
    --gpu-memory-utilization)
      require_value "$@"
      GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    --decode-attention-backend)
      require_value "$@"
      case "$2" in
        torch|triton) ;;
        *) echo "Unknown decode attention backend: $2" >&2; exit 2 ;;
      esac
      DECODE_ATTENTION_BACKEND="$2"
      shift 2
      ;;
    --short-rate)
      require_value "$@"
      SHORT_RATES+=("$2")
      shift 2
      ;;
    --long-rate)
      require_value "$@"
      LONG_RATES+=("$2")
      shift 2
      ;;
    --production-rate)
      require_value "$@"
      PRODUCTION_RATES+=("$2")
      shift 2
      ;;
    --repetitions)
      require_value "$@"
      REPETITIONS="$2"
      shift 2
      ;;
    --ttft-slo-ms)
      require_value "$@"
      TTFT_SLO_MS="$2"
      shift 2
      ;;
    --tpot-slo-ms)
      require_value "$@"
      TPOT_SLO_MS="$2"
      shift 2
      ;;
    --e2e-slo-ms)
      require_value "$@"
      E2E_SLO_MS="$2"
      shift 2
      ;;
    --short-only)
      RUN_SHORT=true
      RUN_LONG=false
      shift
      ;;
    --long-only)
      RUN_SHORT=false
      RUN_LONG=true
      shift
      ;;
    --production-only)
      RUN_SHORT=false
      RUN_LONG=false
      RUN_PRODUCTION=true
      shift
      ;;
    --include-production)
      RUN_PRODUCTION=true
      shift
      ;;
    --skip-tests)
      RUN_TESTS=false
      shift
      ;;
    --push)
      PUSH_RESULTS=true
      shift
      ;;
    --branch)
      require_value "$@"
      RESULT_BRANCH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODELS=("openai-community/gpt2" "distilbert/distilgpt2")
fi
if [[ ${#ENGINES[@]} -eq 0 ]]; then
  ENGINES=("hf" "pagedserve-orca" "pagedserve-sarathi" "vllm")
fi
if [[ ${#SHORT_RATES[@]} -eq 0 ]]; then
  SHORT_RATES=(30 40 50 60 70 80)
fi
if [[ ${#LONG_RATES[@]} -eq 0 ]]; then
  LONG_RATES=(4 8 12 16 20 30)
fi
if [[ ${#PRODUCTION_RATES[@]} -eq 0 ]]; then
  PRODUCTION_RATES=(30 50 60 80 100 120 140 160)
fi

if [[ ! "$REQUESTS_PER_REPLICA" =~ ^[1-9][0-9]*$ ]]; then
  echo "--requests-per-replica must be a positive integer" >&2
  exit 2
fi
if [[ -n "$DURATION_SECONDS" ]] && ! awk -v value="$DURATION_SECONDS" \
    'BEGIN { exit !(value + 0 > 0) }'; then
  echo "--duration-seconds must be positive" >&2
  exit 2
fi
if [[ ! "$MAX_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-batch-size must be a positive integer" >&2
  exit 2
fi
if [[ ! "$KV_CACHE_SAFETY_MB" =~ ^[1-9][0-9]*$ ]]; then
  echo "--kv-cache-safety-mb must be a positive integer" >&2
  exit 2
fi
if ! awk -v value="$GPU_MEMORY_UTILIZATION" \
    'BEGIN { exit !(value + 0 > 0 && value + 0 <= 1) }'; then
  echo "--gpu-memory-utilization must be in (0, 1]" >&2
  exit 2
fi
if [[ ! "$REPETITIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--repetitions must be a positive integer" >&2
  exit 2
fi
if [[ -e "$RESULT_ROOT" ]]; then
  echo "Result directory already exists: $RESULT_ROOT" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked files are modified. Commit or restore them before benchmarking." >&2
  exit 1
fi

mkdir -p "$RESULT_ROOT"

{
  echo "run_id=$RUN_ID"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "git_branch=$(git symbolic-ref --short HEAD 2>/dev/null || echo detached)"
  echo "python=$($PYTHON_BIN --version 2>&1)"
  echo "requests_per_replica=$REQUESTS_PER_REPLICA"
  echo "duration_seconds=${DURATION_SECONDS:-request-count-driven}"
  echo "max_batch_size=$MAX_BATCH_SIZE"
  echo "kv_cache_safety_mb=$KV_CACHE_SAFETY_MB"
  echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
  echo "decode_attention_backend=$DECODE_ATTENTION_BACKEND"
  echo "models=${MODELS[*]}"
  echo "engines=${ENGINES[*]}"
  echo "short_rates=${SHORT_RATES[*]}"
  echo "long_rates=${LONG_RATES[*]}"
  echo "production_rates=${PRODUCTION_RATES[*]}"
  echo "repetitions=$REPETITIONS"
  echo "ttft_slo_ms=${TTFT_SLO_MS:-none}"
  echo "tpot_slo_ms=${TPOT_SLO_MS:-none}"
  echo "e2e_slo_ms=${E2E_SLO_MS:-none}"
  echo
  nvidia-smi -L
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader
} > "$RESULT_ROOT/environment.txt"

"$PYTHON_BIN" -m pip freeze > "$RESULT_ROOT/python-packages.txt"
nvidia-smi -q > "$RESULT_ROOT/nvidia-smi-full.txt"
nvidia-smi topo -m > "$RESULT_ROOT/gpu-topology.txt"
uname -a > "$RESULT_ROOT/uname.txt"
lscpu > "$RESULT_ROOT/cpu-details.txt"
free -h > "$RESULT_ROOT/host-memory.txt"
df -h "$PROJECT_ROOT" > "$RESULT_ROOT/filesystem.txt"
"$PYTHON_BIN" -c \
  'import json, torch; print(json.dumps({"torch_config": torch.__config__.show(), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "device_count": torch.cuda.device_count(), "devices": [{"name": torch.cuda.get_device_name(i), "capability": torch.cuda.get_device_capability(i), "total_memory": torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]}, indent=2))' \
  > "$RESULT_ROOT/torch-hardware.json"

if $RUN_TESTS; then
  env PYTHONPATH=. "$PYTHON_BIN" -m pytest -q -p no:cacheprovider \
    2>&1 | tee "$RESULT_ROOT/tests.log"
fi

slugify() {
  printf '%s' "$1" | tr '/:@' '---' | tr -cd '[:alnum:]._-'
}

run_case() {
  local model_id="$1"
  local engine_label="$2"
  local workload_label="$3"
  local input_length="$4"
  local output_length="$5"
  local trial="$6"
  shift 6
  local rates=("$@")
  local model_slug
  model_slug="$(slugify "$model_id")"
  local case_dir="$RESULT_ROOT/$model_slug/$engine_label/$workload_label/trial-$trial"
  mkdir -p "$case_dir"

  local command=(
    env PYTHONPATH=.
    "$PYTHON_BIN" dual_gpu_capacity_benchmark.py
    --model-id "$model_id"
    --dtype float16
    --input-length "$input_length"
    --output-length "$output_length"
    --num-requests-per-replica "$REQUESTS_PER_REPLICA"
    --max-batch-size "$MAX_BATCH_SIZE"
    --kv-cache-safety-mb "$KV_CACHE_SAFETY_MB"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --seed "$((1234 + (trial - 1) * 10000))"
    --output-dir "$case_dir/raw"
  )

  if [[ -n "$DURATION_SECONDS" ]]; then
    command+=(--duration-seconds "$DURATION_SECONDS")
  fi

  if [[ "$workload_label" == "production-mix" ]]; then
    command+=(
      --arrival-pattern poisson
      --request-shape 128:32:50
      --request-shape 384:64:30
      --request-shape 768:96:15
      --request-shape 900:64:5
    )
  fi

  case "$engine_label" in
    hf) command+=(--engine hf) ;;
    pagedserve-orca)
      command+=(
        --engine pagedserve
        --pagedserve-strategy orca
        --decode-attention-backend "$DECODE_ATTENTION_BACKEND"
      )
      ;;
    pagedserve-sarathi)
      command+=(
        --engine pagedserve
        --pagedserve-strategy sarathi
        --decode-attention-backend "$DECODE_ATTENTION_BACKEND"
      )
      ;;
    vllm) command+=(--engine vllm) ;;
  esac

  local rate
  for rate in "${rates[@]}"; do
    command+=(--request-rate "$rate")
  done
  if [[ -n "$TTFT_SLO_MS" ]]; then
    command+=(--ttft-slo-ms "$TTFT_SLO_MS")
  fi
  if [[ -n "$TPOT_SLO_MS" ]]; then
    command+=(--tpot-slo-ms "$TPOT_SLO_MS")
  fi
  if [[ -n "$E2E_SLO_MS" ]]; then
    command+=(--e2e-slo-ms "$E2E_SLO_MS")
  fi

  printf 'Running %s | %s | %s\n' "$model_id" "$engine_label" "$workload_label"
  printf '%q ' "${command[@]}" > "$case_dir/command.txt"
  printf '\n' >> "$case_dir/command.txt"
  "${command[@]}" 2>&1 | tee "$case_dir/summary.log"
}

for ((trial = 1; trial <= REPETITIONS; trial++)); do
  for model_id in "${MODELS[@]}"; do
    for engine_label in "${ENGINES[@]}"; do
      if $RUN_SHORT; then
        run_case "$model_id" "$engine_label" "in128-out32" 128 32 "$trial" "${SHORT_RATES[@]}"
      fi
      if $RUN_LONG; then
        run_case "$model_id" "$engine_label" "in900-out64" 900 64 "$trial" "${LONG_RATES[@]}"
      fi
      if $RUN_PRODUCTION; then
        run_case "$model_id" "$engine_label" "production-mix" 128 32 "$trial" "${PRODUCTION_RATES[@]}"
      fi
    done
  done
done

{
  echo "# Benchmark run $RUN_ID"
  echo
  echo "## Environment"
  echo
  echo '```text'
  cat "$RESULT_ROOT/environment.txt"
  echo '```'
  while IFS= read -r summary_file; do
    relative_path="${summary_file#"$RESULT_ROOT/"}"
    echo
    echo "## $relative_path"
    echo
    echo '```text'
    cat "$summary_file"
    echo '```'
  done < <(find "$RESULT_ROOT" -name summary.log -type f | sort)
} > "$RESULT_ROOT/REPORT.md"

find "$RESULT_ROOT" -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum > "$RESULT_ROOT/SHA256SUMS"

echo "Benchmark results completed: $RESULT_ROOT"

if $PUSH_RESULTS; then
  ORIGINAL_BRANCH="$(git symbolic-ref --quiet --short HEAD)" || {
    echo "Cannot push results from a detached HEAD." >&2
    exit 1
  }
  if [[ -z "$RESULT_BRANCH" ]]; then
    RESULT_BRANCH="benchmark-results/$RUN_ID"
  fi
  ASKPASS_HELPER=""
  if [[ -n "${GH_TOKEN:-}" ]] && command -v gh >/dev/null 2>&1; then
    gh auth setup-git >/dev/null
  elif [[ -n "${GH_TOKEN:-}" ]]; then
    ASKPASS_HELPER="$(mktemp)"
    printf '%s\n' \
      '#!/usr/bin/env bash' \
      'case "$1" in' \
      '  *Username*) printf "%s\n" "x-access-token" ;;' \
      '  *Password*) printf "%s\n" "$GH_TOKEN" ;;' \
      'esac' > "$ASKPASS_HELPER"
    chmod 700 "$ASKPASS_HELPER"
  fi

  restore_original_branch() {
    local current_branch
    current_branch="$(git symbolic-ref --quiet --short HEAD 2>/dev/null || true)"
    if [[ -n "$current_branch" && "$current_branch" != "$ORIGINAL_BRANCH" ]]; then
      git switch "$ORIGINAL_BRANCH" >/dev/null 2>&1 || true
    fi
    if [[ -n "$ASKPASS_HELPER" && -f "$ASKPASS_HELPER" ]]; then
      rm -f "$ASKPASS_HELPER"
    fi
  }
  trap restore_original_branch EXIT

  git switch -c "$RESULT_BRANCH"
  git add -f "$RESULT_ROOT"
  git -c user.name="${GIT_AUTHOR_NAME:-PagedServe Benchmark Bot}" \
      -c user.email="${GIT_AUTHOR_EMAIL:-benchmark@pagedserve.local}" \
      commit -m "benchmark: add dual-GPU results $RUN_ID"
  if [[ -n "$ASKPASS_HELPER" ]]; then
    GIT_ASKPASS="$ASKPASS_HELPER" GIT_TERMINAL_PROMPT=0 \
      git push -u origin "$RESULT_BRANCH"
  else
    GIT_TERMINAL_PROMPT=0 git push -u origin "$RESULT_BRANCH"
  fi
  git switch "$ORIGINAL_BRANCH"
  if [[ -n "$ASKPASS_HELPER" && -f "$ASKPASS_HELPER" ]]; then
    rm -f "$ASKPASS_HELPER"
  fi
  trap - EXIT

  echo "BENCHMARK_RESULTS_BRANCH=$RESULT_BRANCH"
  echo "Fetch with: git fetch origin $RESULT_BRANCH"
fi
