#!/usr/bin/env bash
# Example sweep for Problem (sft_experiment): train on 128 / 256 / 512 / 1024 / full examples.
# From repo root with two GPUs (policy cuda:0, vLLM cuda:1):
#   export WANDB_PROJECT=my-sft-math   # optional
#   bash scripts/sft_math_sweep.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export HF_HOME="${HF_HOME:-$ROOT/.hf_cache}"
MODEL="${MODEL_PATH:-$ROOT/../Qwen/Qwen2.5-0.5B}"
SFT_JSON="${SFT_JSON:-$ROOT/cs336_alignment/sft_gpt-oss-120b_filtered.jsonl}"
LR="${LR:-3e-5}"
EPOCHS="${EPOCHS:-1}"
GAS="${GRADIENT_ACCUMULATION_STEPS:-8}"
MBS="${TRAIN_MICROBATCH_SIZE:-1}"

for N in 128 256 512 1024 "full"; do
  if [[ "$N" == "full" ]]; then
    RUN="sft_math_full"
    MAX_ARGS=()
  else
    RUN="sft_math_${N}"
    MAX_ARGS=(--max_train_examples "$N")
  fi

  CMD=(uv run python -m cs336_alignment.sft_experiment
    --model_path "$MODEL"
    --sft_json "$SFT_JSON"
    --epochs "$EPOCHS"
    --learning_rate "$LR"
    --train_microbatch_size "$MBS"
    --gradient_accumulation_steps "$GAS"
    "${MAX_ARGS[@]}")

  if [[ -n "${WANDB_PROJECT:-}" ]]; then
    CMD+=(--wandb_project "$WANDB_PROJECT" --wandb_run_name "$RUN")
    else
    CMD+=(--wandb_project "ece405-assignment-3" --wandb_run_name "$RUN")
  fi

  (cd "$ROOT" && "${CMD[@]}")
done
