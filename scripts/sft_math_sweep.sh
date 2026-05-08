#!/usr/bin/env bash
# Example sweep for Problem (sft_experiment): train on 128 / 256 / 512 / 1024 / full examples.
# From repo root with two GPUs (policy cuda:0, vLLM cuda:1). W&B is required.
#   export WANDB_PROJECT=my-sft-math   # optional; default below if unset
#   bash scripts/sft_math_sweep.sh

set -euo pipefail
# Unbuffered Python so Slurm / file logs show lines as they happen.
export PYTHONUNBUFFERED=1

# Default W&B project when unset (syntax is ${VAR:-default}, not ${VAR:"default"}).
WANDB_PROJECT="${WANDB_PROJECT:-ece405-assignment-3}"
export WANDB_PROJECT

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export HF_HOME="${HF_HOME:-$ROOT/.hf_cache}"
MODEL="${MODEL_PATH:-$ROOT/../Qwen/Qwen2.5-0.5B}"
SFT_JSON="${SFT_JSON:-$ROOT/cs336_alignment/sft_gpt-oss-120b_filtered.jsonl}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-2}"
GAS="${GRADIENT_ACCUMULATION_STEPS:-16}"
MBS="${TRAIN_MICROBATCH_SIZE:-2}"
# vLLM eval dominates runtime: each eval copies full weights to the vLLM worker.
EVAL_EVERY_TRAIN_STEPS="${EVAL_EVERY_TRAIN_STEPS:-100}"
WANDB_LOG_TRAIN_EVERY="${WANDB_LOG_TRAIN_EVERY:-10}"
TRAIN_LOG_TERMINAL_EVERY="${TRAIN_LOG_TERMINAL_EVERY:-10}"
# Set EVAL_AT_START=1 once if you want a W&B point at train_step 0 (base model).
EVAL_AT_START_ARGS=()
if [[ "${EVAL_AT_START:-0}" == "1" ]]; then
  EVAL_AT_START_ARGS=(--eval_at_start)
fi

for N in 128 256 512 1024 "full" "full_filtered"; do
  if [[ "$N" == "full" ]]; then
    RUN="sft_math_full"
    MAX_ARGS=()
  elif [[ "$N" == "full_filtered" ]]; then
    RUN="sft_math_full_filtered"
    MAX_ARGS=(--filtered_only)
  else
    RUN="sft_math_${N}"
    MAX_ARGS=(--max_train_examples "$N")
  fi

  echo "================================================================================"
  echo "sft_math_sweep: starting run N=${N} WANDB_PROJECT=${WANDB_PROJECT} RUN=${RUN}"
  echo "================================================================================"

  (cd "$ROOT" && uv run python -u -m cs336_alignment.sft_experiment
    --wandb_project "$WANDB_PROJECT"
    --wandb_run_name "$RUN"
    --model_path "$MODEL"
    --sft_json "$SFT_JSON"
    --epochs "$EPOCHS"
    --learning_rate "$LR"
    --train_microbatch_size "$MBS"
    --gradient_accumulation_steps "$GAS"
    --eval_every_train_steps "$EVAL_EVERY_TRAIN_STEPS"
    --wandb_log_train_every "$WANDB_LOG_TRAIN_EVERY"
    --train_log_terminal_every "$TRAIN_LOG_TERMINAL_EVERY"
    "${EVAL_AT_START_ARGS[@]}"
    "${MAX_ARGS[@]}")
done
