#!/bin/bash
# Expert Iteration Experiment on MATH Dataset
# This script runs expert iteration with varying hyperparameters

set -e

# Activate the vLLM Metal environment for macOS GPU support
source ~/.venv-vllm-metal/bin/activate

# Base configuration
MODEL_PATH="../Qwen/Qwen2.5-0.5B"
TRAIN_DATA="cs336_alignment/train.jsonl"
VAL_DATA="cs336_alignment/sft_val.jsonl"
PROMPT_TEMPLATE="cs336_alignment/prompts/r1_zero.prompt"

# Hyperparameters to vary
EI_STEPS=5
TEMPERATURE=0.7
LEARNING_RATE=1e-5
EVAL_MAX_TOKENS=2048
EVAL_BATCH_SIZE=16

# For macOS Metal, use CPU for policy device since we don't have CUDA
POLICY_DEVICE="cpu"
VLLM_DEVICE="mps"  # macOS GPU
VLLM_GPU_MEMORY=0.45

# Seed
SEED=42

echo "=========================================="
echo "Expert Iteration Configuration"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Train Data: $TRAIN_DATA"
echo "Val Data: $VAL_DATA"
echo "EI Steps: $EI_STEPS"
echo "Temperature: $TEMPERATURE"
echo "Policy Device: $POLICY_DEVICE"
echo "vLLM Device: $VLLM_DEVICE"
echo "=========================================="

# Run with different batch sizes and rollout/epoch configurations
run_config() {
    local config_name=$1
    local ei_batch_size=$2
    local g_rollouts=$3
    local sft_epochs=$4
    
    echo ""
    echo "Running: $config_name"
    echo "  Batch Size: $ei_batch_size"
    echo "  Rollouts: $g_rollouts"
    echo "  SFT Epochs: $sft_epochs"
    echo ""
    
    HF_HOME="$(pwd)/.hf_cache" python -m cs336_alignment.expert_iteration_experiment \
        --model_path "$MODEL_PATH" \
        --train_json "$TRAIN_DATA" \
        --val_json "$VAL_DATA" \
        --prompt_template_path "$PROMPT_TEMPLATE" \
        --ei_steps "$EI_STEPS" \
        --ei_batch_size "$ei_batch_size" \
        --g_rollouts "$g_rollouts" \
        --sft_epochs "$sft_epochs" \
        --temperature "$TEMPERATURE" \
        --learning_rate "$LEARNING_RATE" \
        --eval_max_tokens "$EVAL_MAX_TOKENS" \
        --eval_batch_size "$EVAL_BATCH_SIZE" \
        --policy_device "$POLICY_DEVICE" \
        --vllm_device "$VLLM_DEVICE" \
        --vllm_gpu_memory "$VLLM_GPU_MEMORY" \
        --seed "$SEED" \
        --wandb_project "ece405-expert-iteration" \
        --wandb_run_name "$config_name"
}

# Configuration 1: Smaller batch, more rollouts, more epochs
run_config "ei-batch512-G8-epoch2" 512 8 2

# Configuration 2: Medium batch, medium rollouts, 1 epoch
run_config "ei-batch1024-G4-epoch1" 1024 4 1

# Configuration 3: Larger batch, fewer rollouts, 1 epoch
run_config "ei-batch2048-G2-epoch1" 2048 2 1

echo ""
echo "=========================================="
echo "Expert Iteration Experiments Complete"
echo "=========================================="
