#!/bin/bash
#SBATCH --partition=kill-shared
#SBATCH --job-name=sft-math-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpu=20
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h200_nvl:2
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# Optional site-specific directives (Slurm only reads lines that *start* with #SBATCH).
# Uncomment by deleting the leading "# " so the line begins with #SBATCH:
# #SBATCH --account=YOUR_ACCOUNT
# #SBATCH --constraint=h100
# #SBATCH --qos=normal

# SFT experiment uses two processes on one node: HF policy on cuda:0, vLLM on cuda:1
# (see cs336_alignment/sft_experiment.py --policy_device / --vllm_device defaults).
#
# Submit from the repository root (so paths and HF_HOME resolve correctly):
#   cd /path/to/ece405-assignment3-alignment.nosync
#   export WANDB_API_KEY=...   # or rely on `wandb login` in your $HOME on the cluster
#   sbatch scripts/slurm/sft_math_sweep.slurm
#
# Override sweep hyperparameters with env vars (see scripts/sft_math_sweep.sh):
#   LR=2e-5 EVAL_EVERY_TRAIN_STEPS=150 sbatch scripts/slurm/sft_math_sweep.slurm

set -euo pipefail

export PYTHONUNBUFFERED=1

# Slurm sets the working directory to the submission cwd; keep that as repo root.
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"
mkdir -p logs

# Optional: cluster-specific modules (uncomment as needed).
# module purge
# module load cuda/12.4
# module load cudnn

# Ensure Slurm exposes two GPUs to this job (cuda:0 and cuda:1 inside the job).
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Hugging Face cache (large downloads); default matches sft_math_sweep.sh.
export HF_HOME="${HF_HOME:-$ROOT/.hf_cache}"

# Weights & Biases (required by sft_experiment).
export WANDB_PROJECT="${WANDB_PROJECT:-ece405-assignment-3}"

# Model path: override if the checkpoint is not next to the repo (see sft_math_sweep.sh).
# export MODEL_PATH="/scratch/$USER/models/Qwen2.5-0.5B"

echo "================================================================================"
echo "Slurm job id:     ${SLURM_JOB_ID:-local}"
echo "Node:             ${SLURMD_NODENAME:-unknown}"
echo "Repo root:        $ROOT"
echo "CUDA_VISIBLE:     ${CUDA_VISIBLE_DEVICES:-unset}"
echo "WANDB_PROJECT:    $WANDB_PROJECT"
echo "Start time:       $(date -Is)"
echo "================================================================================"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found in PATH. Install uv on the cluster or load the right module." >&2
  exit 1
fi

bash "$ROOT/scripts/sft_math_sweep.sh"

echo "================================================================================"
echo "Finished:         $(date -Is)"
echo "================================================================================"
