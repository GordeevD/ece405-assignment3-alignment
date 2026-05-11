import wandb
import os
import subprocess

def run_experiment(epochs_per_rollout_batch, train_batch_size, num_steps, run_name):
    # Compute gradient accumulation steps to keep memory usage constant
    # Baseline: train_batch_size = 256, grad_accum = 4 implies memory size of 256 / 4 = 64
    micro_batch_size = 64
    gradient_accumulation_steps = max(1, train_batch_size // micro_batch_size)

    cmd = [
        "python", "cs336_alignment/grpo_off_policy.py",
        "--run_name", run_name,
        "--rollout_batch_size", "256",
        "--epochs_per_rollout_batch", str(epochs_per_rollout_batch),
        "--train_batch_size", str(train_batch_size),
        "--total_steps", str(num_steps),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps)
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

def main():
    # Broad sweep
    print("Starting broad sweep (<50 steps)...")
    epochs = [1, 2, 4]
    batch_sizes = [64, 128, 256]
    for e in epochs:
        for b in batch_sizes:
            run_name = f"grpo_off_policy_broad_e{e}_b{b}"
            run_experiment(e, b, num_steps=50, run_name=run_name)
    
    # Focused sweep
    print("Starting focused sweep (200 steps)...")
    # Choosing e=2 and b=128 based on hypothetical broad sweep results
    epochs_focused = [2, 3]
    batch_sizes_focused = [128, 256]
    for e in epochs_focused:
        for b in batch_sizes_focused:
            run_name = f"grpo_off_policy_focused_e{e}_b{b}"
            run_experiment(e, b, num_steps=200, run_name=run_name)
            
if __name__ == "__main__":
    main()
