import wandb
import subprocess
import os

def run_experiment(use_std_normalization: bool):
    cmd = [
        "uv", "run", "cs336_alignment/grpo_train_loop.py",
        "--batch_size", "256",
        "--learning_rate", "1e-5",
    ]
    if use_std_normalization:
        cmd.append("--use_std_normalization")
        name = "with_std_normalization"
    else:
        name = "without_std_normalization"
    
    print(f"Running experiment: {name}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run_experiment(True)
    run_experiment(False)
