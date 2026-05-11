import subprocess
import os

def run_leaderboard_experiment():
    print("Running optimized GRPO training for leaderboard...")
    cmd = [
        "uv", "run", "cs336_alignment/grpo_train_loop.py",
        "--prompt_template_path", "cs336_alignment/prompts/r1_zero.prompt",
        "--run_name", "grpo_leaderboard_4hrs",
        "--learning_rate", "5e-6",
        "--batch_size", "64",
        "--micro_batch_size", "2",
        "--group_size", "8",
        "--max_prompt_length", "256",
        "--max_completion_length", "1024",
        "--eval_every", "50",
        "--eval_batches", "78", # approx 5k examples if batch_size=64 (78*64 = 4992)
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    run_leaderboard_experiment()
