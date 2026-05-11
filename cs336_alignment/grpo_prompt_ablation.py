#!/usr/bin/env python3
import subprocess
import os

def run_experiment(name, prompt_path, run_name):
    print(f"Running {name}...")
    cmd = [
        "uv", "run", "cs336_alignment/grpo_train_loop.py",
        "--prompt_template_path", prompt_path,
        "--run_name", run_name,
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    run_experiment(
        "R1-Zero Prompt", 
        "cs336_alignment/prompts/r1_zero.prompt", 
        "grpo_prompt_r1_zero"
    )
    run_experiment(
        "Question Only Prompt", 
        "cs336_alignment/prompts/question_only.prompt", 
        "grpo_prompt_question_only"
    )
