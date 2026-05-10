import argparse
import os
import gc
import torch

from cs336_alignment.grpo_train_loop import build_arg_parser, train

def sweep_learning_rates():
    # Suggested sweep for learning rates based on standard GRPO/PPO practice.
    # 1e-6 may be too small, 5e-5 may be too large. 1e-5 is default.
    learning_rates = [1e-6, 5e-6, 1e-5, 5e-5]
    
    for lr in learning_rates:
        print(f"\n" + "="*50)
        print(f"Starting GRPO training with LR: {lr}")
        print(f"="*50)
        
        parser = build_arg_parser()
        # Parse default arguments
        args = parser.parse_args([])
        
        # Override specific arguments
        args.learning_rate = lr
        args.log_dir = f"logs/grpo_lr_sweep/lr_{lr}"
        
        # Make sure the log directory exists
        os.makedirs(args.log_dir, exist_ok=True)
        
        try:
            train(args)
        except Exception as e:
            print(f"Failed or diverged for LR {lr}: {e}")
            
        # Free up memory before the next run
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    sweep_learning_rates()
