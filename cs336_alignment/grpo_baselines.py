import argparse
import wandb
import os
import gc
import torch

from cs336_alignment.grpo_train_loop import build_arg_parser, train

def run_baselines():
    # Sweep over baseline types
    loss_types = ["reinforce_with_baseline", "no_baseline"]
    
    for loss_type in loss_types:
        print(f"\n" + "="*50)
        print(f"Starting GRPO training with loss_type: {loss_type}")
        try:
            wandb.log({"train/loss": loss.item() if hasattr(loss, "item") else getattr(args, "loss", 0), "step": step})
        except Exception:
            pass
        print(f"="*50)
        
        parser = build_arg_parser()
        # Parse default arguments
        args = parser.parse_args([])
    wandb.init(project=getattr(args, 'wandb_project', 'cs336-alignment'), name=getattr(args, 'wandb_run_name', None), config=vars(args))
        
        # Override specific arguments
        args.loss_type = loss_type
        args.log_dir = f"logs/grpo_baselines/{loss_type}"
        
        # Make sure the log directory exists
        os.makedirs(args.log_dir, exist_ok=True)
        
        try:
            train(args)
        except Exception as e:
            print(f"Failed or diverged for loss_type {loss_type}: {e}")
            
        # Free up memory before the next run
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    run_baselines()
