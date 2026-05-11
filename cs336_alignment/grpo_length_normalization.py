import argparse
import unittest.mock
import wandb
import gc
import os
import torch

from cs336_alignment.grpo_train_loop import build_arg_parser, train
from cs336_alignment.masked_normalize import masked_normalize
from cs336_alignment.masked_mean import masked_mean

def get_patched_masked_mean(norm_type, sampling_max_tokens):
    if norm_type == "masked_mean":
        return masked_mean
    elif norm_type == "masked_normalize":
        def patched_normalize(tensor, mask, dim=None):
            # Normalizing by max generation length to compare with mean over variable length
            return masked_normalize(tensor, mask, normalize_constant=float(sampling_max_tokens), dim=dim)
        return patched_normalize
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")

def run_experiment():
    norm_types = ["masked_mean", "masked_normalize"]
    
    for norm_type in norm_types:
        print(f"\n" + "="*50)
        print(f"Starting GRPO training with normalization: {norm_type}")
        print(f"="*50)
        
        parser = build_arg_parser()
        # Parse default arguments
        args = parser.parse_args([])
        
        args.log_dir = f"logs/grpo_length_norm/{norm_type}"
        os.makedirs(args.log_dir, exist_ok=True)
        
        # Init wandb session
        wandb.init(
            project=getattr(args, 'wandb_project', 'cs336-alignment'),
            name=f"grpo-len-norm-{norm_type}",
            config=vars(args),
            reinit=True
        )
        
        patched_func = get_patched_masked_mean(norm_type, args.sampling_max_tokens)
        
        with unittest.mock.patch('cs336_alignment.grpo_microbatch_train_step.masked_mean', new=patched_func):
            try:
                train(args)
            except Exception as e:
                print(f"Failed or diverged for {norm_type}: {e}")
                
        wandb.finish()
        
        # Free up memory before the next run
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    run_experiment()
