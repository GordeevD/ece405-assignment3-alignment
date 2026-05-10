import os
import argparse
import inspect

with open("cs336_alignment/sft_experiment.py", "r") as f:
    sft_content = f.read()

prefix = sft_content.split("def train(args: argparse.Namespace) -> None:")[0]

suffix = """
def train(args: argparse.Namespace) -> None:
    \"\"\"Run the expert iteration experiment.\"\"\"
    step_log = [0]
    
    # Check resources periodically
    _check_system_resources("init")
    
    # 1. Load the initial dataset D
    records = []
    import json
    with open(args.train_json, "r") as f:
        for line in f:
            if not line.strip(): continue
            records.append(json.loads(line))
            
    val_problems, val_gts = load_val_eval_pairs(Path(args.val_json))
            
    _emit(f"[PHASE 1] Initializing Model & Tokenizer on {args.policy_device}")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.utils.data import DataLoader
    from cs336_alignment.sft_experiment import iterate_batches
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(args.policy_device)
    
    # Build vLLM instance
    _emit(f"[PHASE 1.5] Initializing vLLM on {args.vllm_device}")
    llm = init_vllm(args.model_path, args.vllm_device, args.seed, args.vllm_gpu_memory, args.vllm_max_model_len, args.vllm_max_num_seqs)

    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Initialize wandb
    if not args.no_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        wandb.define_metric("global_train_step")
        wandb.define_metric("train/*", step_metric="global_train_step")
        wandb.define_metric("eval/*", step_metric="global_train_step")

    global_train_step = 0
    prompt_template = Path(args.prompt_template_path).read_text()

    import random
    from cs336_alignment.drgrpo_grader import grade, r1_zero_reward_fn, extract_answer
    
    # Baseline eval
    with torch.no_grad():
        load_policy_into_vllm_instance(model, llm)
        eval_metrics = evaluate_math_vllm(llm, prompt_template, val_problems, val_gts, args.eval_max_tokens, args.eval_batch_size)
    if not args.no_wandb:
        wandb.log({"eval/accuracy": eval_metrics["accuracy"], "global_train_step": global_train_step})
    _emit(f"Baseline Eval Accuracy: {eval_metrics['accuracy']*100.0:.2f}%")

    for ei_step in range(1, args.ei_steps + 1):
        _emit(f"\\n--- EXPERT ITERATION STEP {ei_step}/{args.ei_steps} ---")
        
        # Sample Db
        Db = random.sample(records, k=args.ei_batch_size)
        
        # Load the latest policy weights into vLLM engine
        load_policy_into_vllm_instance(model, llm)
        
        # Generate G rollouts
        _emit(f"Sampling {args.g_rollouts} rollouts for {len(Db)} questions...")
        prompts = [build_prompt(ex["problem"], prompt_template) for ex in Db]
        
        params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.eval_max_tokens,
            min_tokens=4,
            n=args.g_rollouts,
            seed=args.seed + ei_step,
            stop=["</answer>"]
        )
        
        generated_responses = []
        expected_gts = []
        
        batch_generations = llm.generate(prompts, params)
        for gen, ex in zip(batch_generations, Db):
            for out in gen.outputs:
                generated_responses.append(out.text)
                expected_gts.append(ex.get("expected_answer"))
        
        # Filter correct using reward function
        _emit("Filtering rollouts by reward...")
        sft_prompts = []
        sft_responses = []
        
        for q, gt, resp in zip(prompts * args.g_rollouts, expected_gts, generated_responses):
            # Evaluate using r1_zero_reward_fn
            # Note: prompt already has "Assistant: <think>", we evaluate the text.
            # To trick r1_zero_reward_fn which expects </think> and <answer>, we add back closing tags if they're there.
            # Wait, r1_zero_reward_fn checks if "</think> <answer>" and "</answer>" are in response.
            resp_with_stop = resp + "</answer>"  # Because vLLM stops *at* </answer> and excludes it by default unless include_stop_str_in_output=True which is false by default. Let's see. vllm's default was false. So it excludes it.
            reward_dict = r1_zero_reward_fn(resp_with_stop, gt, fast=True)
            if reward_dict.get("reward", 0.0) == 1.0:
                sft_prompts.append(q)
                sft_responses.append(resp_with_stop)
                
        _emit(f"Kept {len(sft_prompts)} correct / {len(generated_responses)} total responses.")
        
        if len(sft_prompts) == 0:
            _emit("No correct responses generated in this EI step! Skipping training.")
            continue
            
        # Run SFT
        dataset = SFTStringDataset(sft_prompts, sft_responses)
        
        for epoch in range(args.sft_epochs):
            model.train()
            for accum, batch in enumerate(iterate_batches(dataset, args.train_microbatch_size), start=1):
                from cs336_alignment.get_response_log_probs import get_response_log_probs
                from cs336_alignment.sft_microbatch_train_step import sft_microbatch_train_step
                from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output
                
                b_prompts, b_responses = batch
                tok_batch = tokenize_prompt_and_output(b_prompts, b_responses, tokenizer)
                input_ids = tok_batch["input_ids"].to(args.policy_device)
                labels = tok_batch["labels"].to(args.policy_device)
                response_mask = tok_batch["response_mask"].to(args.policy_device).float()
                
                out = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
                logp = out["log_probs"]
                token_entropy = out["token_entropy"]
                
                # average entropy for tokens inside response mask
                avg_entropy = (token_entropy * response_mask).sum() / max(response_mask.sum().item(), 1.0)
                
                loss, meta = sft_microbatch_train_step(
                    logp, response_mask, args.gradient_accumulation_steps, 1.0
                )
                
                if accum % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    opt.step()
                    opt.zero_grad()
                    global_train_step += 1
                    
                    if not args.no_wandb:
                        wandb.log({
                            "train/loss": float(loss.item() * args.gradient_accumulation_steps),
                            "train/entropy": float(avg_entropy.item()),
                            "global_train_step": global_train_step
                        })
            
        # Eval after EI step
        with torch.no_grad():
            load_policy_into_vllm_instance(model, llm)
            eval_metrics = evaluate_math_vllm(llm, prompt_template, val_problems, val_gts, args.eval_max_tokens, args.eval_batch_size)
            
        if not args.no_wandb:
            wandb.log({"eval/accuracy": eval_metrics["accuracy"], "global_train_step": global_train_step})
        _emit(f"Eval Accuracy after EI Step {ei_step}: {eval_metrics['accuracy']*100.0:.2f}%")

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="../Qwen/Qwen2.5-0.5B")
    p.add_argument("--train_json", type=str, default="cs336_alignment/train.jsonl")
    p.add_argument("--val_json", type=str, default="cs336_alignment/sft_val.jsonl")
    p.add_argument("--prompt_template_path", type=str, default="cs336_alignment/prompts/r1_zero.prompt")
    p.add_argument("--ei_steps", type=int, default=5)
    p.add_argument("--ei_batch_size", type=int, default=1024)
    p.add_argument("--g_rollouts", type=int, default=8)
    p.add_argument("--sft_epochs", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--train_microbatch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--eval_max_tokens", type=int, default=2048)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--policy_device", type=str, default="cuda:0")
    p.add_argument("--vllm_device", type=str, default="cuda:1")
    p.add_argument("--vllm_gpu_memory", type=float, default=0.45)
    p.add_argument("--vllm_max_model_len", type=int, default=8192)
    p.add_argument("--vllm_max_num_seqs", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", type=str, default="ece405-expert-iteration")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--save_model_dir", type=str, default=None)
    return p

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
"""

with open("cs336_alignment/expert_iteration_experiment.py", "w") as f:
    f.write(prefix)
    f.write(suffix)

