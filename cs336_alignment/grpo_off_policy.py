from __future__ import annotations

import argparse
import wandb
import os
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.compute_group_normalized_rewards import compute_group_normalized_rewards
from cs336_alignment.get_response_log_probs import get_response_log_probs
from cs336_alignment.grpo_microbatch_train_step import grpo_microbatch_train_step
from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.expert_iteration_experiment import (
    load_sft_records,
    load_val_eval_pairs,
    build_prompt,
)



def train(args: argparse.Namespace) -> None:
    """GRPO train loop with HF-only generation (no vLLM required)."""
    device = args.policy_device
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model_kwargs = {"trust_remote_code": True, "attn_implementation": "eager"}
    if device.startswith("cuda"):
        model_kwargs["torch_dtype"] = torch.bfloat16
    policy = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    policy.to(device)

    opt = torch.optim.AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    records = load_sft_records(Path(args.train_json))
    train_problems = [r["problem"] for r in records if r.get("problem")]
    train_gts = [r.get("expected_answer") for r in records if r.get("problem")]

    val_problems, val_gts = load_val_eval_pairs(Path(args.val_json))
    prompt_template = Path(args.prompt_template_path).read_text()

    assert args.train_batch_size % args.gradient_accumulation_steps == 0
    micro_train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    assert args.rollout_batch_size % args.group_size == 0
    n_prompts_per_rollout_batch = args.rollout_batch_size // args.group_size
    n_microbatches_per_rollout_batch = args.rollout_batch_size // micro_train_batch_size

    val_steps: list[int] = []
    val_rewards: list[float] = []

    def hf_generate_batch(prompts: list[str], num_per_prompt: int) -> list[list[str]]:
        results: list[list[str]] = []
        policy.eval()
        for idx, p in enumerate(prompts):
            print(f"      Generating prompt {idx + 1}/{len(prompts)} ({num_per_prompt} samples)...", flush=True)
            enc = tokenizer(p, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                gen = policy.generate(
                    **enc,
                    do_sample=args.sampling_temperature > 0.0,
                    temperature=float(args.sampling_temperature),
                    min_new_tokens=args.sampling_min_tokens,
                    max_new_tokens=args.sampling_max_tokens,
                    num_return_sequences=num_per_prompt,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            texts: list[str] = []
            prompt_len = enc["input_ids"].shape[1]
            for i in range(num_per_prompt):
                seq = gen[i]
                texts.append(tokenizer.decode(seq[prompt_len:], skip_special_tokens=True))
            results.append(texts)
        return results

    def eval_hf(problems: list[str], gts: list[str]) -> dict[str, float]:
        correct = 0
        total = 0
        n = min(len(problems), args.eval_batch_size)
        eval_prompts = [build_prompt(p, prompt_template) for p in problems[:n]]
        eval_gts = gts[:n]
        if n == 0:
            return {"accuracy": 0.0}

        print(f"  Evaluating on {n} validation examples...")
        batch_gens = hf_generate_batch(eval_prompts, 1)
        for gens, gt in zip(batch_gens, eval_gts):
            rew = r1_zero_reward_fn(gens[0] + "</answer>", gt)
            total += 1
            if rew.get("answer_reward", 0.0) >= 1.0:
                correct += 1
        return {"accuracy": correct / max(total, 1)}

    print(f"Using policy device: {device}. Starting GRPO training with {args.n_grpo_steps} steps.")

    global_step = 0
    for step in range(1, args.n_grpo_steps + 1):
        step_start = time.time()
        print(f"\n[GRPO Step {step}/{args.n_grpo_steps}] starting...")

        chosen = random.sample(range(len(train_problems)), k=min(n_prompts_per_rollout_batch, len(train_problems)))
        prompts = [build_prompt(train_problems[i], prompt_template) for i in chosen]
        gts = [train_gts[i] for i in chosen]

        print(f"  Generating {len(prompts)} prompts x {args.group_size} rollouts...")
        batch_gens = hf_generate_batch(prompts, args.group_size)

        rollout_responses: list[str] = []
        repeated_gts: list[str] = []
        flat_prompts: list[str] = []
        for i, gens in enumerate(batch_gens):
            for txt in gens:
                rollout_responses.append(txt + "</answer>")
                repeated_gts.append(gts[i])
                flat_prompts.append(prompts[i])

        tok = tokenize_prompt_and_output(flat_prompts, rollout_responses, tokenizer)
        input_ids = tok["input_ids"].to(device)
        labels = tok["labels"].to(device)
        response_mask = tok["response_mask"].to(device).float()

        print(f"  Computing advantages for {len(rollout_responses)} responses...")
        with torch.no_grad():
            advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
                reward_fn=r1_zero_reward_fn,
                rollout_responses=rollout_responses,
                repeated_ground_truths=repeated_gts,
                group_size=args.group_size,
                advantage_eps=args.advantage_eps,
                normalize_by_std=args.use_std_normalization,
            )
        print(f"    Reward mean: {reward_meta['reward_mean']:.4f}, std: {reward_meta['reward_std']:.4f}")

        advantages = advantages.to(device)
        raw_rewards = raw_rewards.to(device)

        print(f"  Computing old log probs...")
        policy.eval()
        with torch.inference_mode():
            old_out = get_response_log_probs(policy, input_ids, labels, return_token_entropy=False)
            old_log_probs = old_out["log_probs"].detach()

        policy.train()
        
        for epoch in range(args.epochs_per_rollout_batch):
            out = get_response_log_probs(policy, input_ids, labels, return_token_entropy=True)
            policy_log_probs = out["log_probs"]

            print(f"  Training epoch {epoch+1}/{args.epochs_per_rollout_batch} with {n_microbatches_per_rollout_batch} microbatches...")
            total_loss = 0.0
            opt.zero_grad()
            for m in range(n_microbatches_per_rollout_batch):
                s = m * micro_train_batch_size
                e = min(s + micro_train_batch_size, len(rollout_responses))
                if s >= e:
                    break

                mb_logp = policy_log_probs[s:e]
                mb_mask = response_mask[s:e]
                mb_adv = advantages[s:e]
                mb_raw = raw_rewards[s:e]
                mb_old_logp = old_log_probs[s:e] if old_log_probs is not None else None

                loss, _meta = grpo_microbatch_train_step(
                    policy_log_probs=mb_logp,
                    response_mask=mb_mask,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    loss_type=args.loss_type,
                    raw_rewards=mb_raw if args.loss_type == "no_baseline" else None,
                    advantages=mb_adv if args.loss_type != "no_baseline" else None,
                    old_log_probs=mb_old_logp,
                    cliprange=args.cliprange,
                )
                total_loss += float(loss.item())

                if (m + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()
                    global_step += 1

            if n_microbatches_per_rollout_batch % args.gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                global_step += 1

        if step % args.eval_interval == 0:
            print("  Running validation...")
            with torch.no_grad():
                eval_acc = eval_hf(val_problems, val_gts)
            val_steps.append(step)
            val_rewards.append(eval_acc["accuracy"])
            elapsed = time.time() - step_start
            print(f"  [step {step}] val_acc={val_rewards[-1]:.4f} loss={total_loss:.6f} elapsed={elapsed:.1f}s")
        try:
            wandb.log({"train/loss": loss.item() if hasattr(loss, "item") else getattr(args, "loss", 0), "step": step})
        except Exception:
            pass

            if args.log_dir:
                os.makedirs(args.log_dir, exist_ok=True)
                try:
                    import matplotlib.pyplot as plt

                    plt.figure(figsize=(8, 4))
                    plt.plot(val_steps, val_rewards, marker="o", label="val_acc")
                    plt.xlabel("step")
                    plt.ylabel("validation accuracy")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.savefig(os.path.join(args.log_dir, "grpo_val_rewards.png"), dpi=100)
                    plt.close()
                except Exception as e:
                    print(f"  Warning: could not save plot: {e}")

                ex_path = os.path.join(args.log_dir, f"example_rollouts_step_{step}.txt")
                with open(ex_path, "w", encoding="utf-8") as f:
                    for i, resp in enumerate(rollout_responses[:8]):
                        f.write(f"=== Example {i + 1} ===\n{resp}\n\n")

    print(f"\nTraining complete! {len(val_steps)} validation checkpoints saved.")
    if args.log_dir:
        print(f"Results saved to {args.log_dir}")



def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GRPO train loop for MATH using HuggingFace only")
    p.add_argument("--model_path", type=str, default="../Qwen/Qwen2.5-0.5B")
    p.add_argument("--train_json", type=str, default="cs336_alignment/train.jsonl")
    p.add_argument("--val_json", type=str, default="cs336_alignment/sft_val.jsonl")
    p.add_argument("--prompt_template_path", type=str, default="cs336_alignment/prompts/r1_zero.prompt")
    p.add_argument("--n_grpo_steps", type=int, default=200)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--advantage_eps", type=float, default=1e-6)
    p.add_argument("--rollout_batch_size", type=int, default=256)
    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--sampling_temperature", type=float, default=1.0)
    p.add_argument("--sampling_min_tokens", type=int, default=4)
    p.add_argument("--sampling_max_tokens", type=int, default=1024)
    p.add_argument("--train_batch_size", type=int, default=256)
    p.add_argument("--gradient_accumulation_steps", type=int, default=128)
    p.add_argument("--loss_type", type=str, default="grpo_clip")
    p.add_argument("--epochs_per_rollout_batch", type=int, default=2)
    p.add_argument("--cliprange", type=float, default=0.2)
    p.add_argument("--use_std_normalization", action="store_true")
    p.add_argument("--policy_device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_interval", type=int, default=5)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--log_dir", type=str, default="logs/grpo")
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    wandb.init(project=getattr(args, 'wandb_project', 'cs336-alignment'), name=getattr(args, 'wandb_run_name', None), config=vars(args))
    train(args)


if __name__ == "__main__":
    main()
