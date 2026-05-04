"""
Supervised fine-tuning on MATH-style reasoning traces with periodic vLLM evaluation.

Designed for two GPUs: policy on ``cuda:0``, vLLM engine on ``cuda:1`` (see ``--policy_device`` / ``--vllm_device``).

Example::

    HF_HOME="$(pwd)/.hf_cache" uv run python -m cs336_alignment.sft_experiment \\
        --wandb_project my-ece405-runs \\
        --wandb_run_name sft-qwen-math-01 \\
        --model_path ../../Qwen/Qwen2.5-0.5B \\
        --sft_json cs336_alignment/sft_gpt-oss-120b_filtered.jsonl \\
        --max_train_examples 512 --epochs 1 --learning_rate 3e-5 \\
        --train_microbatch_size 1 --gradient_accumulation_steps 8
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from cs336_alignment.drgrpo_grader import grade, r1_zero_reward_fn
from cs336_alignment.get_response_log_probs import get_response_log_probs
from cs336_alignment.sft_microbatch_train_step import sft_microbatch_train_step
from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output


def _emit(msg: str) -> None:
    print(msg, flush=True)


def _next_terminal_step(counter: list[int], msg: str) -> None:
    """Monotonic step label for the terminal (one counter for setup + training)."""
    counter[0] += 1
    _emit(f"[step {counter[0]}] {msg}")


def _next_terminal_step_tqdm_safe(counter: list[int], msg: str, *, use_tqdm_write: bool) -> None:
    """Like ``_next_terminal_step`` but uses ``tqdm.write`` when a tqdm bar is active (avoids garbled bars)."""
    counter[0] += 1
    line = f"[step {counter[0]}] {msg}"
    if use_tqdm_write:
        tqdm.write(line)
    else:
        _emit(line)


def init_vllm(
    model_id: str,
    device: str,
    seed: int,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 8192,
    max_num_seqs: int = 64,
) -> LLM:
    """Start vLLM on a dedicated GPU (separate from the HF policy)."""
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM) -> None:
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def load_sft_records(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if raw.startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {path}")
        return data
    out: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def build_prompt(problem: str, template: str) -> str:
    return template.replace("{question}", problem)


def load_val_eval_pairs(val_path: Path) -> tuple[list[str], list[str]]:
    """Load validation problems and ground-truth answers from ``sft_val.jsonl``-style JSON.

    Each record must include ``problem`` and ``expected_answer`` (same schema as
    ``cs336_alignment/sft_val.jsonl``).
    """
    records = load_sft_records(val_path)
    problems: list[str] = []
    ground_truths: list[str] = []
    for ex in records:
        prob = ex.get("problem")
        ans = ex.get("expected_answer")
        if prob is None or ans is None:
            continue
        problems.append(str(prob))
        ground_truths.append(str(ans))
    if not problems:
        raise RuntimeError(f"No rows with 'problem' and 'expected_answer' in {val_path}")
    return problems, ground_truths


class SFTStringDataset(Dataset):
    def __init__(self, prompts: list[str], responses: list[str]):
        assert len(prompts) == len(responses)
        self.prompts = prompts
        self.responses = responses

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return {"prompt": self.prompts[idx], "response": self.responses[idx]}


def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[dict[str, list[str]]]:
    indices = list(range(len(dataset)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        prompts = [dataset[i]["prompt"] for i in chunk]
        responses = [dataset[i]["response"] for i in chunk]
        yield {"prompt": prompts, "response": responses}


def evaluate_math_vllm(
    llm: LLM,
    prompt_template: str,
    problems: list[str],
    ground_truths: list[str],
    max_gen_tokens: int,
    eval_batch_size: int,
    *,
    show_progress: bool = True,
    desc: str = "validation",
) -> dict[str, float]:
    prompts = [build_prompt(p, prompt_template) for p in problems]
    params = SamplingParams(temperature=0.0, max_tokens=max_gen_tokens)
    correct = 0
    total = 0
    n_batches = max(1, math.ceil(len(prompts) / eval_batch_size))
    starts = range(0, len(prompts), eval_batch_size)
    batch_iter = tqdm(
        starts,
        total=n_batches,
        desc=desc,
        unit="batch",
        leave=False,
        disable=not show_progress,
        dynamic_ncols=True,
    )
    for start in batch_iter:
        batch_p = prompts[start : start + eval_batch_size]
        batch_gt = ground_truths[start : start + eval_batch_size]
        outputs = llm.generate(batch_p, params)
        for out, gt in zip(outputs, batch_gt, strict=True):
            text = out.outputs[0].text
            rew = r1_zero_reward_fn(text, gt)
            total += 1
            if rew.get("answer_reward", 0.0) >= 1.0:
                correct += 1
        if show_progress and total:
            acc_so_far = correct / total
            batch_iter.set_postfix(acc=f"{100.0 * acc_so_far:.1f}%", ok=f"{correct}/{total}")
    return {"accuracy": correct / max(total, 1), "n": float(total)}


def _wandb_init(args: argparse.Namespace, extra_config: dict[str, Any]) -> Any:
    """Start a W&B run and bind custom x-axes (explicit metric names; globs are unreliable)."""
    import wandb

    cfg = {**vars(args), **extra_config}
    entity = args.wandb_entity or os.environ.get("WANDB_ENTITY")
    init_kw: dict[str, Any] = {
        "project": args.wandb_project,
        "name": args.wandb_run_name,
        "config": cfg,
    }
    if entity:
        init_kw["entity"] = entity

    run = wandb.init(**init_kw)
    if run is None:
        raise RuntimeError("wandb.init() returned None (check WANDB credentials / project access).")

    # Custom x-axes: register step counters, then bind each series (see W&B "Customize log axes").
    wandb.define_metric("train_step")
    for name in (
        "train/loss",
        "train/masked_neg_log_prob_sum",
        "train/response_token_count",
        "train/epoch",
    ):
        wandb.define_metric(name, step_metric="train_step")

    wandb.define_metric("eval_step")
    for name in ("eval/accuracy", "eval/n", "eval/n_correct"):
        wandb.define_metric(name, step_metric="eval_step")

    mode = os.environ.get("WANDB_MODE", "online")
    _emit(f"Weights & Biases mode: {mode}")
    url = getattr(run, "url", None)
    if url:
        _emit(f"Weights & Biases run URL: {url}")
    elif mode == "offline":
        _emit("Weights & Biases: offline run (sync later with `wandb sync` on the run directory).")
    else:
        _emit("Weights & Biases: run started (no public URL yet).")

    return run


def train(args: argparse.Namespace) -> None:
    step_log: list[int] = [0]
    _next_terminal_step(step_log, "starting — policy device, seeds, optional HF_HOME")
    device = torch.device(args.policy_device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.hf_home:
        os.environ.setdefault("HF_HOME", str(Path(args.hf_home).expanduser().resolve()))
        _next_terminal_step(step_log, f"HF_HOME set to {args.hf_home}")
    else:
        _next_terminal_step(step_log, "HF_HOME unchanged (no --hf_home)")

    _next_terminal_step(step_log, f"reading prompt template: {args.prompt_template_path}")
    template = Path(args.prompt_template_path).read_text(encoding="utf-8")
    _next_terminal_step(step_log, f"loading SFT JSON: {args.sft_json}")
    records = load_sft_records(Path(args.sft_json))
    _next_terminal_step(step_log, f"loaded {len(records)} raw SFT records")
    n_raw = len(records)
    if args.filtered_only:
        filt: list[dict[str, Any]] = []
        for ex in records:
            exp = ex.get("expected_answer")
            ext = ex.get("extracted_answer")
            if exp is None or ext is None:
                continue
            if grade(str(ext).strip(), str(exp).strip(), fast=True):
                filt.append(ex)
        records = filt
        _next_terminal_step(
            step_log,
            f"filtered to {len(records)} / {n_raw} rows (graded-correct extracted answers)",
        )
    else:
        _next_terminal_step(step_log, "not using --filtered_only (no answer-based filter)")

    if args.max_train_examples is not None:
        records = records[: args.max_train_examples]
        _next_terminal_step(
            step_log,
            f"capped training records to max_train_examples={args.max_train_examples}",
        )
    else:
        _next_terminal_step(step_log, "no max_train_examples cap (using all parsed records)")

    prompts = []
    responses = []
    for ex in records:
        problem = ex.get("problem")
        trace = ex.get("reasoning_trace")
        if ex.get("prompt") is not None and ex.get("response") is not None:
            prompts.append(str(ex["prompt"]))
            responses.append(str(ex["response"]))
        elif problem is not None and trace is not None:
            prompts.append(build_prompt(str(problem), template))
            responses.append(str(trace))
        else:
            continue

    if not prompts:
        raise RuntimeError("No training examples after parsing; check JSON fields (problem, reasoning_trace).")

    _next_terminal_step(step_log, f"built {len(prompts)} supervised prompt/response pairs for training")
    train_ds = SFTStringDataset(prompts, responses)
    _next_terminal_step(
        step_log,
        f"loading validation pairs from {Path(args.val_json).expanduser().resolve()}",
    )
    eval_problems, eval_ground_truths = load_val_eval_pairs(Path(args.val_json).expanduser().resolve())
    _next_terminal_step(step_log, f"loaded {len(eval_problems)} validation problems with expected answers")

    # W&B before model load so runs appear immediately and failures still create a partial run.
    _next_terminal_step(step_log, "initializing Weights & Biases")
    wandb_run = _wandb_init(
        args,
        {
            "n_train_examples": len(train_ds),
            "n_val_examples": len(eval_problems),
            "val_json_resolved": str(Path(args.val_json).expanduser().resolve()),
            "sft_json_resolved": str(Path(args.sft_json).expanduser().resolve()),
        },
    )
    _next_terminal_step(step_log, "Weights & Biases run ready (see messages above for mode / URL)")

    _next_terminal_step(
        step_log,
        f"loading tokenizer from {args.model_path} (first run may download; can take minutes)",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    _next_terminal_step(step_log, "tokenizer ready (pad token aligned with EOS if needed)")

    _next_terminal_step(step_log, f"loading policy weights from {args.model_path} in bfloat16")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.to(device)
    model.train()
    _next_terminal_step(step_log, f"policy on {device} in train mode")

    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    _next_terminal_step(
        step_log,
        f"optimizer: AdamW (lr={args.learning_rate}, weight_decay={args.weight_decay})",
    )

    _emit("--- SFT configuration ---")
    _emit(f"  train examples: {len(train_ds)}")
    _emit(f"  val examples:   {len(eval_problems)} (from {args.val_json})")
    _emit(f"  epochs: {args.epochs}  lr: {args.learning_rate}  microbatch: {args.train_microbatch_size}  grad_accum: {args.gradient_accumulation_steps}")
    _emit(f"  policy_device: {args.policy_device}  vllm_device: {args.vllm_device}  skip_vllm_eval: {args.skip_vllm_eval}")
    _emit(f"  eval every {args.eval_every_train_steps} optimizer steps")
    _emit("-------------------------")

    llm: LLM | None = None
    if not args.skip_vllm_eval:
        _next_terminal_step(
            step_log,
            f"starting vLLM on {args.vllm_device} (gpu_mem={args.vllm_gpu_memory}, "
            f"max_model_len={args.vllm_max_model_len}, max_num_seqs={args.vllm_max_num_seqs})",
        )
        llm = init_vllm(
            args.model_path,
            device=args.vllm_device,
            seed=args.seed,
            gpu_memory_utilization=args.vllm_gpu_memory,
            max_model_len=args.vllm_max_model_len,
            max_num_seqs=args.vllm_max_num_seqs,
        )
        _next_terminal_step(step_log, "vLLM engine initialized")
    else:
        _next_terminal_step(step_log, "skipping vLLM (--skip_vllm_eval); validation during training disabled")

    train_step = 0
    eval_step = 0

    def run_eval(tag: str) -> float:
        nonlocal eval_step
        if llm is None:
            return float("nan")
        _next_terminal_step(
            step_log,
            f"eval:{tag} — sync policy to vLLM, generate on {len(eval_problems)} val items",
        )
        model.eval()
        load_policy_into_vllm_instance(model, llm)
        metrics = evaluate_math_vllm(
            llm,
            template,
            eval_problems,
            eval_ground_truths,
            max_gen_tokens=args.eval_max_tokens,
            eval_batch_size=args.eval_batch_size,
            show_progress=not args.no_progress_bar,
            desc=f"eval:{tag}",
        )
        acc = metrics["accuracy"]
        n_ev = int(metrics["n"])
        n_ok = int(round(acc * n_ev))
        _emit(
            f"[eval:{tag}] step={eval_step}  accuracy={acc:.6f} ({n_ok}/{n_ev} correct)  "
            f"({100.0 * acc:.2f}%)"
        )
        wandb_run.log(
            {
                "eval_step": eval_step,
                "eval/accuracy": acc,
                "eval/n": metrics["n"],
                "eval/n_correct": float(n_ok),
            },
            commit=True,
        )
        eval_step += 1
        model.train()
        return acc

    if llm is not None:
        run_eval("init")

    n_microbatches = max(1, math.ceil(len(train_ds) / args.train_microbatch_size))
    use_tqdm_write_for_train = not args.no_progress_bar
    for epoch in range(args.epochs):
        _next_terminal_step(
            step_log,
            f"epoch {epoch + 1}/{args.epochs} — {n_microbatches} microbatches "
            f"(size={args.train_microbatch_size}, grad_accum={args.gradient_accumulation_steps})",
        )
        batches = iterate_batches(train_ds, args.train_microbatch_size, shuffle=True, seed=args.seed + epoch)
        accum = 0
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(
            batches,
            total=n_microbatches,
            desc=f"SFT train epoch {epoch + 1}/{args.epochs}",
            unit="microbatch",
            leave=True,
            disable=args.no_progress_bar,
            dynamic_ncols=True,
        )
        last_loss_v: float | None = None
        last_msum: float | None = None
        last_rtok: float | None = None
        for batch in pbar:
            tok_batch = tokenize_prompt_and_output(
                batch["prompt"],
                batch["response"],
                tokenizer,
            )
            input_ids = tok_batch["input_ids"].to(device)
            labels = tok_batch["labels"].to(device)
            response_mask = tok_batch["response_mask"].to(device).float()

            out = get_response_log_probs(model, input_ids, labels, return_token_entropy=False)
            logp = out["log_probs"]

            loss, meta = sft_microbatch_train_step(
                policy_log_probs=logp,
                response_mask=response_mask,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                normalize_constant=1.0,
            )
            last_loss_v = float(loss.item())
            last_msum = float(meta["masked_neg_log_prob_sum"].item())
            last_rtok = float(meta["response_token_count"].item())
            accum += 1
            if accum % args.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)
                train_step += 1
                loss_v = last_loss_v
                msum = last_msum
                rtok = last_rtok
                wandb_run.log(
                    {
                        "train_step": train_step,
                        "train/loss": loss_v,
                        "train/masked_neg_log_prob_sum": msum,
                        "train/response_token_count": rtok,
                        "train/epoch": float(epoch),
                    },
                    commit=True,
                )
                pbar.set_postfix(
                    opt_step=train_step,
                    loss=f"{loss_v:.4f}",
                    rtok=int(rtok),
                )
                _next_terminal_step_tqdm_safe(
                    step_log,
                    f"optimizer step {train_step} (epoch {epoch + 1}/{args.epochs}) — "
                    f"loss={loss_v:.6f} masked_nll_sum={msum:.4f} response_tokens={int(rtok)}",
                    use_tqdm_write=use_tqdm_write_for_train,
                )
                if llm is not None and train_step % args.eval_every_train_steps == 0:
                    run_eval(f"epoch{epoch}_step{train_step}")

        if accum % args.gradient_accumulation_steps != 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)
            train_step += 1
            if last_loss_v is not None:
                wandb_run.log(
                    {
                        "train_step": train_step,
                        "train/loss": last_loss_v,
                        "train/masked_neg_log_prob_sum": last_msum,
                        "train/response_token_count": last_rtok,
                        "train/epoch": float(epoch),
                    },
                    commit=True,
                )
            if not args.no_progress_bar:
                pbar.set_postfix(opt_step=train_step, loss="tail", rtok="—")
            if (
                last_loss_v is not None
                and last_msum is not None
                and last_rtok is not None
            ):
                _next_terminal_step_tqdm_safe(
                    step_log,
                    f"optimizer step {train_step} (epoch {epoch + 1}/{args.epochs}, tail batch) — "
                    f"loss={last_loss_v:.6f} masked_nll_sum={last_msum:.4f} response_tokens={int(last_rtok)}",
                    use_tqdm_write=use_tqdm_write_for_train,
                )
            else:
                _next_terminal_step_tqdm_safe(
                    step_log,
                    f"optimizer step {train_step} (epoch {epoch + 1}/{args.epochs}, tail batch) — "
                    "no loss stats (skipped microbatch loop?)",
                    use_tqdm_write=use_tqdm_write_for_train,
                )
            if llm is not None and train_step % args.eval_every_train_steps == 0:
                run_eval(f"epoch{epoch}_step{train_step}_tail")

    _next_terminal_step(step_log, "all training epochs complete")
    final_acc = float("nan")
    if llm is not None:
        final_acc = run_eval("final")
    else:
        _next_terminal_step(step_log, "skipping final eval (no vLLM)")

    if args.save_model_dir:
        out_dir = Path(args.save_model_dir)
        _next_terminal_step(step_log, f"saving policy and tokenizer to {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        _next_terminal_step(step_log, f"saved model and tokenizer to {out_dir}")
    else:
        _next_terminal_step(step_log, "no --save_model_dir; skipping checkpoint write")

    _next_terminal_step(step_log, "run finished — summary below")
    _emit("--- Run finished ---")
    if final_acc == final_acc:  # not NaN
        _emit(f"  final validation accuracy: {final_acc:.6f} ({100.0 * final_acc:.2f}%)")
    else:
        _emit("  final validation accuracy: (skipped — no vLLM eval)")

    url = getattr(wandb_run, "url", None)
    if url:
        _emit(f"  W&B URL: {url}")
    _next_terminal_step(step_log, "closing Weights & Biases run")
    wandb_run.finish()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SFT on MATH reasoning JSON with vLLM eval.")
    here = Path(__file__).resolve().parent
    p.add_argument("--model_path", type=str, default="../../Qwen/Qwen2.5-0.5B")
    p.add_argument("--sft_json", type=str, default=str(here / "sft_gpt-oss-120b_filtered.jsonl"))
    p.add_argument(
        "--val_json",
        type=str,
        default=str(here / "sft_val.jsonl"),
        help="Validation JSON/JSONL with 'problem' and 'expected_answer' per row.",
    )
    p.add_argument("--prompt_template_path", type=str, default=str(here / "prompts" / "r1_zero.prompt"))
    p.add_argument("--max_train_examples", type=int, default=None, help="Cap number of SFT records (e.g. 128, 256, …).")
    p.add_argument("--filtered_only", action="store_true", help="Keep only rows whose extracted answer grades correct.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--train_microbatch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--eval_every_train_steps", type=int, default=50)
    p.add_argument("--eval_max_tokens", type=int, default=2048)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--policy_device", type=str, default="cuda:0")
    p.add_argument("--vllm_device", type=str, default="cuda:1")
    p.add_argument("--vllm_gpu_memory", type=float, default=0.85)
    p.add_argument("--vllm_max_model_len", type=int, default=8192)
    p.add_argument("--vllm_max_num_seqs", type=int, default=64)
    p.add_argument("--skip_vllm_eval", action="store_true", help="Train only (single GPU / no vLLM).")
    p.add_argument(
        "--no_progress_bar",
        action="store_true",
        help="Disable tqdm progress bars (plain logs for CI or non-TTY).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--wandb_project",
        type=str,
        required=True,
        help="Weights & Biases project name (required). Log in with `wandb login` or set WANDB_API_KEY.",
    )
    p.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name; default is a timestamp if omitted.",
    )
    p.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (user or team). Default: WANDB_ENTITY env or `wandb login` default.",
    )
    p.add_argument("--save_model_dir", type=str, default=None)
    p.add_argument(
        "--hf_home",
        type=str,
        default=None,
        help="Optional HF cache directory (sets HF_HOME when loading models from the Hub).",
    )
    return p


def main() -> None:
    # Line-buffer stdout when supported (batch jobs often use fully buffered stdout otherwise).
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    args = build_arg_parser().parse_args()
    if not args.wandb_run_name:
        args.wandb_run_name = datetime.now().strftime("sft-%Y%m%d-%H%M%S")
    try:
        import wandb as _wandb_check
    except ImportError:
        print("ERROR: the `wandb` package is required. Install with: uv pip install wandb", flush=True)
        sys.exit(1)
    del _wandb_check
    _emit(
        f"sft_experiment: parsed CLI (project={args.wandb_project!r}, run={args.wandb_run_name!r}) — "
        "entering train()",
    )
    train(args)


if __name__ == "__main__":
    main()
