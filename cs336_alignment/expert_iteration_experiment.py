"""
Supervised fine-tuning on MATH-style reasoning traces with periodic vLLM evaluation.

Designed for two GPUs: policy on ``cuda:0``, vLLM engine on ``cuda:1`` (see ``--policy_device`` / ``--vllm_device``).

JSON training files may be either JSONL or a single JSON array; array files are parsed in chunks
(so capped ``--max_train_examples`` runs do not read the whole file, and full runs avoid one
giant in-memory string).

Example::

    HF_HOME="$(pwd)/.hf_cache" uv run python -m cs336_alignment.sft_experiment \\
        --wandb_project my-ece405-runs \\
        --wandb_run_name sft-qwen-math-01 \\
        --model_path ../Qwen/Qwen2.5-0.5B \\
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
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, TextIO
from unittest.mock import patch

import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams

try:
    from vllm.model_executor import set_random_seed as vllm_set_random_seed
except ImportError:  # newer vLLM (e.g. Metal builds): moved to torch_utils
    from vllm.utils.torch_utils import set_random_seed as vllm_set_random_seed

from cs336_alignment.drgrpo_grader import grade, r1_zero_reward_fn
from cs336_alignment.get_response_log_probs import get_response_log_probs
from cs336_alignment.sft_microbatch_train_step import sft_microbatch_train_step
from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output


def _emit(msg: str) -> None:
    print(msg, flush=True)


def _check_system_resources(label: str = "") -> None:
    """Log current system resource usage."""
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_percent = process.memory_percent()
        cpu_percent = process.cpu_percent(interval=0.1)
        label_str = f" [{label}]" if label else ""
        _emit(f"RESOURCES{label_str}: Memory={mem_info.rss / (1024**3):.1f}GB ({mem_percent:.1f}%), CPU={cpu_percent:.1f}%")
    except ImportError:
        pass  # psutil not available
    except Exception as e:
        pass  # silently skip if monitoring fails


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


def _write_final_result(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON result row for this run."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


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

    # Some vLLM builds (Metal, newer versions) do not expose the same internal
    # module paths. Create the profiling patch only if the target exists to
    # avoid AttributeError during patch creation/entering.
    profiling_patch = None
    try:
        import importlib
        import contextlib

        module_name = "vllm.worker.worker"
        try:
            mod = importlib.import_module(module_name)
            if hasattr(mod.Worker, "_assert_memory_footprint_increased_during_profiling"):
                profiling_patch = patch(f"{module_name}.Worker._assert_memory_footprint_increased_during_profiling", return_value=None)
        except Exception:
            profiling_patch = None

        with contextlib.ExitStack() as stack:
            stack.enter_context(world_size_patch)
            if profiling_patch is not None:
                stack.enter_context(profiling_patch)
            try:
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
            except TypeError as e:
                _emit(f"vLLM init TypeError: {e}; retrying minimal kwargs without 'device' or memory args")
                # Some vLLM versions (or Metal builds) have a different EngineArgs
                # signature that rejects 'device' or 'gpu_memory_utilization'. Retry
                # with a minimal set of supported kwargs and let vLLM choose defaults.
                try:
                    return LLM(
                        model=model_id,
                        dtype=torch.bfloat16,
                        enable_prefix_caching=True,
                        trust_remote_code=True,
                        max_model_len=max_model_len,
                        max_num_seqs=max_num_seqs,
                    )
                except Exception as e2:
                    _emit(f"vLLM init fallback failed: {type(e2).__name__}: {e2}")
                    raise
    except Exception:
        # Fallback: if anything unexpected happens while attempting to patch,
        # just start vLLM without the profiling patch.
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
    # vLLM internals vary across versions and builds (Metal vs CUDA). Try a
    # few common attribute paths and pick the first that exists.
    tried = []
    def _try_path(getter, path_name):
        try:
            target = getter()
        except Exception:
            tried.append(path_name)
            return False
        try:
            target.load_weights(state_dict.items())
            _emit(f"Loaded policy weights into vLLM via {path_name}")
            return True
        except Exception as e:
            # If load_weights doesn't exist or fails, record and continue
            _emit(f"Failed to load weights via {path_name}: {type(e).__name__}: {e}")
            tried.append(path_name)
            return False

    # Candidate accessors (order: historically observed implementations)
    candidates = [
        (lambda: llm.llm_engine.model_executor.driver_worker.model_runner.model, "llm.llm_engine.model_executor.driver_worker.model_runner.model"),
        (lambda: llm.llm_engine.model_runner.model, "llm.llm_engine.model_runner.model"),
        (lambda: llm.engine.model_runner.model, "llm.engine.model_runner.model"),
        (lambda: llm._model.model, "llm._model.model"),
        (lambda: getattr(llm, "model", None), "llm.model"),
    ]

    for getter, name in candidates:
        if _try_path(getter, name):
            return

    # Try some generic vLLM-facing helpers if present
    generic_tries = [
        ("load_weights", lambda obj: getattr(obj, "load_weights", None)),
        ("load_state_dict", lambda obj: getattr(obj, "load_state_dict", None)),
        ("load_model_weights", lambda obj: getattr(obj, "load_model_weights", None)),
    ]
    for fn_name, resolver in generic_tries:
        fn = resolver(llm)
        if fn is not None:
            try:
                _emit(f"Attempting generic vLLM weight load via llm.{fn_name}(...)")
                # Try passing the full state_dict first, then items() as fallback
                try:
                    fn(state_dict)
                except TypeError:
                    fn(list(state_dict.items()))
                _emit(f"Loaded policy weights via llm.{fn_name}")
                return
            except Exception as e:
                _emit(f"Generic vLLM weight load via {fn_name} failed: {type(e).__name__}: {e}")

    # As a last resort, do not raise — continue using vLLM's own loaded weights.
    _emit(
        "WARNING: Unable to load HF policy weights into vLLM for this build. "
        "Continuing with vLLM's original model (weights may differ)."
    )
    return


def _load_json_array_streaming(
    f: TextIO,
    path: Path,
    *,
    max_records: int | None,
    progress_cb: Callable[[str], None] | None,
    progress_every: int,
) -> list[dict[str, Any]]:
    """Parse a top-level JSON array of objects without loading the whole file into RAM.

    The file handle must be positioned immediately after the opening ``[`` (that byte
    must already have been consumed). Stops early when ``max_records`` is reached (skips
    reading/decoding the remainder of the file).
    """
    decoder = json.JSONDecoder()
    buf = ""
    idx = 0
    chunk_size = 4 * 1024 * 1024
    compact_threshold = 1 * 1024 * 1024
    out: list[dict[str, Any]] = []
    n_items = 0
    started_at = time.time()
    last_progress_at = started_at
    progress_timeout = 60  # seconds without progress before warning
    bytes_read = 0
    ended_with_bracket = False
    reached_eof = False

    def _skip_ws() -> None:
        nonlocal idx
        while idx < len(buf) and buf[idx].isspace():
            idx += 1

    def refill() -> bool:
        nonlocal buf, idx, bytes_read, reached_eof
        if idx >= compact_threshold:
            buf = buf[idx:]
            idx = 0
        chunk = f.read(chunk_size)
        if chunk:
            bytes_read += len(chunk)
            buf += chunk
            return True
        reached_eof = True
        return False

    while True:
        _skip_ws()
        if idx >= len(buf):
            if not refill():
                break
            continue

        if buf[idx] == "]":
            idx += 1
            ended_with_bracket = True
            break

        try:
            obj, end = decoder.raw_decode(buf, idx)
        except json.JSONDecodeError:
            if not refill():
                raise ValueError(f"Incomplete or invalid JSON array near entry {n_items + 1} in {path}") from None
            continue

        if not isinstance(obj, dict):
            raise ValueError(f"Expected object entries in JSON array: {path}")
        out.append(obj)
        n_items += 1
        idx = end
        if progress_cb is not None and n_items % progress_every == 0:
            elapsed = max(time.time() - started_at, 1e-9)
            mib = bytes_read / (1024 * 1024)  # UTF-8 chars ≈ bytes for ASCII-heavy JSON
            progress_cb(
                f"parsed {n_items} JSON array rows ({n_items / elapsed:.1f} rows/s, ~{mib:.1f} MiB read)"
            )

        _skip_ws()
        if idx < len(buf) and buf[idx] == ",":
            idx += 1

        if max_records is not None and n_items >= max_records:
            if progress_cb is not None:
                elapsed = max(time.time() - started_at, 1e-9)
                progress_cb(
                    f"stopped JSON array load early at max_records={max_records} "
                    f"({n_items} rows, {n_items / elapsed:.1f} rows/s)"
                )
            return out

    if max_records is None and not ended_with_bracket:
        raise ValueError(f"JSON array in {path} ended before closing ']' (incomplete file or truncated JSON)")

    if max_records is None:
        # Verify no trailing content after the closing bracket
        _skip_ws()
        if idx < len(buf):
            raise ValueError(f"Trailing content after JSON array in {path}: {buf[idx:idx + 120]!r}")
        
        # Continue checking for trailing content if we haven't reached EOF yet
        attempts = 0
        max_attempts = 100  # ~100 chunks = ~400MB max verification
        while not reached_eof and attempts < max_attempts:
            if not refill():
                break
            attempts += 1
            _skip_ws()
            if idx < len(buf):
                raise ValueError(f"Trailing content after JSON array in {path}: {buf[idx:idx + 120]!r}")
        
        if attempts >= max_attempts and not reached_eof:
            if progress_cb is not None:
                progress_cb(f"WARNING: JSON verification stopped after {attempts} chunks (possible very large file or infinite stream)")

    if progress_cb is not None:
        elapsed = max(time.time() - started_at, 1e-9)
        progress_cb(f"completed JSON array load: {n_items} rows ({n_items / elapsed:.1f} rows/s)")
    return out


def load_sft_records(
    path: Path,
    *,
    progress_cb: Callable[[str], None] | None = None,
    progress_every: int = 500,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if progress_cb is not None:
        progress_cb(f"reading records from {path}")

    # Check if file exists and is readable
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    file_size = path.stat().st_size
    if progress_cb is not None:
        progress_cb(f"file size: {file_size / (1024*1024):.1f} MiB")

    with path.open("r", encoding="utf-8") as f:
        first_non_ws: str | None = None
        while True:
            ch = f.read(1)
            if ch == "":
                break
            if not ch.isspace():
                first_non_ws = ch
                break

        if first_non_ws is None:
            return []

        # JSON array mode: chunked read + incremental decode (bounded RAM; optional early stop).
        if first_non_ws == "[":
            if progress_cb is not None:
                progress_cb(
                    "detected JSON array format; streaming parse"
                    + (f" (max_records={max_records})" if max_records is not None else "")
                )
            return _load_json_array_streaming(
                f,
                path,
                max_records=max_records,
                progress_cb=progress_cb,
                progress_every=progress_every,
            )

        # JSONL mode: stream with progress updates.
        out: list[dict[str, Any]] = []
        first_line = (first_non_ws + f.readline()).strip()
        n_lines = 0
        started = time.time()
        if first_line:
            out.append(json.loads(first_line))
            n_lines = 1
            if max_records is not None and n_lines >= max_records:
                if progress_cb is not None:
                    progress_cb(f"stopped JSONL load early at max_records={max_records}")
                return out
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            n_lines += 1
            if max_records is not None and n_lines >= max_records:
                if progress_cb is not None:
                    progress_cb(f"stopped JSONL load early at max_records={max_records}")
                return out
            if progress_cb is not None and n_lines % progress_every == 0:
                elapsed = max(time.time() - started, 1e-9)
                rate = n_lines / elapsed
                progress_cb(f"loaded {n_lines} JSONL rows ({rate:.1f} rows/s)")
        if progress_cb is not None:
            elapsed = max(time.time() - started, 1e-9)
            rate = n_lines / elapsed if n_lines else 0.0
            progress_cb(f"completed JSONL load: {n_lines} rows ({rate:.1f} rows/s)")
        return out


def build_prompt(problem: str, template: str) -> str:
    return template.replace("{question}", problem)


def load_val_eval_pairs(
    val_path: Path,
    *,
    progress_cb: Callable[[str], None] | None = None,
    progress_every: int = 500,
) -> tuple[list[str], list[str]]:
    """Load validation problems and ground-truth answers from ``sft_val.jsonl``-style JSON.

    Each record must include ``problem`` and ``expected_answer`` (same schema as
    ``cs336_alignment/sft_val.jsonl``).
    """
    try:
        records = load_sft_records(
            val_path,
            progress_cb=progress_cb,
            progress_every=progress_every,
        )
    except Exception as e:
        if progress_cb is not None:
            progress_cb(f"ERROR loading validation records: {type(e).__name__}: {e}")
        raise
    
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


def _vllm_generate(llm: LLM, prompts: list[str], params: SamplingParams):
    """Generate with vLLM while suppressing internal tqdm logs when supported."""
    try:
        return llm.generate(prompts, params, use_tqdm=False)
    except TypeError:
        # Older/newer vLLM builds may not support use_tqdm kwarg.
        return llm.generate(prompts, params)


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
    _emit(f"[eval] starting {desc}: {len(problems)} problems, batch_size={eval_batch_size}")
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
        outputs = _vllm_generate(llm, batch_p, params)
        for out, gt in zip(outputs, batch_gt, strict=True):
            text = out.outputs[0].text
            rew = r1_zero_reward_fn(text, gt)
            total += 1
            if rew.get("answer_reward", 0.0) >= 1.0:
                correct += 1
        if show_progress and total:
            acc_so_far = correct / total
            batch_iter.set_postfix(acc=f"{100.0 * acc_so_far:.1f}%", ok=f"{correct}/{total}")
    _emit(f"[eval] completed {desc}: acc={100.0 * (correct / max(total, 1)):.2f}% ({correct}/{total})")
    return {"accuracy": correct / max(total, 1), "n": float(total)}


def _wandb_init(args: argparse.Namespace, extra_config: dict[str, Any]) -> Any:
    """Start a W&B run and bind custom x-axes (explicit metric names; globs are unreliable)."""
    import wandb
    import wandb.errors

    cfg = {**vars(args), **extra_config}
    entity = args.wandb_entity or os.environ.get("WANDB_ENTITY")
    init_kw: dict[str, Any] = {
        "project": args.wandb_project,
        "name": args.wandb_run_name,
        "config": cfg,
        "settings": wandb.Settings(
            init_timeout=args.wandb_init_timeout,
        ),
    }
    if entity:
        init_kw["entity"] = entity

    _emit(f"Weights & Biases: init_timeout={args.wandb_init_timeout}s (WANDB fallback={args.wandb_offline_on_comm_error})")
    try:
        run = wandb.init(**init_kw)
    except wandb.errors.CommError as e:
        if not args.wandb_offline_on_comm_error:
            raise
        _emit(
            f"Weights & Biases: online init failed ({type(e).__name__}: {e}); "
            "retrying with mode='offline' (train continues; sync later via `wandb sync`)."
        )
        init_kw_retry = dict(init_kw)
        init_kw_retry["settings"] = wandb.Settings(init_timeout=max(args.wandb_init_timeout, 120), mode="offline")
        run = wandb.init(**init_kw_retry)
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
    _emit(f"Weights & Biases mode (WANDB_MODE env): {mode}")
    url = getattr(run, "url", None)
    if url:
        _emit(f"Weights & Biases run URL: {url}")
    elif mode == "offline":
        _emit("Weights & Biases: offline run (sync later with `wandb sync` on the run directory).")
    else:
        _emit("Weights & Biases: run started (no public URL yet).")

    return run



def train(args: argparse.Namespace) -> None:
    """Run the expert iteration experiment."""
    step_log = [0]
    
    # Check resources periodically
    _check_system_resources("init")
    
    # 1. Load the initial dataset D (supports JSONL or top-level JSON array)
    # Use the same robust loader as in sft_experiment to handle both formats.
    _next_terminal_step(step_log, "loading training data")
    records = load_sft_records(
        Path(args.train_json),
        max_records=getattr(args, "max_train_examples", None),
        progress_cb=lambda msg: _emit(msg),
        progress_every=500,
    )
            
    val_problems, val_gts = load_val_eval_pairs(Path(args.val_json))
            
    _emit(f"[PHASE 1] Initializing Model & Tokenizer on {args.policy_device}")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.utils.data import DataLoader
    from cs336_alignment.sft_experiment import iterate_batches
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    # Avoid forcing FlashAttention2 (may not be installed). Use `dtype` instead
    # of deprecated `torch_dtype`.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.to(args.policy_device)
    
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
        _emit("[progress] baseline evaluation: syncing policy -> vLLM")
        load_policy_into_vllm_instance(model, llm)
        _emit("[progress] baseline evaluation: running validation")
        eval_metrics = evaluate_math_vllm(llm, prompt_template, val_problems, val_gts, args.eval_max_tokens, args.eval_batch_size)
    if not args.no_wandb:
        wandb.log({"eval/accuracy": eval_metrics["accuracy"], "global_train_step": global_train_step})
    _emit(f"Baseline Eval Accuracy: {eval_metrics['accuracy']*100.0:.2f}%")

    for ei_step in range(1, args.ei_steps + 1):
        _emit(f"\n--- EXPERT ITERATION STEP {ei_step}/{args.ei_steps} ---")
        
        # Sample Db
        Db = random.sample(records, k=args.ei_batch_size)
        
        # Load the latest policy weights into vLLM engine
        _emit(f"[progress] EI {ei_step}/{args.ei_steps}: syncing policy -> vLLM")
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

        _emit(
            f"[progress] EI {ei_step}/{args.ei_steps}: generation started "
            f"({len(prompts)} prompts x {args.g_rollouts} rollouts)"
        )
        batch_generations = _vllm_generate(llm, prompts, params)
        _emit(f"[progress] EI {ei_step}/{args.ei_steps}: generation completed")
        for gen, ex in zip(batch_generations, Db):
            for out in gen.outputs:
                generated_responses.append(out.text)
                expected_gts.append(ex.get("expected_answer"))
        
        # Filter correct using reward function
        _emit("Filtering rollouts by reward...")
        sft_prompts = []
        sft_responses = []
        
        # Reconstruct which prompt corresponds to each response
        # (generated_responses and expected_gts repeat prompts G times)
        prompt_idx = 0
        resp_idx = 0
        for gen_idx, gen in enumerate(batch_generations):
            for rollout_idx, out in enumerate(gen.outputs):
                resp = out.text
                gt = expected_gts[resp_idx]
                q = prompts[gen_idx]
                resp_idx += 1
                
                # Add back closing </answer> tag since vLLM stops at it but excludes it by default
                resp_with_stop = resp + "</answer>"
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
        _emit(
            f"[progress] EI {ei_step}/{args.ei_steps}: SFT started "
            f"({len(dataset)} examples, epochs={args.sft_epochs})"
        )
        
        for epoch in range(args.sft_epochs):
            model.train()
            _emit(f"[progress] EI {ei_step}/{args.ei_steps}: SFT epoch {epoch + 1}/{args.sft_epochs}")
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
        _emit(f"[progress] EI {ei_step}/{args.ei_steps}: SFT completed")
            
        # Eval after EI step
        with torch.no_grad():
            _emit(f"[progress] EI {ei_step}/{args.ei_steps}: post-step eval syncing policy -> vLLM")
            load_policy_into_vllm_instance(model, llm)
            _emit(f"[progress] EI {ei_step}/{args.ei_steps}: post-step eval running validation")
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
