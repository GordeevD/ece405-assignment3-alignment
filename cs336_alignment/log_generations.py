from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def log_generations(
    prompts: Sequence[str],
    generations: Sequence[str],
    ground_truths: Sequence[str],
    rewards: Sequence[Mapping[str, float]],
    response_mean_entropies: Sequence[float],
    response_lengths: Sequence[int] | None = None,
    *,
    step: int | None = None,
    log_key: str = "samples/generations",
    max_rows: int = 16,
    max_chars: int = 4000,
) -> None:
    """Log prompts, model outputs, labels, rewards, entropy, and length stats (e.g. to W&B).

    Per logged row, includes: prompt, generation, ground-truth answer, format / answer /
    total reward, mean token entropy over the response, and response length.

    Also logs batch-level averages: mean response length (all), mean length among
    answer-correct examples (``answer_reward >= 1``), and mean length among the rest.

    Args:
        prompts: Input prompts (one per example).
        generations: Decoded model completions.
        ground_truths: Reference answers for grading / inspection.
        rewards: One mapping per example with ``format_reward``, ``answer_reward``, and
            ``reward`` (total), as produced by graders such as ``r1_zero_reward_fn``.
        response_mean_entropies: Mean per-token entropy over response tokens for each example.
        response_lengths: Token counts per response. If omitted, character length of each
            generation is used as a proxy (prefer passing real token counts from the tokenizer).
        step: Optional global step for ``wandb.log``.
        log_key: W&B key for the example table.
        max_rows: Max table rows (full batch still used for aggregate length metrics).
        max_chars: Truncate long string cells in the table.
    """
    n = len(prompts)
    if not (len(generations) == len(ground_truths) == len(rewards) == len(response_mean_entropies) == n):
        raise ValueError(
            "prompts, generations, ground_truths, rewards, and response_mean_entropies "
            f"must have the same length; got {n}, {len(generations)}, {len(ground_truths)}, "
            f"{len(rewards)}, {len(response_mean_entropies)}"
        )
    if response_lengths is not None and len(response_lengths) != n:
        raise ValueError(
            f"response_lengths must have length {n} when provided; got {len(response_lengths)}"
        )
    if n == 0:
        return

    lengths: list[int] = (
        list(response_lengths) if response_lengths is not None else [len(g) for g in generations]
    )

    correct = [float(r.get("answer_reward", 0.0)) >= 1.0 for r in rewards]
    correct_lens = [lengths[i] for i in range(n) if correct[i]]
    incorrect_lens = [lengths[i] for i in range(n) if not correct[i]]

    avg_len_all = sum(lengths) / n
    avg_len_correct = sum(correct_lens) / len(correct_lens) if correct_lens else math.nan
    avg_len_incorrect = sum(incorrect_lens) / len(incorrect_lens) if incorrect_lens else math.nan

    rows: list[list[Any]] = []
    for i in range(min(n, max_rows)):
        r = rewards[i]
        fmt_r = float(r.get("format_reward", float("nan")))
        ans_r = float(r.get("answer_reward", float("nan")))
        tot_r = float(r.get("reward", float("nan")))
        rows.append(
            [
                _truncate(prompts[i], max_chars),
                _truncate(generations[i], max_chars),
                _truncate(ground_truths[i], max_chars),
                fmt_r,
                ans_r,
                tot_r,
                float(response_mean_entropies[i]),
                int(lengths[i]),
            ]
        )

    try:
        import wandb
    except ImportError:
        logger.debug("log_generations: wandb not installed; skipping")
        return

    if wandb.run is None:
        logger.debug("log_generations: no active wandb run; skipping")
        return

    table = wandb.Table(
        columns=[
            "prompt",
            "generation",
            "ground_truth",
            "format_reward",
            "answer_reward",
            "reward",
            "mean_response_token_entropy",
            "response_length",
        ],
        data=rows,
    )
    payload: dict[str, Any] = {
        log_key: table,
        "samples/avg_response_length": avg_len_all,
    }
    if not math.isnan(avg_len_correct):
        payload["samples/avg_response_length_correct"] = avg_len_correct
    if not math.isnan(avg_len_incorrect):
        payload["samples/avg_response_length_incorrect"] = avg_len_incorrect

    if step is not None:
        wandb.log(payload, step=step)
    else:
        wandb.log(payload)
