from __future__ import annotations

import torch


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run forward (NLL) and backward for one SFT microbatch.

    The loss is mean negative log-probability over the batch, with the usual
    1/gradient_accumulation_steps scaling so microbatch gradients sum to a full
    optimizer step. ``normalize_constant`` divides the loss (e.g. Dr. GRPO-style
    fixed normalizers).
    """
    batch_size = policy_log_probs.shape[0]
    masked_neg_log_prob_sum = -(policy_log_probs * response_mask).sum()
    denom = batch_size * gradient_accumulation_steps * normalize_constant
    loss = masked_neg_log_prob_sum / denom
    loss.backward()
    metadata: dict[str, torch.Tensor] = {
        "masked_neg_log_prob_sum": masked_neg_log_prob_sum.detach(),
        "response_token_count": response_mask.sum().detach(),
    }
    return loss.detach(), metadata
