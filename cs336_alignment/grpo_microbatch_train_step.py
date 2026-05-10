from __future__ import annotations

from typing import Literal

import torch

from cs336_alignment.masked_mean import masked_mean
from cs336_alignment.compute_policy_gradient_loss import compute_policy_gradient_loss


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Execute a forward-and-backward pass on a microbatch for GRPO.

    The function computes a per-token policy-gradient loss using
    ``compute_policy_gradient_loss``, averages across response tokens using
    ``masked_mean``, then averages across the batch and scales the loss for
    gradient accumulation before calling ``backward()``.
    """
    # Compute per-token loss and metadata from the requested loss function
    loss_or_tuple = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    # compute_policy_gradient_loss returns either (loss, metadata) or loss
    if isinstance(loss_or_tuple, tuple):
        per_token_loss, inner_metadata = loss_or_tuple
    else:
        per_token_loss = loss_or_tuple
        inner_metadata = {}

    # Mask and average: first mean across sequence for each example,
    # considering only response tokens, then mean across the batch.
    per_example_mean = masked_mean(per_token_loss, response_mask, dim=1)
    loss = per_example_mean.mean() / float(gradient_accumulation_steps)

    # Backpropagate scaled loss
    loss.backward()

    metadata: dict[str, torch.Tensor] = {
        "per_token_loss": per_token_loss.detach(),
        "per_example_mean": per_example_mean.detach(),
        "response_token_count": response_mask.sum().detach(),
    }
    # Merge inner metadata if present
    for k, v in inner_metadata.items():
        # ensure tensors are detached for metadata
        try:
            metadata[k] = v.detach()
        except Exception:
            metadata[k] = v

    return loss.detach(), metadata
