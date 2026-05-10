from __future__ import annotations

import torch


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token naive policy-gradient loss.

    Args:
        raw_rewards_or_advantages: torch.Tensor shape (batch_size, 1)
            scalar reward or advantage for each rollout response.
        policy_log_probs: torch.Tensor shape (batch_size, sequence_length)
            log-probabilities for each token.

    Returns:
        torch.Tensor shape (batch_size, sequence_length): per-token
        policy-gradient loss (to be aggregated later).
    """
    # Ensure shapes are compatible and on same device/dtype.
    if raw_rewards_or_advantages.ndim == 1:
        raw_rewards_or_advantages = raw_rewards_or_advantages.unsqueeze(1)

    # Broadcast advantage/reward across sequence length
    # raw_rewards_or_advantages: (batch, 1) -> (batch, seq_len)
    advantages = raw_rewards_or_advantages.to(dtype=policy_log_probs.dtype, device=policy_log_probs.device)
    advantages = advantages.expand(-1, policy_log_probs.size(1))

    # Policy gradient (REINFORCE) per-token loss: - advantage * log_prob
    loss = -advantages * policy_log_probs

    return loss
