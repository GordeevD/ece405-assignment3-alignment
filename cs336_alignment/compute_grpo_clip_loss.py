from __future__ import annotations

import torch


def compute_grpo_clip_loss(
	advantages: torch.Tensor,
	policy_log_probs: torch.Tensor,
	old_log_probs: torch.Tensor,
	cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
	"""Compute the per-token GRPO-Clip loss.

	Args:
		advantages: Tensor of shape (batch_size, 1) with per-example advantages.
		policy_log_probs: Tensor of shape (batch_size, sequence_length) with
			per-token log-probs from the policy being trained.
		old_log_probs: Tensor of shape (batch_size, sequence_length) with
			per-token log-probs from the old policy.
		cliprange: Clip parameter epsilon.

	Returns:
		A tuple of (loss, metadata), where loss has shape
		(batch_size, sequence_length).
	"""
	if advantages.ndim == 1:
		advantages = advantages.unsqueeze(1)

	advantages = advantages.to(dtype=policy_log_probs.dtype, device=policy_log_probs.device)
	old_log_probs = old_log_probs.to(dtype=policy_log_probs.dtype, device=policy_log_probs.device)

	broadcast_advantages = advantages.expand_as(policy_log_probs)
	ratios = torch.exp(policy_log_probs - old_log_probs)
	clipped_ratios = torch.clamp(ratios, 1.0 - cliprange, 1.0 + cliprange)

	unclipped_objective = ratios * broadcast_advantages
	clipped_objective = clipped_ratios * broadcast_advantages

	loss = -torch.minimum(unclipped_objective, clipped_objective)
	is_clipped = clipped_objective < unclipped_objective

	metadata = {
		"ratio": ratios,
		"clipped_ratio": clipped_ratios,
		"is_clipped": is_clipped,
	}
	return loss, metadata
