from __future__ import annotations

from typing import Literal

import torch


def compute_policy_gradient_loss(
	policy_log_probs: torch.Tensor,
	loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_noclip"],
	raw_rewards: torch.Tensor | None = None,
	advantages: torch.Tensor | None = None,
	old_log_probs: torch.Tensor | None = None,
	cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
	"""Dispatch to the requested policy-gradient loss implementation."""
	from cs336_alignment.compute_grpo_clip_loss import compute_grpo_clip_loss
	from cs336_alignment.compute_naive_policy_gradient_loss import compute_naive_policy_gradient_loss

	if loss_type == "no_baseline":
		if raw_rewards is None:
			raise ValueError("raw_rewards is required when loss_type='no_baseline'")
		return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}

	if loss_type == "reinforce_with_baseline":
		if advantages is None:
			raise ValueError("advantages is required when loss_type='reinforce_with_baseline'")
		return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}

	if loss_type == "grpo_clip":
		if advantages is None:
			raise ValueError("advantages is required when loss_type='grpo_clip'")
		if old_log_probs is None:
			raise ValueError("old_log_probs is required when loss_type='grpo_clip'")
		if cliprange is None:
			raise ValueError("cliprange is required when loss_type='grpo_clip'")
		return compute_grpo_clip_loss(
			advantages=advantages,
			policy_log_probs=policy_log_probs,
			old_log_probs=old_log_probs,
			cliprange=cliprange,
		)

	if loss_type == "grpo_noclip":
		if advantages is None:
			raise ValueError("advantages is required when loss_type='grpo_noclip'")
		if old_log_probs is None:
			raise ValueError("old_log_probs is required when loss_type='grpo_noclip'")
		ratios = torch.exp(policy_log_probs - old_log_probs.to(policy_log_probs.device))
		if advantages.ndim == 1:
			advantages = advantages.unsqueeze(1).to(policy_log_probs.dtype).to(policy_log_probs.device)
		else:
			advantages = advantages.to(policy_log_probs.dtype).to(policy_log_probs.device)
		loss = - (ratios * advantages.expand_as(policy_log_probs))
		return loss, {"ratio": ratios}

	raise ValueError(f"Unsupported loss_type: {loss_type}")
