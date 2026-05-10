from __future__ import annotations

from collections.abc import Callable

import torch


def compute_group_normalized_rewards(
	reward_fn: Callable[[str, str], dict[str, float]],
	rollout_responses: list[str],
	repeated_ground_truths: list[str],
	group_size: int,
	advantage_eps: float,
	normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
	if len(rollout_responses) != len(repeated_ground_truths):
		raise ValueError("rollout_responses and repeated_ground_truths must have the same length")
	if group_size <= 0:
		raise ValueError("group_size must be positive")
	if len(rollout_responses) % group_size != 0:
		raise ValueError("rollout_responses length must be divisible by group_size")

	rewards: list[float] = []
	format_rewards: list[float] = []
	answer_rewards: list[float] = []

	for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
		scores = reward_fn(response, ground_truth)
		rewards.append(float(scores["reward"]))
		format_rewards.append(float(scores["format_reward"]))
		answer_rewards.append(float(scores["answer_reward"]))

	raw_rewards = torch.tensor(rewards, dtype=torch.float32)
	advantages = torch.empty_like(raw_rewards)

	num_groups = raw_rewards.numel() // group_size
	group_rewards = raw_rewards.view(num_groups, group_size)

	group_means = group_rewards.mean(dim=1, keepdim=True)
	if group_size > 1:
		group_stds = group_rewards.std(dim=1, keepdim=True, unbiased=True)
	else:
		group_stds = torch.zeros_like(group_means)

	if normalize_by_std:
		normalized = (group_rewards - group_means) / (group_stds + advantage_eps)
	else:
		normalized = group_rewards - group_means

	advantages.copy_(normalized.reshape(-1))

	format_reward_tensor = torch.tensor(format_rewards, dtype=torch.float32)
	answer_reward_tensor = torch.tensor(answer_rewards, dtype=torch.float32)
	metadata = {
		"reward_mean": float(raw_rewards.mean().item()),
		"reward_std": float(raw_rewards.std(unbiased=True).item()) if raw_rewards.numel() > 1 else 0.0,
		"reward_min": float(raw_rewards.min().item()),
		"reward_max": float(raw_rewards.max().item()),
		"format_reward_mean": float(format_reward_tensor.mean().item()),
		"answer_reward_mean": float(answer_reward_tensor.mean().item()),
		"group_size": float(group_size),
		"num_groups": float(num_groups),
		"normalize_by_std": float(bool(normalize_by_std)),
		"group_std_mean": float(group_stds.mean().item()),
	}

	return advantages, raw_rewards, metadata


def run_compute_group_normalized_rewards(
	reward_fn: Callable[[str, str], dict[str, float]],
	rollout_responses: list[str],
	repeated_ground_truths: list[str],
	group_size: int,
	advantage_eps: float,
	normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
	return compute_group_normalized_rewards(
		reward_fn=reward_fn,
		rollout_responses=rollout_responses,
		repeated_ground_truths=repeated_ground_truths,
		group_size=group_size,
		advantage_eps=advantage_eps,
		normalize_by_std=normalize_by_std,
	)
