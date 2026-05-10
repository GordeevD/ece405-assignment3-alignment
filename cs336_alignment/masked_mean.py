import torch

def masked_mean(
	tensor: torch.Tensor,
	mask: torch.Tensor,
	dim: int | None = None,
) -> torch.Tensor:
	"""Compute the mean of ``tensor`` along ``dim`` considering only
	elements where ``mask`` is truthy (non-zero / True).

	Args:
		tensor: Tensor of values to average.
		mask: Boolean or numeric mask of the same shape as ``tensor``.
		dim: Dimension to average over. If ``None``, average over all
			masked elements and return a scalar tensor.

	Returns:
		Tensor with the masked mean, following ``torch.mean`` semantics
		for the ``dim`` argument (i.e., reduced dimension removed).
	"""
	mask_tensor = mask.to(dtype=tensor.dtype)
	weighted = tensor * mask_tensor
	if dim is None:
		total = weighted.sum()
		count = mask_tensor.sum()
		return total / count
	summed = weighted.sum(dim=dim)
	counts = mask_tensor.sum(dim=dim)
	return summed / counts
