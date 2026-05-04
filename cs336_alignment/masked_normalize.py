import torch


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant, considering only
    elements where the mask is truthy (e.g. boolean True or numeric 1).

    Args:
        tensor: The tensor to sum and normalize.
        mask: Same shape as ``tensor``; included positions are non-zero / True.
        normalize_constant: Value to divide the masked sum by.
        dim: Dimension to sum along. If ``None``, sum over all dimensions.

    Returns:
        The masked sum divided by ``normalize_constant``.
    """
    weighted = tensor * mask.to(dtype=tensor.dtype)
    summed = weighted.sum() if dim is None else weighted.sum(dim=dim)
    return summed / normalize_constant
