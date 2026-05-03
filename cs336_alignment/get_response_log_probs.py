from typing import Optional
import torch
from transformers import PreTrainedModel
import torch.nn.functional as F
from cs336_alignment.compute_entropy import compute_entropy

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Get per-token conditional log-probabilities (given the previous tokens) from a causal language model,
    and optionally the entropy of the model's next-token distribution.
    
    Args:
        model: PreTrainedModel HuggingFace model used for scoring.
        input_ids: torch.Tensor shape (batch_size, sequence_length).
        labels: torch.Tensor shape (batch_size, sequence_length).
        return_token_entropy: bool If True, also return per-token entropy.
        
    Returns:
        dict[str, torch.Tensor] containing:
            "log_probs": shape (batch_size, sequence_length)
            "token_entropy": optional, shape (batch_size, sequence_length)
    """
    logits = model(input_ids).logits
    log_probs_dist = F.log_softmax(logits, dim=-1)
    
    # gather log-probabilities using labels (which are shifted input_ids)
    # log_probs_dist: (batch_size, sequence_length, vocab_size)
    # labels: (batch_size, sequence_length) -> unsqueeze to (batch_size, sequence_length, 1)
    log_probs = log_probs_dist.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    
    result = {"log_probs": log_probs}
    
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
        
    return result
