import torch
import torch.nn.functional as F

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).
    
    Args:
        logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
            containing unnormalized logits.
            
    Returns:
        torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token
            prediction.
    """
    # Numerically stable entropy using log_softmax
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    # -sum(p * log(p))
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy
