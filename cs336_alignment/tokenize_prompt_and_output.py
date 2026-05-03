import torch
from transformers import PreTrainedTokenizerBase

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).
    """
    batch_size = len(prompt_strs)
    
    prompt_ids_list = []
    output_ids_list = []
    
    # Wait, does the tokenizer add special tokens?
    # Usually for these alignment tasks, we tokenize prompt with special tokens,
    # and output without special tokens (or vice versa).
    # Let's try encode(..., add_special_tokens=False) as a baseline.
    for prompt, output in zip(prompt_strs, output_strs):
        # We might need to handle tokenizer properly
        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        o_ids = tokenizer.encode(output, add_special_tokens=False)
        prompt_ids_list.append(torch.tensor(p_ids, dtype=torch.long))
        output_ids_list.append(torch.tensor(o_ids, dtype=torch.long))
        
    lens = [len(p) + len(o) for p, o in zip(prompt_ids_list, output_ids_list)]
    max_len = max(lens)
    
    input_ids = torch.full((batch_size, max_len - 1), tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0, dtype=torch.long)
    labels = torch.full((batch_size, max_len - 1), tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0, dtype=torch.long)
    response_mask = torch.zeros((batch_size, max_len - 1), dtype=torch.long)
    
    for i, (p_ids, o_ids) in enumerate(zip(prompt_ids_list, output_ids_list)):
        seq = torch.cat([p_ids, o_ids])
        seq_len = len(seq)
        
        # Pad to max_len
        pad_len = max_len - seq_len
        if pad_len > 0:
            pad_tensor = torch.full((pad_len,), tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0, dtype=torch.long)
            seq_padded = torch.cat([seq, pad_tensor])
        else:
            seq_padded = seq
            
        input_ids[i] = seq_padded[:-1]
        labels[i] = seq_padded[1:]
        
        mask_start = len(p_ids) - 1
        mask_end = mask_start + len(o_ids)
        response_mask[i, mask_start:mask_end] = 1
        
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }
