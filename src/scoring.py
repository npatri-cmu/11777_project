import torch

def answer_logprob(logits, input_ids, prompt_length, reduction='sum'):
    """Score answer tokens only, shifted causally; excludes prompt and EOS."""
    n = input_ids.shape[1]
    if not 0 < prompt_length < n or logits.shape[:2] != input_ids.shape:
        raise ValueError('Invalid answer boundary or logits shape')
    chosen = logits[:, prompt_length-1:n-1].float().log_softmax(-1)
    target = input_ids[:, prompt_length:]
    token_scores = chosen.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    if reduction == 'sum':
        return token_scores.sum(-1)
    if reduction == 'mean':
        return token_scores.mean(-1)
    raise ValueError(reduction)
