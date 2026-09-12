"""Greedy/sampling token-loop helper with stop-token handling."""
import torch


def should_stop(new_ids, stop_ids=(), eos_id=None):
    """True when the tail of new_ids hits eos or any stop sequence."""
    if eos_id is not None and len(new_ids) and new_ids[-1] == eos_id:
        return True
    for stop in stop_ids or ():
        stop = list(stop)
        if stop and len(new_ids) >= len(stop) and list(new_ids[-len(stop):]) == stop:
            return True
    return False


def sample_next_token(logits, temperature=0.0, top_k=0, top_p=1.0):
    """Sample one token id from a 1-D logits tensor.

    temperature <= 0 selects the argmax (greedy). Otherwise applies
    top-k / nucleus filtering then multinomial sampling.
    """
    if not torch.is_tensor(logits):
        logits = torch.as_tensor(logits, dtype=torch.float32)
    logits = logits.float().flatten()
    if logits.numel() == 0:
        raise ValueError("sample_next_token got empty logits")
    if temperature is None or temperature <= 0:
        return int(torch.argmax(logits).item())
    temp = float(temperature)
    probs = torch.softmax(logits / temp, dim=-1)
    vocab = probs.numel()
    if top_k and 0 < int(top_k) < vocab:
        k = int(top_k)
        thresh = torch.topk(probs, k).values[-1]
        probs = torch.where(probs >= thresh, probs, torch.zeros_like(probs))
        probs = probs / probs.sum()
    if top_p is not None and 0.0 < float(top_p) < 1.0:
        order = torch.argsort(probs, descending=True)
        ranked = probs[order]
        cum = torch.cumsum(ranked, dim=0)
        keep = cum <= float(top_p)
        keep[0] = True  # always keep the top token
        mask = torch.zeros_like(probs)
        mask[order[keep]] = probs[order[keep]]
        probs = mask / mask.sum()
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def run_token_loop(step_fn, prompt_ids, max_new_tokens,
                   eos_id=None, stop_ids=(), temperature=0.0,
                   top_k=0, top_p=1.0):
    """Drive a token loop: repeatedly call step_fn(ids) -> logits.

    step_fn receives the full id list so far and returns 1-D logits for
    the next token. Returns the list of generated ids (prompt excluded).
    """
    if max_new_tokens <= 0:
        return []
    ids = list(prompt_ids)
    out = []
    for _ in range(int(max_new_tokens)):
        logits = step_fn(ids)
        nxt = sample_next_token(logits, temperature, top_k, top_p)
        ids.append(nxt)
        out.append(nxt)
        if should_stop(out, stop_ids, eos_id):
            break
    return out


def strip_stop_tail(ids, stop_ids=(), eos_id=None):
    """Remove a trailing eos/stop sequence from a generated id list."""
    ids = list(ids)
    if eos_id is not None and ids and ids[-1] == eos_id:
        ids = ids[:-1]
    for stop in stop_ids or ():
        stop = list(stop)
        if stop and len(ids) >= len(stop) and ids[-len(stop):] == stop:
            ids = ids[:-len(stop)]
            break
    return ids
