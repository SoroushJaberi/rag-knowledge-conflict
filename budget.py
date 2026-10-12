"""Thinking budget as a controlled variable ("budget forcing", Muennighoff et al. 2025, arXiv:2501.19393).

Cap:    the model thinks for at most `budget` tokens. If it has not closed its reasoning by then, we insert the
        stop-thinking instruction from the Qwen3 Technical Report (arXiv:2505.09388, Sec. 4.3, "Thinking Budget") and
        let the model write the final answer in a second pass.
Extend: `wait` > 0 replaces the model's first `wait` attempts to close its reasoning with "Wait", as in s1.

Every sequence is returned as (ids, logprobs) where tokens we inserted have log-prob None, so they are neither
counted as generated tokens nor used as answer text or confidence (see scoring.split_answer).
Pure Python: the engine is passed in, so this file is tested with a fake model (test_v2.py).
"""
import zlib

# Verbatim from the Qwen3 Technical Report, Sec. 4.3 (including the "." after </think>).
STOP_TEXT = ("Considering the limited time by the user, I have to give the solution based on the thinking directly "
             "now.\n</think>.\n\n")
WAIT_TEXT = "Wait"
THINK_MAX = 6144   # "unlimited" = the v1 limit; the few traces that reach it (0-3% in v1) also get the stop text
ANSWER_MAX = 64    # tokens for the final answer (as in v1 without thinking)


def request_seed(seed, item_id, condition, step=0):
    """Per-request sampling seed: the same for an item in every run with this seed, independent of batching/resume."""
    return zlib.crc32(f"{seed}|{item_id}|{condition}|{step}".encode()) & 0x7FFFFFFF


def n_generated(ids, lps, think_end=None):
    """Tokens the model generated (not inserted by us); only those before the last closing tag if think_end is given."""
    if think_end is not None and think_end in ids:
        lps = lps[:len(ids) - 1 - ids[::-1].index(think_end)]
    return sum(lp is not None for lp in lps)


def generate(eng, prompts, keys, dec, seed, think, think_end, encode, budget=None, wait=0,
             think_max=THINK_MAX, answer_max=ANSWER_MAX):
    """prompts: token-id lists ending with the assistant header; keys: (item_id, condition) per prompt.

    Returns one (ids, logprobs, forced) per prompt: everything after the prompt, inserted tokens with log-prob None;
    forced = the stop-thinking text was inserted.
    eng.generate(prompts, dec, max_new, seeds, stop_ids) returns (ids, logprobs, finished) per prompt; max_new is an
    int or one int per prompt; finished means the model stopped by itself (end of text, or a stop id kept as last id).
    """
    seeds = lambda step, idx: [request_seed(seed, *keys[k], step) for k in idx]
    if not think:
        outs = eng.generate(prompts, dec, answer_max, seeds(0, range(len(prompts))))
        return [(ids, lps, False) for ids, lps, _ in outs]

    cap = think_max if budget is None else budget
    todo = {}  # prompt index -> "answer" (continue as is) or "force" (insert the stop text first)
    if cap == 0:  # open the reasoning ourselves, unless the chat template already did
        start = encode("<think>\n")
        seq = [([], []) if p[-len(start):] == start else (list(start), [None] * len(start)) for p in prompts]
        todo = {k: "force" for k in range(len(prompts))}
    else:
        seq = []
        outs = eng.generate(prompts, dec, cap, seeds(0, range(len(prompts))), [think_end] if wait else [])
        for k, (ids, lps, fin) in enumerate(outs):
            seq.append((list(ids), list(lps)))
            if think_end not in ids:
                todo[k] = "force"
            elif wait or not fin:
                todo[k] = "answer"  # reasoning closed, answer still to come (or cut by the cap); else complete
        w = encode(WAIT_TEXT)
        for step in range(1, wait + 1):  # s1-style extension: replace the closing tag with "Wait", think on
            idx, left = [], []
            for k in [k for k, how in todo.items() if how == "answer" and seq[k][0][-1:] == [think_end]]:
                ids, lps = seq[k]
                seq[k] = (ids[:-1] + w, lps[:-1] + [None] * len(w))
                rest = cap - n_generated(*seq[k])
                if rest > 0:
                    idx.append(k)
                    left.append(rest)
                else:
                    todo[k] = "force"
            if not idx:
                break
            outs = eng.generate([prompts[k] + seq[k][0] for k in idx], dec, left, seeds(step, idx), [think_end])
            for k, (ids, lps, _) in zip(idx, outs):
                seq[k] = (seq[k][0] + list(ids), seq[k][1] + list(lps))
                todo[k] = "answer" if ids[-1:] == [think_end] else "force"

    eos = set(getattr(eng, "eos_ids", ()))
    for k in [k for k, how in todo.items() if how == "force"]:
        ids, lps = seq[k]
        while ids and ids[-1] in eos and lps[-1] is not None:  # model ended without closing its reasoning
            ids, lps = ids[:-1], lps[:-1]
        s = encode(("" if cap == 0 else "\n\n") + STOP_TEXT)
        seq[k] = (ids + s, lps + [None] * len(s))
    idx = sorted(todo)
    if idx:
        outs = eng.generate([prompts[k] + seq[k][0] for k in idx], dec, answer_max, seeds(wait + 1, idx))
        for k, (ids, lps, _) in zip(idx, outs):
            seq[k] = (seq[k][0] + list(ids), seq[k][1] + list(lps))
    return [(ids, lps, todo.get(k) == "force") for k, (ids, lps) in enumerate(seq)]
