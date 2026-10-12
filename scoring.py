"""Turn a raw generation into (answer, confidence, outcome). Pure Python so it can be unit-tested without a GPU."""
import math
import re
import string

NUM_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
ABSTAIN = ["i dont know", "i do not know", "cannot be determined", "can not be determined", "not enough information",
           "cannot answer", "unable to determine"]


def normalize(s):
    """SQuAD-style normalisation plus number words -> digits."""
    s = s.lower().replace("’", "'")
    s = s.replace("'", "")
    s = re.sub(r"(?<=\d)[.,](?=\d)", "", s)  # keep 9.21 and 820,445 as one token, so "9.21" doesn't match "nine"
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    toks = [NUM_WORDS.get(t, t) for t in s.split() if t not in {"a", "an", "the"}]
    return " ".join(toks)


def contains(hay, needle):
    return bool(needle) and f" {needle} " in f" {hay} "


def matches(pred, golds):
    """True if a gold answer appears in the prediction, or a non-trivial prediction appears inside a gold answer."""
    p = normalize(pred)
    if not p:
        return False
    for g in golds:
        g = normalize(g)
        if contains(p, g) or (len(p) >= 4 and contains(g, p)):
            return True
    return False


def is_abstain(pred):
    p = normalize(pred)
    return p in {"unknown", "not known"} or any(a in p for a in ABSTAIN)


def classify(pred, org_answer, sub_answer):
    """original = true answer, substituted = the answer planted in the misleading document."""
    if pred is None:
        return "truncated"
    o, s = matches(pred, org_answer), matches(pred, sub_answer)
    if o and s:
        return "both"
    if o:
        return "original"
    if s:
        return "substituted"
    if is_abstain(pred):
        return "abstain"
    return "other"


def planted_answer(item, condition):
    """The wrong answer that a condition's document states (NQ-Swap swap, or the plausible swap)."""
    return item["plaus_answer"] if condition == "plausible" else item["sub_answer"]


def split_answer(ids, logprobs, think_end_id, special_ids, thinking, decode):
    """Split generated token ids into reasoning and final answer.

    Returns (reasoning_ids, answer_ids, confidence_logprobs, truncated). Special tokens are dropped from the answer;
    whitespace-only tokens stay in the answer text but are left out of the confidence so they don't inflate it.
    Tokens with log-prob None were inserted by us (budget forcing), not generated, so they never count as answer.
    """
    ids, logprobs = list(ids), list(logprobs)
    if think_end_id is not None and think_end_id in ids:
        k = len(ids) - 1 - ids[::-1].index(think_end_id)
        reason, ans, lps = ids[:k], ids[k + 1:], logprobs[k + 1:]
    elif thinking:
        return ids, [], [], True
    else:
        reason, ans, lps = [], ids, logprobs
    keep = [(t, lp) for t, lp in zip(ans, lps) if t not in special_ids and lp is not None]
    return reason, [t for t, _ in keep], [lp for t, lp in keep if decode([t]).strip()], False


def confidence(logprobs):
    """Geometric-mean token probability of the final answer."""
    return math.exp(sum(logprobs) / len(logprobs)) if logprobs else None


CONFLICT_HINTS = ["contradict", "conflict", "my knowledge", "i recall", "i remember", "but i know", "incorrect",
                  "is wrong", "mistake", "inconsistent", "in reality", "actually"]


def mentions_conflict(reasoning_text):
    """Rough keyword heuristic only; confirm by reading the cases (the errors CSV has a column for this)."""
    t = reasoning_text.lower()
    return any(h in t for h in CONFLICT_HINTS)
