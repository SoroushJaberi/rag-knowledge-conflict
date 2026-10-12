"""Plausible wrong documents: the true answer in the original passage is replaced by a believable wrong answer.

NQ-Swap swaps in an entity of the same type that is often absurd (Dakar "located in South Korea"). Here a generator
model (default: microsoft/phi-4, a different model family than the Qwen3 models under test) proposes wrong answers that
are close to the truth. The passage is rebuilt by replacing every mention of the true answer (word boundaries,
longest answer first), and automatic checks keep a candidate only if it is clearly different from the true answer and
the new passage states it and no longer contains the true answer.

python plausible.py generate              (GPU, vLLM; writes data/plausible_answers.jsonl, stored in the repository)
python plausible.py check                 (no GPU; replays NQ-Swap's own swaps with our replacement rule)
"""
import argparse
import json
import re
from pathlib import Path

from scoring import contains, normalize

GEN_PROMPT = ("Here is a quiz question and its correct answer.\n\nQuestion: {q}\nCorrect answer: {a}\n\n"
              "Write one wrong answer that a well-informed person could find believable: the same kind of thing as the "
              "correct answer (a person for a person, a year for a year, a place for a place, a number for a number), "
              "close to the truth, but definitely not correct and not another name for the correct answer. "
              "Reply with the wrong answer only, in a few words.")
MAX_WORDS = 6


def substitute(text, olds, new):
    """Replace every whole-word mention of any of `olds` (case-insensitive, longest first) with `new`."""
    for o in sorted({o.strip() for o in olds if o.strip()}, key=len, reverse=True):
        text = re.sub(r"(?<!\w)" + re.escape(o) + r"(?!\w)", lambda m: new, text, flags=re.I)
    return text


def kind(answer):
    n = normalize(answer)
    return "year" if re.fullmatch(r"1\d{3}|20\d{2}", n) else "number" if re.fullmatch(r"[\d ]+", n) else "text"


def clean_candidate(text):
    text = (text or "").strip().split("\n")[0].strip()
    text = re.sub(r"^(wrong answer|answer)\s*:\s*", "", text, flags=re.I)
    return text.strip(" \"'*.`")


def check(item, cand):
    """Return (passage, None) if `cand` can be used as the plausible wrong answer, else (None, reason)."""
    c = normalize(cand)
    trues = [normalize(x) for x in item["org_answer"]]
    if not c:
        return None, "empty"
    if len(cand.split()) > MAX_WORDS:
        return None, "too long"
    if any(contains(c, t) or contains(t, c) for t in trues if t):
        return None, "overlaps true answer"
    if kind(cand) != kind(item["org_answer"][0]):
        return None, "different type"
    if contains(normalize(item["org_context"]), c):
        return None, "already in passage"
    passage = substitute(item["org_context"], item["org_answer"], cand)
    np_ = normalize(passage)
    if any(contains(np_, t) for t in trues if t) or not contains(np_, c):
        return None, "replacement failed"
    return passage, None


def load_items(path="data/items.jsonl", plausible="data/plausible_answers.jsonl"):
    """Items with `plaus_answer` and `plaus_context` added where a checked plausible answer exists."""
    items = [json.loads(line) for line in open(path, encoding="utf-8")]
    if plausible and Path(plausible).exists():
        extra = {r["id"]: r["plaus_answer"] for r in map(json.loads, open(plausible, encoding="utf-8"))
                 if r.get("plaus_answer")}
        for it in items:
            if it["id"] in extra:
                passage, why = check(it, extra[it["id"]])
                if passage is None:
                    raise ValueError(f"{it['id']}: stored plausible answer fails the checks ({why})")
                it["plaus_answer"], it["plaus_context"] = [extra[it["id"]]], passage
    return items


def generate(a):
    from vllm import LLM, SamplingParams
    items = [json.loads(line) for line in open(a.data, encoding="utf-8")]
    llm = LLM(model=a.model, seed=a.seed, max_model_len=2048, gpu_memory_utilization=0.90)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": GEN_PROMPT.format(
        q=it["question"].strip().rstrip("?") + "?", a=it["org_answer"][0])}], tokenize=False, add_generation_prompt=True)
        for it in items]
    sp = SamplingParams(n=a.n, temperature=a.temperature, top_p=0.95, max_tokens=24, seed=a.seed)
    outs = llm.generate(prompts, sp)
    reasons = {}
    with open(a.out, "w", encoding="utf-8") as f:
        for it, o in zip(items, outs):
            cands, chosen = [clean_candidate(c.text) for c in o.outputs], None
            why = []
            for c in cands:
                passage, r = check(it, c)
                if passage is not None:
                    chosen = c
                    break
                why.append(r)
            if chosen is None:
                reasons[why[0] if why else "no output"] = reasons.get(why[0] if why else "no output", 0) + 1
            f.write(json.dumps({"id": it["id"], "plaus_answer": chosen, "candidates": cands}, ensure_ascii=False) + "\n")
    kept = sum(1 for r in map(json.loads, open(a.out, encoding="utf-8")) if r["plaus_answer"])
    meta = {"generator": a.model, "prompt": GEN_PROMPT, "samples_per_item": a.n, "temperature": a.temperature,
            "top_p": 0.95, "seed": a.seed, "items": len(items), "kept": kept, "dropped_first_reason": reasons}
    json.dump(meta, open(Path(a.out).with_name("plausible_meta.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))


def replay(a):
    items = [json.loads(line) for line in open(a.data, encoding="utf-8")]
    same = sum(substitute(it["org_context"], it["org_answer"], it["sub_answer"][0]) == it["sub_context"] for it in items)
    print(f"Our replacement rule reproduces NQ-Swap's wrong passage exactly for {same} of {len(items)} items "
          "(differences: NQ-Swap also replaces the answer inside longer words or numbers).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "check"])
    ap.add_argument("--data", default="data/items.jsonl")
    ap.add_argument("--model", default="microsoft/phi-4")
    ap.add_argument("--n", type=int, default=5, help="candidates per item; the first that passes the checks is kept")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="data/plausible_answers.jsonl")
    a = ap.parse_args()
    generate(a) if a.cmd == "generate" else replay(a)


if __name__ == "__main__":
    main()
