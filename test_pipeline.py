"""End-to-end check without a GPU: a fake model runs through run_eval.py (incl. resume) and analyze.py.

python test_pipeline.py --data <items.jsonl from prepare_data.py> --work <scratch folder>
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import run_eval

THINK_END, EOS = 1, 2


class FakeTok:
    all_special_ids = [THINK_END, EOS]

    def get_vocab(self):
        return {"</think>": THINK_END, "<eos>": EOS}

    def decode(self, ids, skip_special_tokens=False):
        return "".join("" if i in (THINK_END, EOS) else chr(i) for i in ids)

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt, enable_thinking):
        return ("[THINK]" if enable_thinking else "") + msgs[0]["content"]


class FakeEngine:
    """Behaves like a model that knows ~70% of answers, follows misleading documents half the time."""
    items = {}

    def __init__(self, a):
        self.tok = FakeTok()

    def generate(self, prompts, dec, max_new, seed):
        out = []
        for p in prompts:
            think = p.startswith("[THINK]")
            it = next(it for it in self.items.values() if run_eval.question(it["question"]) in p)
            r = random.Random(len(p) * 7919 + seed)
            if "Document:" not in p:
                ans = it["org_answer"][0] if r.random() < 0.7 else "I don't know"
            elif it["sub_context"] in p:
                ans = it["sub_answer"][0] if r.random() < 0.5 else it["org_answer"][0]
            elif it["org_context"] in p:
                ans = it["org_answer"][0]
            else:
                ans = "I don't know" if r.random() < 0.6 else "Paris"
            conf = -r.random() * (1.5 if ans == it["sub_answer"][0] else 0.3)  # less sure when following a wrong doc
            body = [ord(c) for c in " " + ans] + [EOS]
            pre = [ord(c) for c in "hmm, this contradicts my knowledge"] + [THINK_END] if think else []
            ids = pre + body
            out.append((ids, [conf] * len(ids)))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    work = Path(a.work)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    items = [json.loads(l) for l in open(a.data, encoding="utf-8")][:40]
    small = work / "items.jsonl"
    small.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    FakeEngine.items = {i["id"]: i for i in items}
    run_eval.VLLMEngine = FakeEngine

    for model, think, limit in [("fake/Fake-1.7B", 0, 15), ("fake/Fake-1.7B", 0, 0), ("fake/Fake-8B", 0, 0),
                                ("fake/Fake-1.7B", 1, 0), ("fake/Fake-8B", 1, 0)]:
        sys.argv = ["run_eval.py", "--model", model, "--think", str(think), "--data", str(small),
                    "--out", str(work / "results"), "--chunk", "25"] + (["--limit", str(limit)] if limit else [])
        run_eval.main()

    preds = [json.loads(l) for l in open(work / "results/Fake-1.7B_nothink/predictions.jsonl", encoding="utf-8")]
    keys = [(p["id"], p["condition"]) for p in preds]
    assert len(keys) == len(set(keys)) == 40 * 4, "resume must not duplicate or skip items"
    print("run_eval: resume OK,", len(preds), "predictions")

    import analyze
    sys.argv = ["analyze.py", "--results", str(work / "results"), "--data", str(small), "--n-errors", "10"]
    analyze.main()
    s = json.load(open(work / "results/summary.json"))
    assert set(s) == {"Fake-1.7B_nothink", "Fake-8B_nothink", "Fake-1.7B_think", "Fake-8B_think"}
    assert all(0 <= v["overwrite_known"][0] <= 1 for v in s.values())
    assert s["Fake-8B_think"]["overwritten_reasoning_mentions_conflict"] == 1.0
    print("analyze: OK; files:", sorted(p.name for p in (work / "results").iterdir()))


if __name__ == "__main__":
    main()
