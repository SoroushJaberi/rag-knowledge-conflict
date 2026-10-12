"""End-to-end check without a GPU: a fake model runs through run_eval.py (incl. resume) and analyze.py (v1), and
through run_plan.py, trace_labels.py and analyze_v2.py (study v2, Step 1).

python test_pipeline.py --data <items.jsonl from prepare_data.py> --work <scratch folder>
"""
import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

import budget as bf
import run_eval

THINK_END, EOS, THINK_START = 1, 2, 3
STOP_RAW = bf.STOP_TEXT.replace("</think>", "\x01")


class FakeTok:
    all_special_ids = [EOS]

    def get_vocab(self):
        return {"</think>": THINK_END, "<eos>": EOS, "<think>": THINK_START}

    def encode(self, s, add_special_tokens=False):
        return [ord(c) for c in s.replace("</think>", "\x01").replace("<think>", "\x03")]

    def decode(self, ids, skip_special_tokens=False):
        return "".join({THINK_END: "</think>", EOS: "", THINK_START: "<think>"}.get(i, chr(i)) for i in ids)

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt, enable_thinking=False):
        return ("[THINK]" if enable_thinking else "[NOTHINK]") + msgs[0]["content"] + "[A]"


class FakeEngine:
    """Knows ~70% of answers, follows wrong documents half the time (less with the `own` prompt), and with thinking
    writes a reasoning trace that mentions the conflict, so budgets, "Wait" and the stop text can be exercised."""
    items = {}
    eos_ids = {EOS}

    def __init__(self, a):
        self.tok = FakeTok()

    def plan(self, head):
        it = next(it for it in self.items.values() if run_eval.question(it["question"]) in head)
        r = random.Random(len(head) * 7919)
        if "Document:" not in head:
            return it, (it["org_answer"][0] if r.random() < 0.7 else "I don't know"), "recall"
        for field, planted, p in (("sub_context", "sub_answer", 0.5), ("plaus_context", "plaus_answer", 0.6)):
            if it.get(field) and it[field] in head:
                p = 0.2 if "if it contradicts what you know" in head else p
                return it, (it[planted][0] if r.random() < p else it["org_answer"][0]), "hmm, this contradicts my knowledge. "
        if it["org_context"] in head:
            return it, it["org_answer"][0], "the document says so. "
        return it, ("I don't know" if r.random() < 0.6 else "Paris"), "unrelated. "

    def rest(self, head, g):
        it, ans, note = self.plan(head)
        final = "\n\n" + ans + "\x02"
        if not head.startswith("[THINK]"):
            return final
        if g.endswith(STOP_RAW):
            return ans + "\x02"
        if "\x01" in g:
            return final[len(g.rsplit("\x01", 1)[1]):]
        if g.endswith("Wait"):
            return "more thought\x01" + final
        full = "\x03" + note * 4 + "\x01" + final
        assert full.startswith(g), (head[-80:], g)
        return full[len(g):]

    def generate(self, prompts, dec, max_new, seeds, stop_ids=()):
        mx = max_new if isinstance(max_new, list) else [max_new] * len(prompts)
        out = []
        for p, m in zip(prompts, mx):
            head, g = "".join(map(chr, p)).split("[A]", 1)
            it, ans, _ = self.plan(head)
            ids = [ord(c) for c in self.rest(head, g)][:m]
            cut = next((k + 1 for k, t in enumerate(ids) if t in {EOS, *stop_ids}), None)
            ids = ids[:cut] if cut else ids
            conf = -random.Random(len(head)).random() * (1.5 if ans not in it["org_answer"] else 0.3)
            out.append((ids, [conf] * len(ids), cut is not None))
        return out


class FakeJudge:
    def yes_probs(self, texts):
        return [0.9 if "contradicts" in t.split('"""')[1] else 0.1 for t in texts]


def plausible_file(items, path):
    from plausible import check, kind
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            k = kind(it["org_answer"][0])
            cand = "Plausibleville" if k == "text" else "1888" if k == "year" else "77"
            f.write(json.dumps({"id": it["id"], "plaus_answer": cand if check(it, cand)[0] else None}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    work = Path(a.work)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    items = [json.loads(line) for line in open(a.data, encoding="utf-8")][:40]
    small = work / "items.jsonl"
    small.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    plausible_file(items, work / "plausible_answers.jsonl")
    FakeEngine.items = {i["id"]: i for i in run_eval.load_items(small, work / "plausible_answers.jsonl")}
    run_eval.VLLMEngine = FakeEngine

    # v1 command lines: unchanged run names and behaviour
    for model, think, limit in [("fake/Fake-1.7B", 0, 15), ("fake/Fake-1.7B", 0, 0), ("fake/Fake-8B", 0, 0),
                                ("fake/Fake-1.7B", 1, 0), ("fake/Fake-8B", 1, 0)]:
        sys.argv = ["run_eval.py", "--model", model, "--think", str(think), "--data", str(small),
                    "--out", str(work / "results"), "--chunk", "25"] + (["--limit", str(limit)] if limit else [])
        run_eval.main()
    preds = [json.loads(line) for line in open(work / "results/Fake-1.7B_nothink/predictions.jsonl", encoding="utf-8")]
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

    # Study v2, Step 1: the full plan for two models (the second call resumes and must not add anything)
    import run_plan
    v2 = work / "results_v2"
    for model in ("fake/Fake-8B", "fake/Fake-14B", "fake/Fake-14B"):
        sys.argv = ["run_plan.py", "--model", model, "--data", str(small), "--plausible",
                    str(work / "plausible_answers.jsonl"), "--out", str(v2), "--chunk", "15"]
        run_plan.main()
    names = sorted(p.name for p in v2.iterdir())
    assert "Fake-14B_think_b0_s14" in names and "Fake-14B_think_wait1_s14" in names and "Fake-8B_nothink_own_s14" in names
    b0 = [json.loads(line) for line in open(v2 / "Fake-8B_think_b0_s14/predictions.jsonl", encoding="utf-8")]
    assert all(p["n_thinking"] == 0 and p["forced_stop"] and p["answer"] for p in b0)
    b128 = [json.loads(line) for line in open(v2 / "Fake-8B_think_b128_s14/predictions.jsonl", encoding="utf-8")]
    assert all(p["n_thinking"] <= 128 for p in b128) and any(p["forced_stop"] for p in b128)
    n_pl = sum("plaus_context" in it for it in FakeEngine.items.values())
    pl = [json.loads(line) for line in open(v2 / "Fake-8B_think_s14/predictions.jsonl", encoding="utf-8")]
    assert len({(p["id"], p["condition"]) for p in pl}) == len(pl) == 40 * 4 + n_pl > 40 * 4
    cfg = json.load(open(v2 / "Fake-8B_think_s14/config.json", encoding="utf-8"))
    assert cfg["conditions"] == ["closed_book", "gold", "misleading", "irrelevant", "plausible"] and len(cfg["calls"]) == 2
    usage = [json.loads(line) for line in open(v2 / "Fake-8B_think_b128_s14/usage.jsonl", encoding="utf-8")]
    assert {u["condition"] for u in usage} == {"closed_book", "gold", "misleading"}
    print("run_plan: OK;", len(names), "runs")

    import trace_labels
    trace_labels.make_judge = lambda args: FakeJudge()
    common = ["--data", str(small), "--plausible", str(work / "plausible_answers.jsonl"), "--labels", str(work / "labels")]
    sys.argv = ["trace_labels.py", "sheet", "--v1-results", str(work / "results"), "--sheet-runs", "Fake-8B_think"] + common
    trace_labels.main()
    sys.argv = ["trace_labels.py", "judge", "--runs", f"{work / 'results'},{v2}", "--max-per-condition", "5"] + common
    trace_labels.main()
    sheet, key = trace_labels.label_files(work / "labels")
    rows = list(csv.DictReader(open(sheet, encoding="utf-8-sig")))
    for r in rows:  # a fake labeller who agrees with the fake judge
        r.update({q: "yes" if "contradicts" in r["reasoning"] else "no" for q in trace_labels.QUESTIONS})
    with open(sheet, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("trace_labels: OK;", len(rows), "rows in the sheet")

    import analyze_v2
    sys.argv = ["analyze_v2.py", "--results", f"{work / 'results'},{v2}", "--data", str(small), "--plausible",
                str(work / "plausible_answers.jsonl"), "--labels", str(work / "labels"), "--out", str(v2)]
    analyze_v2.main()
    s = json.load(open(v2 / "summary_step1.json"))
    assert {"Fake-8B_think_own_s14", "Fake-14B_nothink_neutral_s14"} <= set(s["prompts"])
    assert s["prompts"]["Fake-8B_think_own_s14"]["vs_base"]["n"] > 0
    assert {"Fake-8B_think_b0_s14", "Fake-8B_think_s14", "Fake-8B_think_wait1_s14"} <= set(s["budgets"])
    assert s["budgets"]["Fake-8B_think_b0_s14"]["thinking_tokens"] == 0
    assert "Fake-8B_think_spread" in s["seeds"] and s["plausible"] and s["labels"]
    assert all(v["kappa"] == 1.0 for v in s["agreement"].values()), s["agreement"]
    print("analyze_v2: OK")


if __name__ == "__main__":
    main()
