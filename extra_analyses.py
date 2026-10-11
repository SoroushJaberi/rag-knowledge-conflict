"""Analyses behind the README that summary.md does not cover. Writes results/extra_analyses.md and
results/spotcheck_traces.md (a random sample of reasoning traces for checking by hand).

python extra_analyses.py   (needs data/items.jsonl from prepare_data.py and the runs in results/)
"""
import json
import random
from math import comb
from pathlib import Path
from statistics import median

from analyze import read_predictions
from metrics import auroc
from scoring import classify, matches, mentions_conflict, normalize

SIZES = ["1.7B", "4B", "8B", "14B"]
MODES = ["nothink", "think"]
CONFLICT_WORDS = ["however", "but ", "according to the document", "error", "incorrect", "actually"]

items = {it["id"]: it for it in map(json.loads, open("data/items.jsonl", encoding="utf-8"))}
IDS = list(items)


def load(name):
    by = {}
    for p in read_predictions(Path("results") / name):
        it = items[p["id"]]
        p["outcome"] = classify(p["answer"], it["org_answer"], it["sub_answer"])
        by.setdefault(p["condition"], {})[p["id"]] = p
    return by


def sign_test(a, b):
    """Exact two-sided sign test on discordant pairs."""
    n, k = a + b, max(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n) if n else float("nan")


def disagree(a, b):
    """Closed-book answer and with-document answer name different things ("I don't know" never counts)."""
    if not a or not b or "know" in a.lower() or "know" in b.lower():
        return False
    return not (matches(a, [b]) or matches(b, [a]) or normalize(a) == normalize(b))


runs = {(s, m): load(f"Qwen3-{s}_{m}") for s in SIZES for m in MODES}
known = {k: [i for i in IDS if r["closed_book"][i]["outcome"] == "original"] for k, r in runs.items()}
over = lambda r, ids: sum(r["misleading"][i]["outcome"] == "substituted" for i in ids) / len(ids)
L = ["# Extra analyses", "", "All numbers re-scored from the stored answers with `scoring.py`.", ""]

# 1) The same facts for every model: those the 14B answers correctly without a document (thinking off)
ref = known[("14B", "nothink")]
L += [f"## 1. Overwrite rate on one fixed set of facts (n = {len(ref)}, known by Qwen3-14B, thinking off)", "",
      "| Model | Thinking off | Thinking on |", "|---|---|---|"]
L += [f"| {s} | {over(runs[(s, 'nothink')], ref):.1%} | {over(runs[(s, 'think')], ref):.1%} |" for s in SIZES]

# 2) Thinking on vs off on the facts each size knows in both modes
L += ["", "## 2. Thinking on vs off, same facts (known in both modes)", "",
      "| Model | n | Overwrite, off | Overwrite, on | Overwritten only with thinking on | only with thinking off | "
      "Exact sign test p |", "|---|---|---|---|---|---|---|"]
for s in SIZES:
    off, on = runs[(s, "nothink")], runs[(s, "think")]
    both = [i for i in known[(s, "nothink")] if i in set(known[(s, "think")])]
    a = sum(on["misleading"][i]["outcome"] == "substituted" and off["misleading"][i]["outcome"] != "substituted" for i in both)
    b = sum(off["misleading"][i]["outcome"] == "substituted" and on["misleading"][i]["outcome"] != "substituted" for i in both)
    L.append(f"| {s} | {len(both)} | {over(off, both):.1%} | {over(on, both):.1%} | {a} | {b} | {sign_test(a, b):.4f} |")

# 3) Does the reasoning mention a conflict? (keyword heuristic, scoring.mentions_conflict)
L += ["", "## 3. Reasoning that mentions a conflict (keyword heuristic), known facts, thinking on", "",
      "| Model | Overwritten answers whose reasoning mentions a conflict | Wrong doc: mentions | Correct doc: mentions "
      "(false alarm) | AUROC as a wrong-doc detector | Median generated tokens per answer, with a document |",
      "|---|---|---|---|---|---|"]
for s in SIZES:
    r, ids = runs[(s, "think")], known[(s, "think")]
    ow = [i for i in ids if r["misleading"][i]["outcome"] == "substituted"]
    mis = [mentions_conflict(r["misleading"][i]["reasoning"]) for i in ids]
    gold = [mentions_conflict(r["gold"][i]["reasoning"]) for i in ids]
    auc = auroc([float(x) for x in gold + mis], [0] * len(ids) + [1] * len(ids))
    med = median(p["n_generated"] for c in ("gold", "misleading") for p in r[c].values())
    L.append(f"| {s} | {sum(mentions_conflict(r['misleading'][i]['reasoning']) for i in ow) / len(ow):.0%} | "
             f"{sum(mis) / len(ids):.0%} | {sum(gold) / len(ids):.0%} | {auc:.2f} | {med:.0f} |")
hedge = {s: sum(any(w in (p["answer"] or "").lower() for w in CONFLICT_WORDS)
                for p in runs[(s, "think")]["misleading"].values()) for s in SIZES}
L += ["", "Final answers under a wrong document that contain a hedge word (" + ", ".join(f'"{w.strip()}"' for w in
      CONFLICT_WORDS) + "): " + ", ".join(f"{s} {n}/1000" for s, n in hedge.items()) + "."]

# 4) Cheap black-box detectors: how well do they flag a wrong document?
L += ["", "## 4. Cheap detectors: correct doc (0) vs wrong doc (1), AUROC", "",
      "*Ask twice*: flag when the answer without a document disagrees with the answer given the document "
      "(about 6 extra generated tokens). *Answer confidence*: low confidence = flag.", "",
      "| Run | Ask twice, known facts | Ask twice, all 1,000 questions | Answer confidence, known facts | "
      "Answer confidence, all questions |", "|---|---|---|---|---|"]
for (s, m), r in runs.items():
    cells = []
    for kind in ("ask", "conf"):
        for ids in (known[(s, m)], IDS):
            y = [0] * len(ids) + [1] * len(ids)
            if kind == "ask":
                x = [float(disagree(r["closed_book"][i]["answer"], r[c][i]["answer"])) for c in ("gold", "misleading") for i in ids]
            else:
                x = [-(r[c][i]["confidence"] or 1) for c in ("gold", "misleading") for i in ids]
            cells.append(f"{auroc(x, y):.2f}")
    L.append(f"| {s} {m} | {' | '.join(cells)} |")

Path("results/extra_analyses.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))

# 5) Random sample of overwritten known facts with full reasoning, for checking by hand
random.seed(1)
S = ["# Spot check: reasoning traces of overwritten known facts", "",
     "Random sample (seed 1): 6 cases each from Qwen3-14B and Qwen3-8B with thinking on, where the model answered "
     "correctly without a document but gave the planted answer with the wrong document. Labels are in "
     "`spotcheck_labels.csv`.", ""]
for s in ("14B", "8B"):
    r = runs[(s, "think")]
    pool = [i for i in known[(s, "think")] if r["misleading"][i]["outcome"] == "substituted"]
    for i in random.sample(pool, 6):
        it, p = items[i], r["misleading"][i]
        S += [f"## Qwen3-{s} · {i}", f"- **Question:** {it['question']}", f"- **True answer:** {it['org_answer'][0]}",
              f"- **Planted answer:** {it['sub_answer'][0]}", f"- **Answer without a document:** {r['closed_book'][i]['answer']}",
              f"- **Answer with the wrong document:** {p['answer']} (confidence {p['confidence']:.3f})", "",
              "> " + " ".join(p["reasoning"].replace("<think>", "").split()), ""]
Path("results/spotcheck_traces.md").write_text("\n".join(S) + "\n", encoding="utf-8")
print("\nwrote results/extra_analyses.md and results/spotcheck_traces.md")
