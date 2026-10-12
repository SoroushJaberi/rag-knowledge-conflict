"""Study v2, Step 1: all runs for one model, most important first, with one model load.

python run_plan.py --model Qwen/Qwen3-8B                       (all parts, written to results_v2/)
python run_plan.py --model Qwen/Qwen3-8B --parts base,prompts  (only some parts)
Resumable: finished questions are skipped, so after a disconnect re-run the same line.
The plan is listed in STUDY_V2.md; v1 (results/, seed 13) serves as one more replicate of the base runs.
"""
import argparse

import run_eval

SEED, SEED2 = 14, 15  # v1 used 13 (with per-chunk seeds); all v2 runs share seed 14 unless they are the replicate
ALL4 = ["closed_book", "gold", "misleading", "irrelevant"]
DOCS = ["gold", "misleading", "irrelevant"]    # closed-book answers come from the base run with the same seed
CB_DOCS = ["closed_book", "gold", "misleading"]  # thinking also changes closed-book accuracy, so budgets include it
PARTS = {
    "base": [dict(think=t, prompt="base", conditions=ALL4, seed=SEED) for t in (0, 1)],
    "prompts": [dict(think=t, prompt=p, conditions=DOCS, seed=SEED) for p in ("warn", "neutral", "own") for t in (0, 1)],
    "budgets": [dict(think=1, prompt="base", budget=b, conditions=CB_DOCS, seed=SEED) for b in (0, 128, 256, 512)],
    "seeds": [dict(think=t, prompt="base", conditions=ALL4, seed=SEED2) for t in (0, 1)],
    "plausible": [dict(think=t, prompt="base", conditions=["plausible"], seed=SEED) for t in (0, 1)],
    "wait": [dict(think=1, prompt="base", wait=1, conditions=CB_DOCS, seed=SEED)],
}
CHUNK = 1000  # one chunk per condition: the same batch size for every v2 run, so their costs are comparable


def plan(model, parts):
    out = []
    for part in parts:
        for p in PARTS[part]:
            out.append(dict(model=model, think=bool(p["think"]), prompt=p["prompt"], budget=p.get("budget"),
                            wait=p.get("wait", 0), decoding="recommended", seed=p["seed"],
                            conditions=list(p["conditions"]), max_new_tokens=0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--parts", default=",".join(PARTS))
    ap.add_argument("--data", default="data/items.jsonl")
    ap.add_argument("--plausible", default="data/plausible_answers.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="only the first N items (smoke test)")
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--out", default="results_v2")
    run_eval.engine_args(ap)
    a = ap.parse_args()
    parts = a.parts.split(",")
    items = run_eval.load_items(a.data, a.plausible)
    if "plausible" in parts and not any("plaus_context" in it for it in items):
        print(f"skipping part 'plausible': {a.plausible} not found (run plausible.py generate first)")
        parts.remove("plausible")
    items = items[:a.limit] if a.limit else items
    run_eval.run_all(a, items, plan(a.model, parts))


if __name__ == "__main__":
    main()
