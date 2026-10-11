"""Build the evaluation set from NQ-Swap (Longpre et al., 2021): one item per question with four evidence conditions.

Conditions built at run time from each item:
  closed_book  - no document
  gold         - the original Wikipedia passage (contains the true answer)
  misleading   - the same passage with the answer swapped for another entity
  irrelevant   - the passage of a different question
"""
import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

from scoring import normalize, contains

DATASET = "pminervini/NQ-Swap"


def load_rows():
    try:
        from datasets import load_dataset
        return [dict(r) for r in load_dataset(DATASET, split="dev")]
    except ImportError:  # no `datasets` installed: page through the Hugging Face datasets-server API
        rows, off = [], 0
        while True:
            url = (f"https://datasets-server.huggingface.co/rows?dataset={DATASET}&config=default&split=dev"
                   f"&offset={off}&length=100")
            page = json.load(urllib.request.urlopen(url, timeout=60))
            rows += [r["row"] for r in page["rows"]]
            off += 100
            if off >= page["num_rows_total"]:
                return rows


def clean(text):
    text = re.sub(r"<[^>]+>", " ", text)  # NQ passages keep HTML tags like <P>, <Td>
    return re.sub(r"\s+", " ", text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--max-words", type=int, default=300, help="drop passages longer than this")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    rows = load_rows()
    stats = {"rows": len(rows)}
    by_q = {}
    for r in rows:
        by_q.setdefault(normalize(r["question"]), []).append(r)
    stats["unique_questions"] = len(by_q)

    items, drop = [], {"overlap": 0, "answer_not_in_passage": 0, "too_long": 0}
    for q in sorted(by_q):
        r = rng.choice(by_q[q])
        org, sub = [x for x in r["org_answer"] if x.strip()], [x for x in r["sub_answer"] if x.strip()]
        oc, sc = clean(r["org_context"]), clean(r["sub_context"])
        no, ns = [normalize(x) for x in org], [normalize(x) for x in sub]
        if not org or not sub or any(contains(x, y) or contains(y, x) for x in no for y in ns):
            drop["overlap"] += 1
            continue
        if not any(contains(normalize(oc), x) for x in no) or not any(contains(normalize(sc), y) for y in ns):
            drop["answer_not_in_passage"] += 1
            continue
        if len(oc.split()) > a.max_words or len(sc.split()) > a.max_words:
            drop["too_long"] += 1
            continue
        items.append({"question": r["question"].strip(), "org_answer": org, "sub_answer": sub,
                      "org_context": oc, "sub_context": sc})
    stats["dropped"], stats["eligible"] = drop, len(items)

    rng.shuffle(items)
    items = items[:a.n]
    for i, it in enumerate(items):
        it["id"] = f"nq{i:04d}"

    # Irrelevant passage: another item's gold passage that mentions neither answer
    for i, it in enumerate(items):
        answers = [normalize(x) for x in it["org_answer"] + it["sub_answer"]]
        while True:
            j = rng.randrange(len(items))
            cand = normalize(items[j]["org_context"])
            if j != i and not any(contains(cand, x) for x in answers):
                break
        it["irr_context"], it["irr_source"] = items[j]["org_context"], items[j]["id"]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "items.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    stats["written"] = len(items)
    stats["mean_words_gold"] = round(sum(len(x["org_context"].split()) for x in items) / max(len(items), 1), 1)
    json.dump(stats, open(out / "stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
