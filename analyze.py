"""Summarise all runs: outcome shares, overwrite rate, whether confidence flags wrong evidence, figures, error lists."""
import argparse
import csv
import gzip
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from metrics import auroc, bootstrap_ci, ece
from scoring import classify, mentions_conflict

CONDS = ["closed_book", "gold", "misleading", "irrelevant"]
OUTCOMES = ["original", "substituted", "both", "abstain", "other", "truncated"]


def size_of(model):
    m = re.search(r"(\d+(?:\.\d+)?)B", model)
    return float(m.group(1)) if m else float("nan")


def read_predictions(run_dir):
    """predictions.jsonl, or the compressed predictions.jsonl.gz stored in the repository."""
    path = Path(run_dir) / "predictions.jsonl"
    f = open(path, encoding="utf-8") if path.exists() else gzip.open(str(path) + ".gz", "rt", encoding="utf-8")
    with f:
        return [json.loads(line) for line in f]


def load_runs(root, items):
    runs = []
    for cfg_path in sorted(Path(root).glob("*/config.json")):
        cfg = json.load(open(cfg_path, encoding="utf-8"))
        preds = read_predictions(cfg_path.parent)
        by = {}
        for p in preds:
            it = items[p["id"]]  # re-score from the stored answer text, so scoring fixes apply without new GPU runs
            p["outcome"] = classify(p["answer"], it["org_answer"], it["sub_answer"])
            by.setdefault(p["condition"], {})[p["id"]] = p
        u_path = cfg_path.parent / "usage.jsonl"
        usage = [json.loads(l) for l in open(u_path, encoding="utf-8")] if u_path.exists() else []
        runs.append({"name": cfg_path.parent.name, "cfg": cfg, "size": size_of(cfg["model"]), "by": by, "usage": usage})
    return sorted(runs, key=lambda r: (r["cfg"]["think"], r["size"], r["name"]))


def fmt(x, ci=None, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    s = f"{100 * x:.1f}%" if pct else f"{x:.3f}"
    if ci and not np.isnan(ci[0]):
        s += f" [{100 * ci[0]:.0f}–{100 * ci[1]:.0f}]" if pct else f" [{ci[0]:.2f}–{ci[1]:.2f}]"
    return s


def summarise(run):
    by, out = run["by"], {"per_condition": {}}
    for c in CONDS:
        ps = list(by.get(c, {}).values())
        if not ps:
            continue
        cnt = Counter(p["outcome"] for p in ps)
        ans = [p for p in ps if p["outcome"] not in ("abstain", "truncated") and p["confidence"] is not None]
        conf = [p["confidence"] for p in ans]
        corr = [p["outcome"] in ("original", "both") for p in ans]
        out["per_condition"][c] = {
            "n": len(ps), **{o: cnt[o] / len(ps) for o in OUTCOMES},
            "mean_conf": float(np.mean(conf)) if conf else float("nan"),
            "auroc_correct": auroc(conf, corr), "ece": ece(conf, corr),
            "mean_tokens": float(np.mean([p["n_generated"] for p in ps])),
        }
    cb, gold, mis, irr = (by.get(c, {}) for c in CONDS)
    known = [i for i, p in cb.items() if p["outcome"] == "original" and i in mis]
    unknown = [i for i, p in cb.items() if p["outcome"] != "original" and i in mis]
    out["n_known"] = len(known)
    if known:
        ow = np.array([mis[i]["outcome"] == "substituted" for i in known], float)
        out["overwrite_known"] = (float(ow.mean()), bootstrap_ci(np.mean, ow))
        out["resist_known"] = float(np.mean([mis[i]["outcome"] in ("original", "both") for i in known]))
        out["abstain_known"] = float(np.mean([mis[i]["outcome"] == "abstain" for i in known]))
        # stricter: "I don't know" only counts as noticing the conflict if the model answered with the correct doc
        out["abstain_known_paired"] = float(np.mean([mis[i]["outcome"] == "abstain" and i in gold
                                                     and gold[i]["outcome"] != "abstain" for i in known]))
    if unknown:
        out["overwrite_unknown"] = float(np.mean([mis[i]["outcome"] == "substituted" for i in unknown]))
    # Does confidence drop when the document contradicts what the model knows?
    pairs = [(gold[i]["confidence"], mis[i]["confidence"]) for i in known if i in gold
             and gold[i]["outcome"] == "original" and mis[i]["outcome"] == "substituted"
             and gold[i]["confidence"] is not None and mis[i]["confidence"] is not None]
    out["n_pairs"] = len(pairs)
    if len(pairs) >= 10:
        g, m = np.array(pairs).T
        f = lambda g_, m_: auroc(np.concatenate([g_, m_]), np.r_[np.ones(len(g_)), np.zeros(len(m_))])
        out["conflict_auroc"] = (f(g, m), bootstrap_ci(f, g, m))
        out["conf_gap"] = float(np.mean(g - m))
    if irr:
        ps = list(irr.values())
        out["irrelevant"] = {k: float(np.mean([p["outcome"] in v for p in ps])) for k, v in
                             {"abstain": ("abstain",), "memory_correct": ("original", "both"),
                              "made_up": ("other", "substituted")}.items()}
    u = run.get("usage", [])
    n = sum(x["n_prompts"] for x in u)
    if n:
        energy = [x["energy_j"] for x in u]
        out["cost"] = {"tokens_per_answer": sum(x["generated_tokens"] for x in u) / n,
                       "gpu_seconds_per_1000": 1000 * sum(x["seconds"] for x in u) / n,
                       "wh_per_1000": None if None in energy else 1000 * sum(energy) / 3600 / n}
    if run["cfg"]["think"]:
        over = [mis[i] for i in known if mis[i]["outcome"] == "substituted"]
        if over:
            out["overwritten_reasoning_mentions_conflict"] = float(np.mean([mentions_conflict(p["reasoning"]) for p in over]))
    return out


def write_errors(run, items, root, n):
    by, cb = run["by"], run["by"].get("closed_book", {})
    mis = by.get("misleading", {})
    rows = [p for i, p in mis.items() if cb.get(i, {}).get("outcome") == "original" and p["outcome"] == "substituted"]
    rows.sort(key=lambda p: -(p["confidence"] or 0))
    with open(Path(root) / f"errors_{run['name']}.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "true answer", "planted answer", "closed-book answer", "answer with misleading doc",
                     "confidence", "reasoning (first 3000 chars)", "heuristic: mentions conflict",
                     "YOU: noticed conflict? (yes/no/partly)", "YOU: why it followed the document", "YOU: notes"])
        for p in rows[:n]:
            it = items[p["id"]]
            w.writerow([p["id"], it["question"], " | ".join(it["org_answer"]), " | ".join(it["sub_answer"]),
                        cb[p["id"]]["answer"], p["answer"], round(p["confidence"] or 0, 3), p["reasoning"][:3000],
                        "yes" if p["reasoning"] and mentions_conflict(p["reasoning"]) else "", "", "", ""])
    return len(rows[:n])


def figures(runs, sums, root):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping figures")
        return
    fig_dir = Path(root) / "figures"
    fig_dir.mkdir(exist_ok=True)
    for key, ylabel, fname in [("overwrite_known", "Overwrite rate on known facts", "overwrite_vs_size.png"),
                               ("conflict_auroc", "Conflict-detection AUROC (0.5 = chance)", "conflict_auroc_vs_size.png")]:
        plt.figure(figsize=(5, 3.6))
        for think, label in [(False, "thinking off"), (True, "thinking on")]:
            pts = sorted((r["size"], sums[r["name"]][key]) for r in runs
                         if r["cfg"]["think"] == think and r["cfg"]["prompt"] == "base" and key in sums[r["name"]])
            if not pts:
                continue
            x = [p[0] for p in pts]; y = [p[1][0] for p in pts]
            lo = [p[1][0] - p[1][1][0] for p in pts]; hi = [p[1][1][1] - p[1][0] for p in pts]
            plt.errorbar(x, y, yerr=[lo, hi], marker="o", capsize=3, label=label)
        if key == "conflict_auroc":
            plt.axhline(0.5, color="grey", lw=0.8, ls="--")
        sizes = sorted({r["size"] for r in runs})
        plt.xscale("log"); plt.minorticks_off(); plt.xticks(sizes, [f"{s:g}B" for s in sizes])
        plt.xlabel("Model size (parameters)"); plt.ylabel(ylabel)
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(fig_dir / fname, dpi=200); plt.close()

    fig, axes = plt.subplots(1, len(runs), figsize=(2.6 * len(runs), 3.4), sharey=True, squeeze=False)
    colors = {"original": "#2e7d32", "both": "#9ccc65", "substituted": "#c62828", "abstain": "#90a4ae",
              "other": "#ffb300", "truncated": "#5d4037"}
    for ax, r in zip(axes[0], runs):
        pc = sums[r["name"]]["per_condition"]
        conds = [c for c in CONDS if c in pc]
        bottom = np.zeros(len(conds))
        for o in OUTCOMES:
            v = np.array([pc[c][o] for c in conds])
            ax.bar(range(len(conds)), v, bottom=bottom, color=colors[o], label=o)
            bottom += v
        short = {"closed_book": "no\ndoc", "gold": "correct\ndoc", "misleading": "wrong\ndoc", "irrelevant": "unrelated\ndoc"}
        ax.set_xticks(range(len(conds)), [short[c] for c in conds], fontsize=7)
        ax.set_title(r["name"].replace("_", "\n"), fontsize=8)
    axes[0][0].set_ylabel("Share of answers")
    axes[0][-1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout(); plt.savefig(fig_dir / "outcomes_by_condition.png", dpi=200); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default="data/items.jsonl")
    ap.add_argument("--n-errors", type=int, default=50)
    a = ap.parse_args()
    items = {it["id"]: it for it in map(json.loads, open(a.data, encoding="utf-8"))}
    runs = load_runs(a.results, items)
    base = {(r["cfg"]["model"], r["cfg"]["think"]): r for r in runs if r["cfg"]["prompt"] == "base"}
    for r in runs:  # runs with document conditions only (e.g. --prompt warn) reuse the base run's closed-book answers
        b = base.get((r["cfg"]["model"], r["cfg"]["think"]))
        if "closed_book" not in r["by"] and b is not None:
            r["by"]["closed_book"] = b["by"]["closed_book"]
    sums = {r["name"]: summarise(r) for r in runs}

    L = ["## Headline: when the document contradicts a fact the model knows", "",
         "| Run | Size | Thinking | Closed-book acc. | Known facts | Overwritten by wrong doc [95% CI] | Kept true answer | "
         "Said \"I don't know\" (of which: answered with correct doc) | Confidence flags wrong doc (AUROC) [95% CI] |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        s = sums[r["name"]]
        cb = s["per_condition"].get("closed_book", {})
        ow, ca = s.get("overwrite_known"), s.get("conflict_auroc")
        L.append(f"| {r['name']} | {r['size']:g}B | {'on' if r['cfg']['think'] else 'off'} | "
                 f"{fmt(cb.get('original', float('nan')) + cb.get('both', 0), pct=True)} | {s['n_known']} | "
                 f"{fmt(ow[0], ow[1], pct=True) if ow else '–'} | {fmt(s.get('resist_known'), pct=True)} | "
                 f"{fmt(s.get('abstain_known'), pct=True)} ({fmt(s.get('abstain_known_paired'), pct=True)}) | "
                 f"{fmt(ca[0], ca[1]) if ca else '–'} |")
    L += ["", "## Irrelevant document (correct behaviour: say \"I don't know\" or answer from memory)", "",
          "| Run | I don't know | Correct from memory | Made-up answer |", "|---|---|---|---|"]
    for r in runs:
        ir = sums[r["name"]].get("irrelevant", {})
        L.append(f"| {r['name']} | {fmt(ir.get('abstain'), pct=True)} | {fmt(ir.get('memory_correct'), pct=True)} | "
                 f"{fmt(ir.get('made_up'), pct=True)} |")
    L += ["", "## Per condition", "",
          "| Run | Condition | True answer | Planted answer | I don't know | Other | Truncated | Mean conf. | "
          "AUROC (conf. vs correct) | ECE | Avg. tokens |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        for c, v in sums[r["name"]]["per_condition"].items():
            L.append(f"| {r['name']} | {c} | {fmt(v['original'] + v['both'], pct=True)} | {fmt(v['substituted'], pct=True)} | "
                     f"{fmt(v['abstain'], pct=True)} | {fmt(v['other'], pct=True)} | {fmt(v['truncated'], pct=True)} | "
                     f"{fmt(v['mean_conf'])} | {fmt(v['auroc_correct'])} | {fmt(v['ece'])} | {v['mean_tokens']:.0f} |")
    L += ["", "## Cost of answering (generation only, all 4 conditions)", "",
          "| Run | Thinking | Tokens per answer | GPU seconds per 1,000 answers | Energy per 1,000 answers (Wh) |",
          "|---|---|---|---|---|"]
    for r in runs:
        c = sums[r["name"]].get("cost")
        if c:
            wh = "–" if c["wh_per_1000"] is None else f"{c['wh_per_1000']:.1f}"
            L.append(f"| {r['name']} | {'on' if r['cfg']['think'] else 'off'} | {c['tokens_per_answer']:.0f} | "
                     f"{c['gpu_seconds_per_1000']:.0f} | {wh} |")
    thinking = [(r["name"], sums[r["name"]]["overwritten_reasoning_mentions_conflict"]) for r in runs
                if "overwritten_reasoning_mentions_conflict" in sums[r["name"]]]
    if thinking:
        L += ["", "## Thinking runs: overwritten answers whose reasoning mentions a conflict (keyword heuristic, "
              "check by hand)", ""] + [f"- {n}: {fmt(v, pct=True)}" for n, v in thinking]
    (Path(a.results) / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump(sums, open(Path(a.results) / "summary.json", "w"), indent=2, default=float)
    print("\n".join(L))
    for r in runs:
        write_errors(r, items, a.results, a.n_errors)
    figures(runs, sums, a.results)


if __name__ == "__main__":
    main()
