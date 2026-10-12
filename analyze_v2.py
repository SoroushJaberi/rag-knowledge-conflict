"""Study v2, Step 1 analysis: prompts, thinking budgets, seeds, plausible swaps, trace labels.

python analyze_v2.py     (v1 runs in results/ + v2 runs in results_v2/ -> results_v2/summary_step1.md and .json)

Known facts: a fact counts as known if the closed-book answer is correct in the majority of the base replicates of
that model and mode (v1 seed 13, v2 seeds 14 and 15; whichever exist). The same fixed set is used for every prompt and
budget, so comparisons are paired (exact sign test on the facts whose outcome changed).
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze import fmt, read_predictions, size_of
from metrics import auroc, bootstrap_ci, cohen_kappa, sign_test
from plausible import load_items
from scoring import classify, matches, mentions_conflict, normalize, planted_answer
from trace_labels import QUESTIONS, label_files

CB_DOCS = ["closed_book", "gold", "misleading"]
PROMPT_ORDER = ["base", "warn", "neutral", "own"]


# ---------- loading ----------
def load(roots, items):
    runs = {}
    for root in roots:
        for cfg_path in sorted(Path(root).glob("*/config.json")):
            d = cfg_path.parent
            cfg = json.load(open(cfg_path, encoding="utf-8"))
            cfg.setdefault("budget", None)
            cfg.setdefault("wait", 0)
            cfg.setdefault("decoding_name", "greedy" if cfg["decoding"].get("temperature") == 0 else "recommended")
            cfg["v1"] = "seeding" not in cfg  # v1 code: one seed per chunk, cost log not split by condition
            by = defaultdict(dict)
            for p in read_predictions(d):
                it = items.get(p["id"])
                if it is None or (p["condition"] == "plausible" and "plaus_answer" not in it):
                    continue
                p["outcome"] = classify(p["answer"], it["org_answer"], planted_answer(it, p["condition"]))
                by[p["condition"]][p["id"]] = p
            jl = lambda f: [json.loads(line) for line in open(d / f, encoding="utf-8")] if (d / f).exists() else []
            name = d.name if d.name not in runs else f"{Path(root).name}/{d.name}"
            runs[name] = dict(name=name, dir=d.name, root=root, cfg=cfg, by=dict(by), usage=jl("usage.jsonl"),
                              judge={(r["id"], r["condition"]): r for r in jl("judge.jsonl")}, size=size_of(cfg["model"]))
    return runs


def find(runs, model, think, prompt="base", budget=None, wait=0, seed=None, v1=False, decoding="recommended"):
    for r in runs.values():
        c = r["cfg"]
        if (c["model"], c["think"], c["prompt"], c["budget"], c["wait"], c["decoding_name"], c["v1"]) == \
                (model, think, prompt, budget, wait, decoding, v1) and (seed is None or c["seed"] == seed):
            return r
    return None


def closed_book(runs, run):
    """The run's own closed-book answers, or those of the base run with the same model, mode, seed and code."""
    if "closed_book" in run["by"]:
        return run["by"]["closed_book"]
    c = run["cfg"]
    ref = find(runs, c["model"], c["think"], seed=c["seed"], v1=c["v1"], decoding=c["decoding_name"])
    return ref["by"].get("closed_book") if ref else None


def replicates(runs, model, think):
    reps = [find(runs, model, think, seed=13, v1=True)] + [find(runs, model, think, seed=s) for s in (14, 15)]
    return [r for r in reps if r and "closed_book" in r["by"]]


def known_set(runs, model, think):
    """Facts answered correctly without a document in the majority of the base replicates."""
    reps = replicates(runs, model, think)
    votes = defaultdict(int)
    for r in reps:
        for i, p in r["by"]["closed_book"].items():
            votes[i] += p["outcome"] == "original"
    return {i for i, v in votes.items() if v > len(reps) / 2}, [r["name"] for r in reps]


# ---------- measures ----------
def rates(run, cond, ids):
    ps = [run["by"][cond][i] for i in sorted(ids) if i in run["by"].get(cond, {})]
    if not ps:
        return None
    o = np.array([p["outcome"] for p in ps])
    ow = (o == "substituted").astype(float)
    conf = [p["confidence"] for p in ps if p["outcome"] == "substituted" and p["confidence"] is not None]
    return dict(n=len(ps), overwrite=float(ow.mean()), ci=bootstrap_ci(np.mean, ow),
                kept=float(np.isin(o, ["original", "both"]).mean()), idk=float((o == "abstain").mean()),
                conf_followed=float(np.mean(conf)) if conf else None)


def accuracy(run, cond, ids=None):
    ps = [p for i, p in run["by"].get(cond, {}).items() if ids is None or i in ids]
    return float(np.mean([p["outcome"] in ("original", "both") for p in ps])) if ps else None


def share(run, cond, outcomes):
    ps = list(run["by"].get(cond, {}).values())
    return float(np.mean([p["outcome"] in outcomes for p in ps])) if ps else None


def paired(ref, ref_cond, run, cond, ids):
    """Facts overwritten only in `run` (more) or only in `ref` (fewer), with the exact sign test p."""
    a, b = ref["by"].get(ref_cond, {}), run["by"].get(cond, {})
    ids = [i for i in ids if i in a and i in b]
    more = sum(b[i]["outcome"] == "substituted" and a[i]["outcome"] != "substituted" for i in ids)
    fewer = sum(a[i]["outcome"] == "substituted" and b[i]["outcome"] != "substituted" for i in ids)
    return dict(n=len(ids), more=more, fewer=fewer, p=sign_test(more, fewer))


def cost(run, conds=None):
    """Generation cost per answer. v2 runs log cost per condition; v1 runs only for all four conditions together."""
    u = [x for x in run["usage"] if conds is None or x.get("condition") in conds]
    n = sum(x["n_prompts"] for x in u)
    if not n:
        return None
    e = [x["energy_j"] for x in u]
    return dict(tokens=sum(x["generated_tokens"] for x in u) / n, gpu_s_per_1000=1000 * sum(x["seconds"] for x in u) / n,
                wh_per_1000=None if None in e else 1000 * sum(e) / 3600 / n)


def mean_of(run, cond, key, ids=None):
    ps = [p for i, p in run["by"].get(cond, {}).items() if ids is None or i in ids]
    return float(np.mean([float(p.get(key) or 0) for p in ps])) if ps else None


def disagree(a, b):
    """'Ask twice': the closed-book answer and the with-document answer name different things."""
    if not a or not b or "know" in a.lower() or "know" in b.lower():
        return False
    return not (matches(a, [b]) or matches(b, [a]) or normalize(a) == normalize(b))


def ask_twice_auroc(cb, gold, wrong, ids):
    ids = [i for i in ids if i in cb and i in gold and i in wrong]
    x = [float(disagree(cb[i]["answer"], d[i]["answer"])) for d in (gold, wrong) for i in ids]
    return auroc(x, [0] * len(ids) + [1] * len(ids))


# ---------- formatting ----------
def pct(x, ci=None):
    return fmt(x, ci, pct=True)


def num(x, d=0):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def overwrite_cell(k):
    return pct(k["overwrite"], k["ci"]) if k else "–"


def paired_cell(pr):
    if not pr:
        return "–"
    return f"{pr['fewer']} / {pr['more']}" + ("" if pr["fewer"] + pr["more"] == 0 else f" (p = {pr['p']:.2g})")


def short(model):
    return model.split("/")[-1]


def models_of(runs):
    return sorted({r["cfg"]["model"] for r in runs.values() if not r["cfg"]["v1"]}, key=size_of)


# ---------- sections ----------
def section_v1_lengths(runs, L, S):
    """Why the budgets are 0-512: how long the v1 reasoning was (generated tokens per answer, thinking runs)."""
    rows = [r for r in runs.values() if r["cfg"]["v1"] and r["cfg"]["think"] and r["cfg"]["prompt"] == "base"
            and r["size"] >= 4]
    if not rows:
        return
    cuts = (128, 256, 512, 1024, 2048)
    L += ["## 0. Reasoning length in v1 (why the budgets stop at 512)", "",
          "Share of v1 answers (base prompt, thinking on) that used more than N generated tokens.", "",
          "| Run | Condition | Median tokens | " + " | ".join(f"> {c}" for c in cuts) + " |", "|---|---|---|" + "---|" * len(cuts)]
    for r in sorted(rows, key=lambda r: r["size"]):
        for cond in ("gold", "misleading"):
            n = np.array([p["n_generated"] for p in r["by"][cond].values()])
            S["v1_lengths"][f"{r['name']}/{cond}"] = {"median": float(np.median(n)), **{str(c): float(np.mean(n > c)) for c in cuts}}
            L.append(f"| {r['name']} | {cond} | {np.median(n):.0f} | " + " | ".join(pct(np.mean(n > c)) for c in cuts) + " |")
    L.append("")


def section_prompts(runs, L, S):
    L += ["## 1. Prompt variants (H1): is overwriting driven by the instruction to use the document?", "",
          "Wrong document = NQ-Swap swap. `base` and `warn` tell the model to use the document; `neutral` only says the "
          "document may be relevant; `own` also allows answering from what the model knows. Seed 14 for all. "
          "\"Fewer / more\" = known facts overwritten only with `base` / only with this prompt.", "",
          "| Model | Thinking | Prompt | Known facts | Overwritten [95% CI] | Kept true answer | I don't know | "
          "Fewer / more than base | Correct doc: accuracy (all) | Correct doc: accuracy (unknown facts) | "
          "Unrelated doc: I don't know | Unrelated doc: correct from memory |", "|" + "---|" * 12]
    for model in models_of(runs):
        for think in (False, True):
            known, _ = known_set(runs, model, think)
            ref = find(runs, model, think, seed=14)
            for prompt in PROMPT_ORDER:
                r = find(runs, model, think, prompt=prompt, seed=14)
                if not r or not known:
                    continue
                k = rates(r, "misleading", known)
                pr = paired(ref, "misleading", r, "misleading", known) if ref and prompt != "base" else None
                unknown = set(r["by"].get("gold", {})) - known
                row = dict(known=k, vs_base=pr, gold_all=accuracy(r, "gold"), gold_unknown=accuracy(r, "gold", unknown),
                           irrelevant_idk=share(r, "irrelevant", ("abstain",)),
                           irrelevant_memory=share(r, "irrelevant", ("original", "both")))
                S["prompts"][r["name"]] = row
                L.append(f"| {short(model)} | {'on' if think else 'off'} | {prompt} | {k['n'] if k else 0} | "
                         f"{overwrite_cell(k)} | {pct(k and k['kept'])} | {pct(k and k['idk'])} | {paired_cell(pr)} | "
                         f"{pct(row['gold_all'])} | {pct(row['gold_unknown'])} | {pct(row['irrelevant_idk'])} | "
                         f"{pct(row['irrelevant_memory'])} |")
    L.append("")


def section_budgets(runs, L, S):
    L += ["## 2. Thinking budget (H2): does more thinking reduce overwriting, and what does it cost?", "",
          "Base prompt, seed 14. Budget = maximum thinking tokens; at the limit the Qwen3 stop-thinking text is inserted "
          "and the model answers. `unlimited` = up to 6,144 tokens (as in v1). `wait1` = the model's first attempt to "
          "stop thinking is replaced by \"Wait\". Overwriting on the fixed known-fact set of the thinking mode; cost per "
          "answer over the closed-book, correct-doc and wrong-doc conditions.", "",
          "| Model | Setting | Thinking tokens (wrong doc, mean) | Stopped by the budget (wrong doc) | Closed-book accuracy | "
          "Known facts | Overwritten [95% CI] | Fewer / more than unlimited | Correct doc: accuracy | Tokens per answer | "
          "GPU s per 1,000 | Wh per 1,000 |", "|" + "---|" * 12]
    curves = {}
    for model in models_of(runs):
        known, _ = known_set(runs, model, True)
        ref = find(runs, model, True, seed=14)
        rows = [("no thinking", find(runs, model, False, seed=14))]
        rows += [(f"budget {b}", find(runs, model, True, budget=b, seed=14)) for b in (0, 128, 256, 512)]
        rows += [("unlimited", ref), ("wait1", find(runs, model, True, wait=1, seed=14))]
        for label, r in rows:
            if not r or not known:
                continue
            k = rates(r, "misleading", known)
            pr = paired(ref, "misleading", r, "misleading", known) if ref and r is not ref else None
            c = cost(r, CB_DOCS)
            row = dict(thinking_tokens=mean_of(r, "misleading", "n_thinking"),
                       forced=mean_of(r, "misleading", "forced_stop"), closed_book=accuracy(r, "closed_book"),
                       known=k, vs_unlimited=pr, gold=accuracy(r, "gold"), cost=c)
            S["budgets"][r["name"]] = row
            curves.setdefault(model, []).append((label, row))
            L.append(f"| {short(model)} | {label} | {num(row['thinking_tokens'])} | {pct(row['forced'])} | "
                     f"{pct(row['closed_book'])} | {k['n'] if k else 0} | {overwrite_cell(k)} | {paired_cell(pr)} | "
                     f"{pct(row['gold'])} | {num(c and c['tokens'])} | {num(c and c['gpu_s_per_1000'])} | "
                     f"{num(c and c['wh_per_1000'], 1)} |")
    L.append("")
    return curves


def section_seeds(runs, L, S):
    L += ["## 3. Seeds: are the v1 numbers luck?", "",
          "Base prompt. v1 = the published run (seed 13, v1 code); s14 and s15 = v2 code. \"Own known facts\" = correct "
          "closed-book answer in that run (the v1 definition); \"fixed set\" = majority of the replicates.", "",
          "| Model | Thinking | Run | Closed-book accuracy | Own known facts | Overwritten (own known) | "
          "Overwritten (fixed set) | Unrelated doc: I don't know |", "|---|---|---|---|---|---|---|---|"]
    for model in models_of(runs):
        for think in (False, True):
            known, names = known_set(runs, model, think)
            reps = replicates(runs, model, think)
            own_rates = []
            for r in reps:
                own = {i for i, p in r["by"]["closed_book"].items() if p["outcome"] == "original"}
                ko, kf = rates(r, "misleading", own), rates(r, "misleading", known)
                own_rates.append(ko["overwrite"] if ko else np.nan)
                S["seeds"][r["name"]] = dict(closed_book=accuracy(r, "closed_book"), own=ko, fixed=kf)
                L.append(f"| {short(model)} | {'on' if think else 'off'} | {r['name']} | {pct(accuracy(r, 'closed_book'))} | "
                         f"{ko['n'] if ko else 0} | {overwrite_cell(ko)} | {overwrite_cell(kf)} | "
                         f"{pct(share(r, 'irrelevant', ('abstain',)))} |")
            if len(reps) > 1:
                both = [i for i in known if all(i in r["by"].get("misleading", {}) for r in reps)]
                same = np.mean([len({r["by"]["misleading"][i]["outcome"] == "substituted" for r in reps}) == 1
                                for i in both]) if both else np.nan
                S["seeds"][f"{short(model)}_{'think' if think else 'nothink'}_spread"] = dict(
                    mean=float(np.nanmean(own_rates)), sd=float(np.nanstd(own_rates, ddof=1)), same_outcome=float(same),
                    n_fixed=len(known), replicates=names)
                L.append(f"| {short(model)} | {'on' if think else 'off'} | mean ± SD over {len(reps)} runs | | | "
                         f"{100 * np.nanmean(own_rates):.1f}% ± {100 * np.nanstd(own_rates, ddof=1):.1f} | "
                         f"same outcome in all runs: {pct(same)} of {len(both)} fixed-set facts | |")
    L.append("")


def section_plausible(runs, items, L, S):
    n_pl = sum("plaus_answer" in it for it in items.values())
    L += ["## 4. Plausible wrong documents: do models resist only absurd swaps?", "",
          f"{n_pl} of {len(items)} items have a checked plausible swap (plausible.py). Same known facts, same seed (14): "
          "NQ-Swap swap vs plausible swap. \"Ask twice\" = flag the document when the closed-book answer and the "
          "with-document answer differ (AUROC, correct doc vs wrong doc).", "",
          "| Model | Thinking | Known facts | Overwritten: NQ-Swap | Overwritten: plausible | Fewer / more with plausible | "
          "Confidence when following: NQ-Swap / plausible | Ask twice AUROC, known: NQ-Swap / plausible | "
          "Ask twice AUROC, all: NQ-Swap / plausible |", "|" + "---|" * 9]
    for model in models_of(runs):
        for think in (False, True):
            ref = find(runs, model, think, seed=14)
            pl = next((r for r in runs.values() if "plausible" in r["by"] and r["cfg"]["model"] == model
                       and r["cfg"]["think"] == think and r["cfg"]["seed"] == 14 and r["cfg"]["prompt"] == "base"), None)
            if not ref or not pl:
                continue
            known, _ = known_set(runs, model, think)
            ids = [i for i in known if i in pl["by"]["plausible"]]
            a, b = rates(ref, "misleading", ids), rates(pl, "plausible", ids)
            pr = paired(ref, "misleading", pl, "plausible", ids)
            cb, gold = closed_book(runs, ref), ref["by"]["gold"]
            alls = list(pl["by"]["plausible"])
            auc = [ask_twice_auroc(cb, gold, w, s) for s in (ids, alls) for w in (ref["by"]["misleading"], pl["by"]["plausible"])]
            S["plausible"][pl["name"]] = dict(nq_swap=a, plausible=b, paired=pr, ask_twice=auc)
            L.append(f"| {short(model)} | {'on' if think else 'off'} | {len(ids)} | {overwrite_cell(a)} | {overwrite_cell(b)} | "
                     f"{paired_cell(pr)} | {num(a and a['conf_followed'], 2)} / {num(b and b['conf_followed'], 2)} | "
                     f"{num(auc[0], 2)} / {num(auc[1], 2)} | {num(auc[2], 2)} / {num(auc[3], 2)} |")
    L.append("")


def section_labels(runs, L, S, labels_dir):
    judged = [r for r in runs.values() if r["judge"]]
    if not judged:
        return
    L += ["## 5. What the reasoning says (LLM judge; check the agreement table before using these numbers)", "",
          "Known facts only. Share of traces where the judge answers yes (P(yes) > 0.5). Keyword = the v1 heuristic.", "",
          "| Run | Wrong doc, overwritten: n | states true fact | notices conflict | cites instruction | keyword | "
          "Correct doc: n | notices conflict (false alarm) | keyword |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(judged, key=lambda r: (r["size"], r["name"])):
        cb = closed_book(runs, r) or {}
        sel = lambda cond, outcome=None: [(i, p) for i, p in r["by"].get(cond, {}).items() if (i, cond) in r["judge"]
                                          and cb.get(i, {}).get("outcome") == "original"
                                          and (outcome is None or p["outcome"] == outcome)]
        yes = lambda pairs, cond, q: np.mean([(r["judge"][(i, cond)][q] or 0) > 0.5 for i, _ in pairs]) if pairs else None
        kw = lambda pairs: np.mean([mentions_conflict(p["reasoning"]) for _, p in pairs]) if pairs else None
        cond = "plausible" if "plausible" in r["by"] else "misleading"
        ow, gd = sel(cond, "substituted"), sel("gold")
        row = dict(n_overwritten=len(ow), **{q: yes(ow, cond, q) for q in QUESTIONS}, keyword=kw(ow),
                   n_gold=len(gd), gold_conflict=yes(gd, "gold", "notices_conflict"), gold_keyword=kw(gd))
        S["labels"][r["name"]] = row
        L.append(f"| {r['name']} | {len(ow)} | {pct(row['states_true_fact'])} | {pct(row['notices_conflict'])} | "
                 f"{pct(row['cites_instruction'])} | {pct(row['keyword'])} | {len(gd)} | {pct(row['gold_conflict'])} | "
                 f"{pct(row['gold_keyword'])} |")
    L.append("")
    sheet_path, key_path = label_files(labels_dir)
    if not sheet_path.exists() or not key_path.exists():
        return
    key = {r["row"]: r for r in csv.DictReader(open(key_path, encoding="utf-8"))}
    hand = [r for r in csv.DictReader(open(sheet_path, encoding="utf-8-sig")) if all(r[q].strip() for q in QUESTIONS)]
    if not hand:
        L += [f"Hand labels: none filled in yet ({sheet_path.as_posix()}).", ""]
        return
    L += ["### Agreement of the judge with the hand labels", "",
          "| Question | n | Agreement | Cohen's kappa | Keyword heuristic: kappa |", "|---|---|---|---|---|"]
    for q in QUESTIONS:
        pairs = []
        for h in hand:
            k = key[h["row"]]
            r = runs.get(k["run"])
            j = r and r["judge"].get((k["id"], k["condition"]))
            if j and j[q] is not None:
                kwd = mentions_conflict(r["by"][k["condition"]][k["id"]]["reasoning"])
                pairs.append((h[q].strip().lower(), "yes" if j[q] > 0.5 else "no", "yes" if kwd else "no"))
        if pairs:
            hs, js, ks = zip(*pairs)
            row = dict(n=len(pairs), agreement=float(np.mean([a == b for a, b in zip(hs, js)])), kappa=cohen_kappa(hs, js),
                       keyword_kappa=cohen_kappa(hs, ks) if q == "notices_conflict" else None)
            S["agreement"][q] = row
            L.append(f"| {q} | {row['n']} | {pct(row['agreement'])} | {num(row['kappa'], 2)} | "
                     f"{num(row['keyword_kappa'], 2)} |")
    L.append("")


def section_costs(runs, L):
    L += ["## 6. All v2 runs: generation cost", "", "Per answer, over all conditions of the run (generation only).", "",
          "| Run | Conditions | Answers | Tokens per answer | GPU s per 1,000 | Wh per 1,000 | Software |",
          "|---|---|---|---|---|---|---|"]
    for r in sorted(runs.values(), key=lambda r: (r["size"], r["name"])):
        if r["cfg"]["v1"]:
            continue
        c = cost(r)
        sw = sorted({f"vLLM {v['versions'].get('vllm', '?')}, {v['versions'].get('gpu', '?')}" for v in r["cfg"]["calls"]})
        L.append(f"| {r['name']} | {', '.join(r['by'])} | {sum(len(x) for x in r['by'].values())} | "
                 f"{num(c and c['tokens'])} | {num(c and c['gpu_s_per_1000'])} | {num(c and c['wh_per_1000'], 1)} | "
                 f"{'; '.join(sw)} |")
    L.append("")


def figure(curves, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping the figure")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for model, rows in curves.items():
        pts = [(row["cost"]["tokens"], row["known"]["overwrite"], row["closed_book"], label) for label, row in rows
               if row["cost"] and row["known"] and label != "no thinking"]
        if not pts:
            continue
        x, y, cb, labels = zip(*sorted(pts))
        axes[0].plot(x, [100 * v for v in y], marker="o", label=short(model))
        axes[1].plot(x, [100 * v for v in cb], marker="o", label=short(model))
        for xi, yi, lab in zip(x, y, labels):
            axes[0].annotate(lab.replace("budget ", ""), (xi, 100 * yi), fontsize=6, xytext=(3, 3), textcoords="offset points")
    axes[0].set_ylabel("Known facts overwritten (%)")
    axes[1].set_ylabel("Closed-book accuracy (%)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Generated tokens per answer (log scale)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    plt.tight_layout()
    (out / "figures").mkdir(exist_ok=True)
    plt.savefig(out / "figures" / "budget_curves.png", dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results,results_v2")
    ap.add_argument("--data", default="data/items.jsonl")
    ap.add_argument("--plausible", default="data/plausible_answers.jsonl")
    ap.add_argument("--labels", default="labels", help="folder of the hand-label sheet and its key")
    ap.add_argument("--out", default="results_v2")
    a = ap.parse_args()
    items = {it["id"]: it for it in load_items(a.data, a.plausible)}
    runs = load(a.results.split(","), items)
    S = {k: {} for k in ("v1_lengths", "prompts", "budgets", "seeds", "plausible", "labels", "agreement")}
    L = ["# Study v2, Step 1: results", "", "Generated by `analyze_v2.py`; re-scored from the stored answers.", ""]
    section_v1_lengths(runs, L, S)
    section_prompts(runs, L, S)
    curves = section_budgets(runs, L, S)
    section_seeds(runs, L, S)
    section_plausible(runs, items, L, S)
    section_labels(runs, L, S, a.labels)
    section_costs(runs, L)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    (out / "summary_step1.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump(S, open(out / "summary_step1.json", "w"), indent=2, default=float)
    figure(curves, out)
    print("\n".join(L))


if __name__ == "__main__":
    main()
