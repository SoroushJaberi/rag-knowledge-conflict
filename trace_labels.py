"""Labels for reasoning traces: a sheet for labelling by hand, and an LLM judge that answers the same three questions.

  states_true_fact   Does the reasoning state the true answer as the model's own knowledge?
  notices_conflict   Does it say that the document conflicts with what the model knows, or may be wrong?
  cites_instruction  Does it give the instruction to use the document as the reason to prefer the document?

The judge (default microsoft/phi-4, not a Qwen model) sees exactly what the hand labeller sees: the question, the
correct answer and the reasoning (not the document, not the condition). It answers each question with one token;
we store P(yes) / (P(yes) + P(no)). Agreement with the hand labels (Cohen's kappa) is reported by analyze_v2.py.

python trace_labels.py sheet                       (no GPU) 50 v1 traces -> labels/hand_labels.csv (+ hidden key)
python trace_labels.py judge                       (GPU, vLLM) writes judge.jsonl into the selected thinking runs
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

QUESTIONS = {
    "states_true_fact": ("Does the reasoning state, as the model's own knowledge (for example \"I remember\", \"I know\", "
                         "\"actually\", \"in reality\"), that the correct answer is {true} or something equivalent? "
                         "Only repeating what the document says does not count."),
    "notices_conflict": ("Does the reasoning say that the document conflicts with what the model knows, or that the "
                         "document may be wrong or contain an error?"),
    "cites_instruction": ("Does the reasoning give the instruction to answer from the document (for example \"the user "
                          "wants the answer based on the document\") as a reason to prefer the document over the "
                          "model's own knowledge?"),
}
JUDGE_PROMPT = ("Below is the reasoning that an AI model wrote before answering a question about a document.\n\n"
                "Question: {q}\nCorrect answer (from the answer key): {true}\n\nReasoning:\n\"\"\"\n{trace}\n\"\"\"\n\n"
                "{ask}\nAnswer with yes or no.")
MAX_CHARS = 30000  # spreadsheet cells hold at most 32,767 characters


def label_files(folder="labels"):
    """The sheet to fill in by hand, and the key that says which run, item and condition each row comes from."""
    return Path(folder) / "hand_labels.csv", Path(folder) / "hand_labels_key.csv"


def clean_trace(text):
    return " ".join(text.replace("<think>", " ").split())[:MAX_CHARS]


def sheet(a):
    """Stratified random sample of v1 thinking traces: per run, 15 overwritten known facts, 5 known facts that resisted
    the wrong document and 5 correct-document traces. The traces of the v1 spot check are left out."""
    import analyze_v2 as av
    items = {it["id"]: it for it in av.load_items(a.data, None)}
    runs = av.load([a.v1_results], items)
    spot_path = Path(a.v1_results) / "spotcheck_labels.csv"
    spot = {(r["model"], r["id"]) for r in csv.DictReader(open(spot_path, encoding="utf-8"))} if spot_path.exists() else set()
    rng, rows = random.Random(a.seed), []
    for name in a.sheet_runs.split(","):
        r, model = runs[name], name.split("_")[0]
        cb, mis = r["by"]["closed_book"], r["by"]["misleading"]
        known = sorted(i for i, p in cb.items() if p["outcome"] == "original" and (model, i) not in spot)
        strata = [("misleading", [i for i in known if mis[i]["outcome"] == "substituted"], 15),
                  ("misleading", [i for i in known if mis[i]["outcome"] != "substituted"], 5),
                  ("gold", known, 5)]
        for cond, pool, n in strata:
            rows += [(name, i, cond) for i in rng.sample(pool, min(n, len(pool)))]
    rng.shuffle(rows)
    sheet_path, key_path = label_files(a.labels)
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sheet_path, "w", encoding="utf-8-sig", newline="") as f, open(key_path, "w", encoding="utf-8", newline="") as g:
        w, k = csv.writer(f), csv.writer(g)
        w.writerow(["row", "question", "correct answer", "reasoning"] + list(QUESTIONS) + ["notes"])
        k.writerow(["row", "run", "id", "condition"])
        for n, (run, i, cond) in enumerate(rows, 1):
            it = items[i]
            w.writerow([n, it["question"], " | ".join(it["org_answer"]),
                        clean_trace(runs[run]["by"][cond][i]["reasoning"])] + [""] * (len(QUESTIONS) + 1))
            k.writerow([n, run, i, cond])
    print(f"wrote {sheet_path} ({len(rows)} traces) and {key_path}")


def hand_label_keys(folder):
    key_path = label_files(folder)[1]
    if not key_path.exists():
        return set()
    return {(r["run"], r["id"], r["condition"]) for r in csv.DictReader(open(key_path, encoding="utf-8"))}


class VLLMJudge:
    def __init__(self, a):
        from vllm import LLM
        self.llm = LLM(model=a.model, seed=0, max_model_len=a.max_model_len, gpu_memory_utilization=0.90,
                       enable_prefix_caching=True)
        self.tok = self.llm.get_tokenizer()

    def yes_probs(self, texts):
        from vllm import SamplingParams
        prompts = [self.tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False,
                                                add_generation_prompt=True) for t in texts]
        out = []
        for o in self.llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0, logprobs=20), use_tqdm=False):
            p = {"yes": 0.0, "no": 0.0}
            for lp in o.outputs[0].logprobs[0].values():  # the 20 most likely first tokens ("Yes", " yes", "No", ...)
                w = (lp.decoded_token or "").strip().lower()
                if w in p:
                    p[w] += math.exp(lp.logprob)
            out.append(p["yes"] / (p["yes"] + p["no"]) if p["yes"] + p["no"] > 0 else None)
        return out


def make_judge(a):
    return VLLMJudge(a)


def select(run, closed_book, hand, a):
    """Known facts (correct closed-book answer) with a non-empty trace; at most a.max_per_condition per condition
    (random, fixed per run), plus every trace in the hand-label sheet."""
    rng, out = random.Random(f"{a.seed}|{run['name']}"), []
    for cond in ("misleading", "plausible", "gold"):
        ps = run["by"].get(cond, {})
        ids = sorted(i for i, p in ps.items() if i in closed_book and closed_book[i]["outcome"] == "original"
                     and p["reasoning"].strip())
        keep = set(rng.sample(ids, min(len(ids), a.max_per_condition)))
        keep |= {i for (r, i, c) in hand if r == run["name"] and c == cond and i in ps}
        out += [(i, cond) for i in sorted(keep)]
    return out


def judge(a):
    import analyze_v2 as av
    from run_eval import gpu_energy_j
    items = {it["id"]: it for it in av.load_items(a.data, a.plausible)}
    runs = av.load(a.runs.split(","), items)
    hand = hand_label_keys(a.labels)
    sizes = set(a.models.split(","))
    eng = None
    for run in runs.values():
        cfg = run["cfg"]
        if not cfg["think"] or cfg.get("budget") == 0 or f"{run['size']:g}B" not in sizes:
            continue
        cb = av.closed_book(runs, run) or {}
        path = Path(run["root"]) / run["dir"] / "judge.jsonl"
        done = {(r["id"], r["condition"]) for r in map(json.loads, open(path, encoding="utf-8"))} if path.exists() else set()
        todo = [x for x in select(run, cb, hand, a) if x not in done]
        print(f"{run['name']}: {len(done)} judged, {len(todo)} to go", flush=True)
        if not todo:
            continue
        eng = eng or make_judge(a)
        for s in range(0, len(todo), a.chunk):
            part = todo[s:s + a.chunk]
            texts = [JUDGE_PROMPT.format(q=items[i]["question"], true=" | ".join(items[i]["org_answer"]),
                                         trace=clean_trace(run["by"][c][i]["reasoning"]),
                                         ask=QUESTIONS[k].format(true=items[i]["org_answer"][0]))
                     for i, c in part for k in QUESTIONS]
            e0, t0 = gpu_energy_j(), time.time()
            p = eng.yes_probs(texts)
            e1, dt = gpu_energy_j(), time.time() - t0
            with open(path, "a", encoding="utf-8") as f:
                for n, (i, c) in enumerate(part):
                    f.write(json.dumps({"id": i, "condition": c, "judge": a.model,
                                        **{k: p[n * len(QUESTIONS) + j] for j, k in enumerate(QUESTIONS)}}) + "\n")
            with open(path.with_name("judge_usage.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"n_traces": len(part), "n_prompts": len(texts), "seconds": round(dt, 2),
                                    "energy_j": None if e0 is None or e1 is None else round(e1 - e0, 1)}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sheet", "judge"])
    ap.add_argument("--data", default="data/items.jsonl")
    ap.add_argument("--plausible", default="data/plausible_answers.jsonl")
    ap.add_argument("--runs", default="results,results_v2", help="result folders to judge")
    ap.add_argument("--models", default="8B,14B")
    ap.add_argument("--labels", default="labels", help="folder of the hand-label sheet and its key")
    ap.add_argument("--v1-results", default="results")
    ap.add_argument("--sheet-runs", default="Qwen3-8B_think,Qwen3-14B_think")
    ap.add_argument("--model", default="microsoft/phi-4", help="judge model")
    ap.add_argument("--max-per-condition", type=int, default=300)
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--chunk", type=int, default=500, help="traces per saved chunk")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    sheet(a) if a.cmd == "sheet" else judge(a)


if __name__ == "__main__":
    main()
