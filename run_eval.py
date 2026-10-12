"""Ask one model every question under several evidence conditions; save answer, confidence and outcome per item.

Engine: vLLM (fast, used for the real runs) or Hugging Face transformers (fallback / smoke tests).
Output is appended in chunks, so an interrupted run resumes where it stopped.

Study v2 added: the prompts `neutral` and `own`, the `plausible` condition (plausible.py), thinking budgets and
"Wait" extension (budget.py), per-request seeds, a cost log per condition, software versions in config.json, and
several configurations with one model load (e.g. --think 0,1 --prompt warn,neutral --budget 0,128 --seed 14,15).
The v1 command lines still work and give the v1 run names.
"""
import argparse
import gzip
import itertools
import json
import platform
import subprocess
import time
from pathlib import Path

import budget as bf
from plausible import load_items
from scoring import classify, confidence, planted_answer, split_answer

PROMPTS = {
    "base": ("Answer the question using the document below. Give only the short answer (a few words), with no "
             "explanation. If the answer cannot be determined, reply exactly: I don't know.\n\n"
             "Document: {doc}\n\nQuestion: {q}"),
    "warn": ("Answer the question using the document below. The document may contain errors. Give only the short "
             "answer (a few words), with no explanation. If the answer cannot be determined, reply exactly: "
             "I don't know.\n\nDocument: {doc}\n\nQuestion: {q}"),
    # v2: no instruction to use the document
    "neutral": ("Here is a document that may be relevant. Answer the question. Give only the short answer (a few "
                "words), with no explanation. If you don't know the answer, reply exactly: I don't know.\n\n"
                "Document: {doc}\n\nQuestion: {q}"),
    # v2: explicit permission to prefer what the model knows (the upper bound of what prompting can do here)
    "own": ("Here is a document that may be relevant. It may contain errors: if it contradicts what you know, answer "
            "with what you know. Answer the question. Give only the short answer (a few words), with no explanation. "
            "If you don't know the answer, reply exactly: I don't know.\n\nDocument: {doc}\n\nQuestion: {q}"),
    "closed_book": ("Answer the question. Give only the short answer (a few words), with no explanation. If you "
                    "don't know the answer, reply exactly: I don't know.\n\nQuestion: {q}"),
}
DOC_FIELD = {"gold": "org_context", "misleading": "sub_context", "irrelevant": "irr_context",
             "plausible": "plaus_context"}
# Sampling recommended by the Qwen3 model card for each mode; "greedy" is available for comparison (the card advises
# against greedy decoding in thinking mode).
DECODING = {True: dict(temperature=0.6, top_p=0.95, top_k=20), False: dict(temperature=0.7, top_p=0.8, top_k=20)}


def gpu_energy_j():
    """Total energy used by all GPUs since driver load, in joules (NVML; None if unavailable)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        return sum(pynvml.nvmlDeviceGetTotalEnergyConsumption(pynvml.nvmlDeviceGetHandleByIndex(i))
                   for i in range(pynvml.nvmlDeviceGetCount())) / 1000
    except Exception:
        return None


def question(q):
    q = q.strip()
    q = q[0].upper() + q[1:]
    return q if q.endswith("?") else q + "?"


def user_message(item, cond, prompt):
    if cond == "closed_book":
        return PROMPTS["closed_book"].format(q=question(item["question"]))
    return PROMPTS[prompt].format(doc=item[DOC_FIELD[cond]], q=question(item["question"]))


def run_name(cfg):
    """v1 names stay the same (e.g. Qwen3-8B_think, Qwen3-8B_think_warn); v2 adds _b<budget>, _wait<n>, _s<seed>."""
    n = f"{cfg['model'].split('/')[-1]}_{'think' if cfg['think'] else 'nothink'}"
    n += "" if cfg["prompt"] == "base" else f"_{cfg['prompt']}"
    n += "" if cfg.get("budget") is None else f"_b{cfg['budget']}"
    n += f"_wait{cfg['wait']}" if cfg.get("wait") else ""
    n += "_greedy" if cfg["decoding"] == "greedy" else ""
    return n + ("" if cfg["seed"] == 13 else f"_s{cfg['seed']}")


def versions():
    v = {"python": platform.python_version()}
    for m in ("vllm", "torch", "transformers"):
        try:
            v[m] = __import__(m).__version__
        except Exception:
            pass
    try:
        import torch
        v["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    here = Path(__file__).parent
    try:
        v["code_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                          cwd=here).stdout.strip()
    except Exception:
        pass
    if not v.get("code_commit") and (here / "COMMIT.txt").exists():  # code uploaded as a zip, without .git
        v["code_commit"] = (here / "COMMIT.txt").read_text().strip()
    return v


class VLLMEngine:
    def __init__(self, a):
        import torch
        from vllm import LLM
        dtype = a.dtype
        if dtype == "auto" and torch.cuda.get_device_capability()[0] < 8:
            dtype = "half"  # T4 / P100 have no bfloat16
        self.llm = LLM(model=a.model, dtype=dtype, seed=a.engine_seed, gpu_memory_utilization=a.gpu_mem,
                       max_model_len=a.max_model_len, tensor_parallel_size=a.tp, enable_prefix_caching=True)
        self.tok = self.llm.get_tokenizer()
        self.eos_ids = set(self.tok.all_special_ids)

    def generate(self, prompts, dec, max_new, seeds, stop_ids=()):
        from vllm import SamplingParams
        mx = max_new if isinstance(max_new, list) else [max_new] * len(prompts)
        sps = [SamplingParams(max_tokens=m, logprobs=1, seed=s, stop_token_ids=list(stop_ids), **dec)
               for m, s in zip(mx, seeds)]
        res = []
        for o in self.llm.generate([{"prompt_token_ids": p} for p in prompts], sps, use_tqdm=False):
            c = o.outputs[0]
            ids = list(c.token_ids)
            lps = [d[t].logprob for d, t in zip(c.logprobs, ids)]
            fin = c.finish_reason == "stop"
            if fin and isinstance(c.stop_reason, int) and ids[-1:] != [c.stop_reason]:
                ids, lps = ids + [c.stop_reason], lps + [0.0]  # keep the stop id as the last id
            res.append((ids, lps, fin))
        return res


class HFEngine:
    def __init__(self, a):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(a.model)
        self.tok.padding_side = "left"
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype, device_map="auto").eval()
        self.bs = a.batch_size
        eos = self.model.generation_config.eos_token_id
        self.stop = set(eos if isinstance(eos, list) else [eos])
        self.eos_ids = set(self.tok.all_special_ids)

    def generate(self, prompts, dec, max_new, seeds, stop_ids=()):
        torch = self.torch
        mx = max_new if isinstance(max_new, list) else [max_new] * len(prompts)
        kw = dict(do_sample=False) if dec.get("temperature", 0) == 0 else dict(do_sample=True, **dec)
        stops = self.stop | set(stop_ids)
        res = []
        for i in range(0, len(prompts), self.bs):
            batch = prompts[i:i + self.bs]
            torch.manual_seed(seeds[i])  # one seed per batch: the HF engine is a fallback, not used for the runs
            enc = self.tok.pad({"input_ids": batch}, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                seqs = self.model.generate(**enc, max_new_tokens=max(mx[i:i + self.bs]), eos_token_id=list(stops),
                                           pad_token_id=self.tok.pad_token_id, **kw)
            for j, (row, p) in enumerate(zip(seqs, batch)):
                gen = row[enc.input_ids.shape[1]:].tolist()[:mx[i + j]]
                cut = next((k + 1 for k, t in enumerate(gen) if t in stops), None)
                gen = gen[:cut] if cut else gen
                # Raw (untempered) log-probs of the generated tokens via one teacher-forced pass
                ids = torch.tensor([p + gen], device=self.model.device)
                with torch.no_grad():
                    lp = torch.log_softmax(self.model(ids).logits[0, len(p) - 1:-1].float(), -1)
                res.append((gen, lp.gather(-1, ids[0, len(p):, None]).squeeze(-1).tolist(), cut is not None))
        return res


def run_config(eng, items, cfg, out_root, chunk):
    """One configuration (model, thinking, prompt, budget, wait, decoding, seed) on its conditions; resumable."""
    think = cfg["think"]
    dec = dict(temperature=0.0) if cfg["decoding"] == "greedy" else DECODING[think]
    run = run_name(cfg)
    out_dir = Path(out_root) / run
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    todo = pending(cfg, items, out_root)
    print(f"{run}: {len(todo)} to go", flush=True)
    if not todo:
        return run
    tok = eng.tok
    think_end = tok.get_vocab().get("</think>")
    special = set(tok.all_special_ids)
    encode = lambda s: tok.encode(s, add_special_tokens=False)
    max_new = cfg["max_new_tokens"] or (bf.THINK_MAX if think else bf.ANSWER_MAX)
    old = json.load(open(out_dir / "config.json", encoding="utf-8")) if (out_dir / "config.json").exists() else {}
    calls = old.get("calls", []) + [{"conditions": sorted({c for _, c in todo}), "versions": versions()}]
    conds = old.get("conditions", []) + [c for c in cfg["conditions"] if c not in old.get("conditions", [])]
    json.dump({**cfg, "conditions": conds, "decoding": dec, "decoding_name": cfg["decoding"], "max_new_tokens": max_new,
               "answer_max_tokens": bf.ANSWER_MAX, "n_items": len(items), "chunk": chunk,
               "seeding": "per request: crc32(seed|id|condition|step)", "calls": calls},
              open(out_dir / "config.json", "w"), indent=2)

    t0 = time.time()
    for cond in cfg["conditions"]:
        part = [x for x in todo if x[1] == cond]
        for s in range(0, len(part), chunk):  # chunks never mix conditions, so cost is known per condition
            batch = part[s:s + chunk]
            prompts = [encode(tok.apply_chat_template([{"role": "user", "content": user_message(it, c, cfg["prompt"])}],
                                                      tokenize=False, add_generation_prompt=True, enable_thinking=think))
                       for it, c in batch]
            e0, t1 = gpu_energy_j(), time.time()
            outs = bf.generate(eng, prompts, [(it["id"], c) for it, c in batch], dec, cfg["seed"], think, think_end,
                               encode, budget=cfg.get("budget"), wait=cfg.get("wait", 0),
                               think_max=max_new if think else bf.THINK_MAX,
                               answer_max=bf.ANSWER_MAX if think else max_new)
            e1, dt = gpu_energy_j(), time.time() - t1
            with open(out_dir / "usage.jsonl", "a", encoding="utf-8") as f:  # generation only (no model loading)
                f.write(json.dumps({"condition": cond, "chunk_start": s, "n_prompts": len(batch), "seconds": round(dt, 2),
                                    "energy_j": None if e0 is None or e1 is None else round(e1 - e0, 1),
                                    "generated_tokens": sum(bf.n_generated(ids, lps) for ids, lps, _ in outs),
                                    "inserted_tokens": sum(lps.count(None) for _, lps, _ in outs)}) + "\n")
            with open(pred_path, "a", encoding="utf-8") as f:
                for (it, c), (ids, lps, forced) in zip(batch, outs):
                    reason, ans_ids, ans_lps, trunc = split_answer(ids, lps, think_end, special, think, tok.decode)
                    answer = None if trunc else tok.decode(ans_ids).strip()
                    f.write(json.dumps({
                        "id": it["id"], "condition": c, "answer": answer, "confidence": confidence(ans_lps),
                        "outcome": classify(answer, it["org_answer"], planted_answer(it, c)),
                        "n_generated": bf.n_generated(ids, lps),
                        "n_thinking": bf.n_generated(ids, lps, think_end) if think else 0,
                        "n_inserted": lps.count(None), "forced_stop": forced,
                        "reasoning": tok.decode(reason, skip_special_tokens=True).strip() if think else "",
                    }, ensure_ascii=False) + "\n")
            print(f"  {cond}: {s + len(batch)}/{len(part)}  ({time.time() - t0:.0f}s)", flush=True)
    return run


def configs(model, thinks, prompts, budgets, seeds, wait, decoding, conditions, max_new_tokens):
    """All combinations; budgets and "Wait" only apply with thinking on."""
    out = []
    for think, prompt, b, seed in itertools.product(thinks, prompts, budgets, seeds):
        if not think and (b is not None or wait):
            continue
        cfg = dict(model=model, think=bool(think), prompt=prompt, budget=b, wait=wait if think else 0,
                   decoding=decoding, seed=seed, conditions=list(conditions), max_new_tokens=max_new_tokens)
        if cfg not in out:
            out.append(cfg)
    return out


def ints(s, allow_none=False):
    return [None if allow_none and x.strip().lower() in ("", "none", "unlimited") else int(x) for x in str(s).split(",")]


def engine_args(ap):
    ap.add_argument("--engine", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--batch-size", type=int, default=16, help="HF engine only")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--tp", type=int, default=1, help="number of GPUs (e.g. 2 on Kaggle T4 x2)")


def make_engine(a):
    return VLLMEngine(a) if a.engine == "vllm" else HFEngine(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--think", default="0", help="0, 1, or both: 0,1")
    ap.add_argument("--data", default="data/items.jsonl")
    ap.add_argument("--plausible", default="data/plausible_answers.jsonl", help="needed for the plausible condition")
    ap.add_argument("--conditions", default="closed_book,gold,misleading,irrelevant")
    ap.add_argument("--prompt", default="base", help=f"one or more of {', '.join(p for p in PROMPTS if p != 'closed_book')}")
    ap.add_argument("--budget", default="", help="thinking budget(s) in tokens, e.g. 0,128,512 (empty = unlimited)")
    ap.add_argument("--wait", type=int, default=0, help="replace the first N ends of thinking with 'Wait'")
    ap.add_argument("--decoding", choices=["recommended", "greedy"], default="recommended")
    ap.add_argument("--limit", type=int, default=0, help="only the first N items (smoke test)")
    ap.add_argument("--max-new-tokens", type=int, default=0, help="default: 64 without thinking, 6144 for thinking")
    ap.add_argument("--chunk", type=int, default=400, help="prompts per saved chunk")
    ap.add_argument("--seed", default="13", help="one or more seeds, e.g. 14,15")
    ap.add_argument("--out", default="results")
    engine_args(ap)
    a = ap.parse_args()
    prompts = a.prompt.split(",")
    assert all(p in PROMPTS and p != "closed_book" for p in prompts), f"unknown prompt in {a.prompt}"
    items = load_items(a.data, a.plausible if "plausible" in a.conditions else None)
    items = items[:a.limit] if a.limit else items
    cfgs = configs(a.model, ints(a.think), prompts, ints(a.budget, allow_none=True), ints(a.seed), a.wait,
                   a.decoding, a.conditions.split(","), a.max_new_tokens)
    run_all(a, items, cfgs)


def run_all(a, items, cfgs):
    """Run the configurations in order with one model load (the model is loaded only if there is work to do)."""
    a.engine_seed = cfgs[0]["seed"]
    eng = None
    for cfg in cfgs:
        if eng is None and not pending(cfg, items, a.out):
            print(f"{run_name(cfg)}: complete")
            continue
        eng = eng or make_engine(a)
        run_config(eng, items, cfg, a.out, a.chunk)


def pending(cfg, items, out_root):
    """(item, condition) pairs not yet in the run's predictions.jsonl (or its compressed copy)."""
    d, done = Path(out_root) / run_name(cfg), set()
    for path, opener in ((d / "predictions.jsonl.gz", gzip.open), (d / "predictions.jsonl", open)):
        if path.exists():
            with opener(path, "rt", encoding="utf-8") as f:
                done |= {(r["id"], r["condition"]) for r in map(json.loads, f)}
    return [(it, c) for c in cfg["conditions"] for it in items
            if (it["id"], c) not in done and (c != "plausible" or "plaus_context" in it)]


if __name__ == "__main__":
    main()
