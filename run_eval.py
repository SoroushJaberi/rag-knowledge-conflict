"""Ask one model every question under four evidence conditions; save answer, confidence and outcome per item.

Engine: vLLM (fast, used for the real runs) or Hugging Face transformers (fallback / smoke tests).
Output is appended in chunks, so an interrupted run resumes where it stopped.
"""
import argparse
import json
import time
from pathlib import Path

from scoring import classify, confidence, split_answer

PROMPTS = {
    "base": ("Answer the question using the document below. Give only the short answer (a few words), with no "
             "explanation. If the answer cannot be determined, reply exactly: I don't know.\n\n"
             "Document: {doc}\n\nQuestion: {q}"),
    "warn": ("Answer the question using the document below. The document may contain errors. Give only the short "
             "answer (a few words), with no explanation. If the answer cannot be determined, reply exactly: "
             "I don't know.\n\nDocument: {doc}\n\nQuestion: {q}"),
    "closed_book": ("Answer the question. Give only the short answer (a few words), with no explanation. If you "
                    "don't know the answer, reply exactly: I don't know.\n\nQuestion: {q}"),
}
DOC_FIELD = {"gold": "org_context", "misleading": "sub_context", "irrelevant": "irr_context"}
# Sampling recommended by the Qwen3 model card for each mode; "greedy" is available for comparison.
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


class VLLMEngine:
    def __init__(self, a):
        import torch
        from vllm import LLM
        dtype = a.dtype
        if dtype == "auto" and torch.cuda.get_device_capability()[0] < 8:
            dtype = "half"  # T4 / P100 have no bfloat16
        self.llm = LLM(model=a.model, dtype=dtype, seed=a.seed, gpu_memory_utilization=a.gpu_mem,
                       max_model_len=a.max_model_len, tensor_parallel_size=a.tp)
        self.tok = self.llm.get_tokenizer()

    def generate(self, prompts, dec, max_new, seed):
        from vllm import SamplingParams
        sp = SamplingParams(max_tokens=max_new, logprobs=1, seed=seed, **dec)
        res = []
        for o in self.llm.generate(prompts, sp, use_tqdm=False):
            c = o.outputs[0]
            ids = list(c.token_ids)
            res.append((ids, [d[t].logprob for d, t in zip(c.logprobs, ids)]))
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
        self.stop = set(self.model.generation_config.eos_token_id if isinstance(
            self.model.generation_config.eos_token_id, list) else [self.model.generation_config.eos_token_id])

    def generate(self, prompts, dec, max_new, seed):
        torch = self.torch
        torch.manual_seed(seed)
        kw = dict(do_sample=False) if dec.get("temperature", 0) == 0 else dict(do_sample=True, **dec)
        res = []
        for i in range(0, len(prompts), self.bs):
            enc = self.tok(prompts[i:i + self.bs], return_tensors="pt", padding=True, add_special_tokens=False)
            enc = enc.to(self.model.device)
            with torch.no_grad():
                seqs = self.model.generate(**enc, max_new_tokens=max_new, pad_token_id=self.tok.pad_token_id, **kw)
            for row, prompt_ids, prompt_mask in zip(seqs, enc.input_ids, enc.attention_mask):
                gen = row[enc.input_ids.shape[1]:].tolist()
                cut = next((k + 1 for k, t in enumerate(gen) if t in self.stop), len(gen))
                gen = gen[:cut]
                # Raw (untempered) log-probs of the generated tokens via one teacher-forced pass
                p = prompt_ids[prompt_mask.bool()].tolist()
                ids = torch.tensor([p + gen], device=self.model.device)
                with torch.no_grad():
                    lp = torch.log_softmax(self.model(ids).logits[0, len(p) - 1:-1].float(), -1)
                res.append((gen, lp.gather(-1, ids[0, len(p):, None]).squeeze(-1).tolist()))
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--think", type=int, choices=[0, 1], default=0)
    ap.add_argument("--data", default="data/items.jsonl")
    ap.add_argument("--conditions", default="closed_book,gold,misleading,irrelevant")
    ap.add_argument("--prompt", choices=["base", "warn"], default="base")
    ap.add_argument("--decoding", choices=["recommended", "greedy"], default="recommended")
    ap.add_argument("--engine", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--limit", type=int, default=0, help="only the first N items (smoke test)")
    ap.add_argument("--max-new-tokens", type=int, default=0, help="default: 64 without thinking, 6144 with")
    ap.add_argument("--chunk", type=int, default=400, help="prompts per saved chunk")
    ap.add_argument("--batch-size", type=int, default=16, help="HF engine only")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--tp", type=int, default=1, help="number of GPUs (e.g. 2 on Kaggle T4 x2)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    think = bool(a.think)
    max_new = a.max_new_tokens or (6144 if think else 64)
    dec = dict(temperature=0.0) if a.decoding == "greedy" else DECODING[think]
    run = f"{a.model.split('/')[-1]}_{'think' if think else 'nothink'}" + ("" if a.prompt == "base" else f"_{a.prompt}")
    if a.decoding == "greedy":
        run += "_greedy"
    out_dir = Path(a.out) / run
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"

    items = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    if a.limit:
        items = items[:a.limit]
    todo = [(it, c) for it in items for c in a.conditions.split(",")]
    done = set()
    if pred_path.exists():
        done = {(r["id"], r["condition"]) for r in map(json.loads, open(pred_path, encoding="utf-8"))}
    todo = [(it, c) for it, c in todo if (it["id"], c) not in done]
    print(f"{run}: {len(done)} done, {len(todo)} to go")
    if not todo:
        return

    eng = VLLMEngine(a) if a.engine == "vllm" else HFEngine(a)
    tok = eng.tok
    vocab = tok.get_vocab()
    think_end = vocab.get("</think>")
    special = set(tok.all_special_ids)
    json.dump({"model": a.model, "think": think, "prompt": a.prompt, "decoding": dec, "max_new_tokens": max_new,
               "engine": a.engine, "seed": a.seed, "n_items": len(items)},
              open(out_dir / "config.json", "w"), indent=2)

    t0 = time.time()
    for s in range(0, len(todo), a.chunk):
        chunk = todo[s:s + a.chunk]
        prompts = [tok.apply_chat_template([{"role": "user", "content": user_message(it, c, a.prompt)}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=think)
                   for it, c in chunk]
        e0, t1 = gpu_energy_j(), time.time()
        outs = eng.generate(prompts, dec, max_new, a.seed + s)
        e1, dt = gpu_energy_j(), time.time() - t1
        with open(out_dir / "usage.jsonl", "a", encoding="utf-8") as f:  # cost of generation only (no model loading)
            f.write(json.dumps({"chunk_start": s, "n_prompts": len(chunk), "seconds": round(dt, 2),
                                "energy_j": None if e0 is None or e1 is None else round(e1 - e0, 1),
                                "generated_tokens": sum(len(ids) for ids, _ in outs)}) + "\n")
        with open(pred_path, "a", encoding="utf-8") as f:
            for (it, c), (ids, lps) in zip(chunk, outs):
                reason, ans_ids, ans_lps, trunc = split_answer(ids, lps, think_end, special, think, tok.decode)
                answer = None if trunc else tok.decode(ans_ids).strip()
                f.write(json.dumps({
                    "id": it["id"], "condition": c, "answer": answer, "confidence": confidence(ans_lps),
                    "outcome": classify(answer, it["org_answer"], it["sub_answer"]),
                    "n_generated": len(ids), "reasoning": tok.decode(reason, skip_special_tokens=True).strip() if think else "",
                }, ensure_ascii=False) + "\n")
        print(f"  {s + len(chunk)}/{len(todo)}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
