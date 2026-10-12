# Study v2 (work in progress): cheap ways not to be misled by a wrong document

**Question:** what is the cheapest way to avoid being misled by a wrong retrieved document?
Study v1 (tag `v1`, [README](README.md)) found that Qwen3 models follow a wrong document for 49–68% of the facts they
know, and that thinking costs about 10–19× more GPU energy without making them resist. v2 has three steps:

1. **Make the v1 result solid** (this branch): prompt variants, thinking budget as a controlled variable, more seeds,
   plausible wrong documents, and better labels for the reasoning traces.
2. Compare cheap detectors of wrong documents by detection quality *and* cost (tokens, GPU time, energy).
3. A policy that spends extra computation only when a cheap detector is unsure.

This file was written before any v2 run. Results will be added below the hypotheses, never in place of them.

## Hypotheses (written 12 Oct 2026, before running)
- **H1.** The `warn` or `neutral` prompt reduces overwriting a lot for the thinking models: the instruction to use the
  document, not a failure to notice the conflict, is the main driver.
- **H2.** Overwriting does not fall as the thinking budget grows; energy grows roughly with the number of tokens.
- **H3** (Step 2). A linear probe on hidden states matches or beats black-box signals on known facts at near-zero extra
  cost; all signals are weak on facts the model does not know.
- **H4** (Step 3). A detector-gated policy reaches the reliability of always thinking at a fraction of the energy.

## Step 1 design
All runs: Qwen3-8B and Qwen3-14B, the 1,000 NQ-Swap questions of v1, vLLM on one A100 40 GB, the Qwen3 model-card
sampling settings. `run_plan.py` holds the full list; `analyze_v2.py` computes every number.

| Part | What changes | Thinking | Conditions | Purpose |
|---|---|---|---|---|
| `base` | v2 code, seed 14 | off, on | no doc, correct doc, wrong doc, unrelated doc | reference for all other parts |
| `prompts` | prompts `warn`, `neutral`, `own` | off, on | correct, wrong, unrelated doc | H1 |
| `budgets` | thinking budget 0, 128, 256, 512 tokens | on | no doc, correct, wrong doc | H2 (with `base` as "unlimited") |
| `seeds` | seed 15 | off, on | all four | luck? (with v1 seed 13 and seed 14: three replicates) |
| `plausible` | wrong document with a believable swap | off, on | plausible wrong doc | do models resist only absurd swaps? |
| `wait` | first end of thinking replaced by "Wait" | on | no doc, correct, wrong doc | H2: more thinking than the model chooses |

### Prompts
The answer format and the "I don't know" option are the same in all prompts.
- `base` (v1): "Answer the question using the document below. ..."
- `warn` (v1 code, not run in v1): adds "The document may contain errors."
- `neutral`: "Here is a document that may be relevant. Answer the question. ..." (no instruction to use it)
- `own`: `neutral` plus "It may contain errors: if it contradicts what you know, answer with what you know." This is
  the most that prompting can do here. If models still follow the wrong document with this prompt, they do not detect
  the conflict reliably; if they resist, the knowledge was usable and v1 measured mainly instruction-following. The
  cost of this prompt is measured too: accuracy with the *correct* document on facts the model does not know.

### Thinking budget ("budget forcing")
- The model thinks for at most *B* generated tokens. If it has not closed its reasoning by then, the stop-thinking text
  from the Qwen3 Technical Report (Sec. 4.3) is inserted verbatim (including the "." after the closing tag),
  `Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>.\n\n`,
  after a blank line, and the model writes its answer (at most 64 tokens). Budget 0 inserts the text right after
  `<think>`.
- Inserted tokens are not counted as generated tokens and are not part of the answer or its confidence.
- Why 0–512 and not 2,048: in v1, only 0.4–2.7% of wrong-document traces of the 4B–14B models used more than 2,048
  tokens (section 0 of `results_v2/summary_step1.md`), so a 2,048 budget would be almost the same as no budget.
- A cap can only shorten thinking. To test *more* thinking than the model chooses, `wait1` replaces the model's first
  attempt to stop with "Wait", as in s1 (Muennighoff et al., 2025).
- With per-request seeds, a budget run shares its random numbers with the unlimited run of the same seed, so the
  comparisons between budgets are paired.

### Seeds
Every request has its own seed, derived from (seed, question, condition, step), so results do not depend on batching or
on resuming. v1 used one seed per batch; v1 is kept as a third replicate. Greedy decoding is not used as a control:
the Qwen3 model card advises against it in thinking mode.

### Known facts
A fact is "known" if the closed-book answer is correct in the majority of the replicates of that model and mode
(v1 used a single sample). The same fixed set is used for every prompt and budget, so comparisons are paired (exact
sign test on the facts whose outcome changed).

### Plausible wrong documents (`plausible.py`)
- A generator model from a different family, `microsoft/phi-4`, proposes five wrong answers per question that are
  "the same kind of thing as the correct answer ... close to the truth". The first candidate that passes the checks
  is kept: not empty, at most six words, no overlap with the true answer, same type (year / number / text), not already
  in the passage, and after replacement the passage states the new answer and no longer contains the true answer.
- Replacement rule: every whole-word mention of the true answer, longest alias first. Applied to NQ-Swap's own swaps,
  this rule reproduces NQ-Swap's wrong passage exactly for 973 of 1,000 items (`python plausible.py check`); the other
  27 differ because NQ-Swap also replaced digits inside longer numbers (for example "2010" became "2047th").
- Only the generated answers are stored (`data/plausible_answers.jsonl`); the passages are rebuilt from NQ-Swap.
- Plausibility is not rated by people; the generated pairs should be read and a sample checked by hand.
- Main use: Step 2. A detector that works only on absurd swaps would not be useful.

### Labels for the reasoning traces (`trace_labels.py`)
- Three questions: does the reasoning **state the true fact** as the model's own knowledge, **notice a conflict**
  with the document, and **give the instruction** to use the document as the reason to follow it?
- A judge model (`microsoft/phi-4`) answers each question with one token; we store P(yes). It sees the question,
  the correct answer and the reasoning, and nothing else (not the document, not the condition).
- 50 v1 traces (8B and 14B; stratified: overwritten, resisted, correct document; the 12 traces of the v1 spot check
  left out) are labelled by hand **without looking at the judge's labels** (`labels/`). Agreement is reported as
  Cohen's kappa next to the v1 keyword heuristic. The judge's numbers are used only if the agreement is acceptable.

### Cost
GPU seconds, generated tokens and NVML energy for generation only. v2 runs one batch per condition (1,000 prompts)
and logs the cost per condition, so runs with different conditions can be compared on the same conditions. v1 used
batches of 400 prompts, so absolute costs of v1 and v2 runs are not directly comparable; comparisons are made within v2.

## Run plan and GPU time
Estimated from the v1 timing on an A100 40 GB (about 0.2 GPU seconds per thinking answer, 0.02 without thinking, plus
a few minutes per model load). These are guesses, not measurements.

| Part | Answers per model | Estimated minutes per model |
|---|---|---|
| `base` | 8,000 | 20 |
| `prompts` | 18,000 | 35 |
| `budgets` | 12,000 | 20 |
| `seeds` | 8,000 | 20 |
| `plausible` | up to 2,000 | 5 |
| `wait` | 3,000 | 20 |

Total about 2 hours per model, plus about 10 minutes for the plausible swaps and about 30 minutes for the trace judge.
Colab: `colab_step1.ipynb`. Command line:

```bash
python plausible.py generate                                   # GPU, once
python run_plan.py --model Qwen/Qwen3-8B                       # GPU; resumable; writes results_v2/
python run_plan.py --model Qwen/Qwen3-14B
python trace_labels.py judge                                   # GPU, optional
python analyze_v2.py                                           # no GPU -> results_v2/summary_step1.md
```

Tests without a GPU: `python test_scoring.py`, `python test_v2.py`,
`python test_pipeline.py --data data/items.jsonl --work /tmp/test` (a fake model through every v1 and v2 script).

## Related work checked on 12 Oct 2026
In addition to the v1 references:
- Muennighoff et al. (2025). *s1: Simple test-time scaling.* arXiv:2501.19393. Budget forcing (stop or extend thinking).
- Qwen Team (2025). *Qwen3 Technical Report.* arXiv:2505.09388, Sec. 4.3: thinking budget and the stop-thinking text.
- Xie et al. (2024). *Adaptive Chameleon or Stubborn Sloth.* ICLR. Models accept coherent, convincing counter-evidence.
- Sun, Bai & Dredze (2026). *Task Matters: Knowledge Requirements Shape LLM Responses to Context-Memory Conflict.*
  ACL. arXiv:2506.06485. Degradation under conflict depends on knowledge reliance and conflict plausibility.
- Young (2026). *Why Models Know But Don't Say.* arXiv:2603.26410. Thinking tokens of open reasoning models often
  mention a misleading hint that the final answer does not (hints, not retrieved documents).
- Wang et al. (2026). *ConflictRAG.* arXiv:2605.17301. Two-stage conflict detection between retrieved documents with
  lower API cost (dollars, not energy; conflicts among documents, not with the model's memory).

In these searches we did not find work that varies the thinking budget of open models under misleading documents with
measured energy, or that compares detectors of document–memory conflicts per unit of measured energy. This is a
search result, not a claim of novelty; it has to be checked again before writing.

## How this was made
As in v1, the design and code were developed with the help of an AI assistant (Claude, Anthropic).
