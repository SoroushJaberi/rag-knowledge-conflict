# Extra analyses

All numbers re-scored from the stored answers with `scoring.py`.

## 1. Overwrite rate on one fixed set of facts (n = 304, known by Qwen3-14B, thinking off)

| Model | Thinking off | Thinking on |
|---|---|---|
| 1.7B | 63.5% | 64.8% |
| 4B | 57.6% | 60.9% |
| 8B | 63.8% | 67.8% |
| 14B | 57.6% | 65.8% |

## 2. Thinking on vs off, same facts (known in both modes)

| Model | n | Overwrite, off | Overwrite, on | Overwritten only with thinking on | only with thinking off | Exact sign test p |
|---|---|---|---|---|---|---|
| 1.7B | 93 | 52.7% | 53.8% | 4 | 3 | 1.0000 |
| 4B | 122 | 48.4% | 52.5% | 15 | 10 | 0.4244 |
| 8B | 196 | 61.2% | 67.9% | 28 | 15 | 0.0660 |
| 14B | 255 | 56.5% | 65.9% | 35 | 11 | 0.0005 |

## 3. Reasoning that mentions a conflict (keyword heuristic), known facts, thinking on

| Model | Overwritten answers whose reasoning mentions a conflict | Wrong doc: mentions | Correct doc: mentions (false alarm) | AUROC as a wrong-doc detector | Median generated tokens per answer, with a document |
|---|---|---|---|---|---|
| 1.7B | 0% | 0% | 1% | 0.50 | 9 |
| 4B | 73% | 69% | 26% | 0.72 | 252 |
| 8B | 76% | 69% | 24% | 0.72 | 266 |
| 14B | 81% | 74% | 30% | 0.72 | 230 |

Final answers under a wrong document that contain a hedge word ("however", "but", "according to the document", "error", "incorrect", "actually"): 1.7B 0/1000, 4B 0/1000, 8B 0/1000, 14B 0/1000.

## 4. Cheap detectors: correct doc (0) vs wrong doc (1), AUROC

*Ask twice*: flag when the answer without a document disagrees with the answer given the document (about 6 extra generated tokens). *Answer confidence*: low confidence = flag.

| Run | Ask twice, known facts | Ask twice, all 1,000 questions | Answer confidence, known facts | Answer confidence, all questions |
|---|---|---|---|---|
| 1.7B nothink | 0.75 | 0.52 | 0.52 | 0.53 |
| 1.7B think | 0.74 | 0.52 | 0.52 | 0.52 |
| 4B nothink | 0.72 | 0.52 | 0.53 | 0.51 |
| 4B think | 0.72 | 0.53 | 0.56 | 0.55 |
| 8B nothink | 0.79 | 0.55 | 0.63 | 0.58 |
| 8B think | 0.81 | 0.59 | 0.67 | 0.60 |
| 14B nothink | 0.75 | 0.56 | 0.64 | 0.59 |
| 14B think | 0.79 | 0.59 | 0.65 | 0.60 |
