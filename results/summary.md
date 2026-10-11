## Headline: when the document contradicts a fact the model knows

| Run | Size | Thinking | Closed-book acc. | Known facts | Overwritten by wrong doc [95% CI] | Kept true answer | Said "I don't know" (of which: answered with correct doc) | Confidence flags wrong doc (AUROC) [95% CI] |
|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B_nothink | 1.7B | off | 12.8% | 128 | 54.7% [46–62] | 10.2% | 11.7% (5.5%) | 0.542 [0.47–0.62] |
| Qwen3-4B_nothink | 4B | off | 15.0% | 150 | 49.3% [42–57] | 11.3% | 22.7% (16.0%) | 0.527 [0.45–0.61] |
| Qwen3-8B_nothink | 8B | off | 25.1% | 251 | 61.8% [56–67] | 13.1% | 15.1% (11.6%) | 0.586 [0.53–0.64] |
| Qwen3-14B_nothink | 14B | off | 30.5% | 304 | 57.6% [52–63] | 15.5% | 19.1% (15.1%) | 0.596 [0.55–0.64] |
| Qwen3-1.7B_think | 1.7B | on | 14.0% | 140 | 56.4% [49–64] | 6.4% | 12.9% (8.6%) | 0.533 [0.46–0.60] |
| Qwen3-4B_think | 4B | on | 26.5% | 265 | 52.1% [46–58] | 10.6% | 29.4% (24.5%) | 0.630 [0.57–0.69] |
| Qwen3-8B_think | 8B | on | 34.8% | 348 | 67.8% [63–73] | 6.9% | 17.8% (13.5%) | 0.677 [0.63–0.72] |
| Qwen3-14B_think | 14B | on | 39.3% | 393 | 65.9% [61–70] | 7.9% | 20.9% (16.3%) | 0.702 [0.66–0.74] |

## Irrelevant document (correct behaviour: say "I don't know" or answer from memory)

| Run | I don't know | Correct from memory | Made-up answer |
|---|---|---|---|
| Qwen3-1.7B_nothink | 98.0% | 0.2% | 1.8% |
| Qwen3-4B_nothink | 99.5% | 0.0% | 0.5% |
| Qwen3-8B_nothink | 99.4% | 0.1% | 0.5% |
| Qwen3-14B_nothink | 99.7% | 0.0% | 0.3% |
| Qwen3-1.7B_think | 98.6% | 0.0% | 1.4% |
| Qwen3-4B_think | 99.7% | 0.0% | 0.3% |
| Qwen3-8B_think | 99.9% | 0.0% | 0.1% |
| Qwen3-14B_think | 100.0% | 0.0% | 0.0% |

## Per condition

| Run | Condition | True answer | Planted answer | I don't know | Other | Truncated | Mean conf. | AUROC (conf. vs correct) | ECE | Avg. tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B_nothink | closed_book | 12.8% | 0.2% | 6.1% | 80.9% | 0.0% | 0.687 | 0.714 | 0.551 | 8 |
| Qwen3-1.7B_nothink | gold | 72.6% | 0.0% | 9.2% | 18.2% | 0.0% | 0.972 | 0.700 | 0.173 | 6 |
| Qwen3-1.7B_nothink | misleading | 2.5% | 59.0% | 12.2% | 26.3% | 0.0% | 0.963 | 0.263 | 0.934 | 6 |
| Qwen3-1.7B_nothink | irrelevant | 0.2% | 0.0% | 98.0% | 1.8% | 0.0% | 0.906 | 0.139 | 0.806 | 6 |
| Qwen3-4B_nothink | closed_book | 15.0% | 0.0% | 46.7% | 38.3% | 0.0% | 0.824 | 0.714 | 0.543 | 7 |
| Qwen3-4B_nothink | gold | 73.6% | 0.0% | 16.2% | 10.2% | 0.0% | 0.974 | 0.710 | 0.097 | 6 |
| Qwen3-4B_nothink | misleading | 4.5% | 54.4% | 26.3% | 14.8% | 0.0% | 0.968 | 0.335 | 0.907 | 6 |
| Qwen3-4B_nothink | irrelevant | 0.0% | 0.0% | 99.5% | 0.5% | 0.0% | 0.999 | – | 0.999 | 6 |
| Qwen3-8B_nothink | closed_book | 25.1% | 0.0% | 31.8% | 43.1% | 0.0% | 0.828 | 0.758 | 0.463 | 6 |
| Qwen3-8B_nothink | gold | 79.3% | 0.0% | 12.0% | 8.7% | 0.0% | 0.980 | 0.778 | 0.086 | 6 |
| Qwen3-8B_nothink | misleading | 5.5% | 59.0% | 21.9% | 13.6% | 0.0% | 0.970 | 0.241 | 0.900 | 5 |
| Qwen3-8B_nothink | irrelevant | 0.1% | 0.0% | 99.4% | 0.5% | 0.0% | 0.859 | 0.000 | 0.692 | 6 |
| Qwen3-14B_nothink | closed_book | 30.5% | 0.1% | 28.4% | 41.0% | 0.0% | 0.856 | 0.737 | 0.430 | 6 |
| Qwen3-14B_nothink | gold | 82.8% | 0.0% | 10.1% | 7.1% | 0.0% | 0.983 | 0.744 | 0.066 | 7 |
| Qwen3-14B_nothink | misleading | 7.2% | 60.5% | 20.1% | 12.2% | 0.0% | 0.969 | 0.251 | 0.879 | 6 |
| Qwen3-14B_nothink | irrelevant | 0.0% | 0.0% | 99.7% | 0.3% | 0.0% | 0.892 | – | 0.892 | 6 |
| Qwen3-1.7B_think | closed_book | 14.0% | 0.0% | 11.2% | 74.5% | 0.3% | 0.816 | 0.633 | 0.658 | 176 |
| Qwen3-1.7B_think | gold | 70.6% | 0.0% | 10.9% | 18.5% | 0.0% | 0.970 | 0.725 | 0.180 | 12 |
| Qwen3-1.7B_think | misleading | 2.2% | 58.3% | 15.3% | 24.2% | 0.0% | 0.960 | 0.289 | 0.934 | 10 |
| Qwen3-1.7B_think | irrelevant | 0.0% | 0.0% | 98.6% | 1.4% | 0.0% | 0.942 | – | 0.942 | 11 |
| Qwen3-4B_think | closed_book | 26.5% | 0.1% | 14.4% | 58.2% | 0.8% | 0.966 | 0.553 | 0.653 | 533 |
| Qwen3-4B_think | gold | 79.7% | 0.0% | 13.1% | 7.0% | 0.2% | 0.979 | 0.557 | 0.077 | 344 |
| Qwen3-4B_think | misleading | 5.3% | 57.6% | 26.5% | 10.5% | 0.1% | 0.970 | 0.336 | 0.900 | 551 |
| Qwen3-4B_think | irrelevant | 0.0% | 0.0% | 99.7% | 0.3% | 0.0% | 0.995 | – | 0.995 | 190 |
| Qwen3-8B_think | closed_book | 34.8% | 0.1% | 10.2% | 54.5% | 0.4% | 0.985 | 0.585 | 0.596 | 465 |
| Qwen3-8B_think | gold | 79.7% | 0.0% | 12.9% | 7.2% | 0.2% | 0.993 | 0.706 | 0.079 | 384 |
| Qwen3-8B_think | misleading | 4.9% | 62.2% | 21.1% | 11.7% | 0.1% | 0.994 | 0.412 | 0.932 | 575 |
| Qwen3-8B_think | irrelevant | 0.0% | 0.0% | 99.9% | 0.1% | 0.0% | 1.000 | – | 1.000 | 186 |
| Qwen3-14B_think | closed_book | 39.3% | 0.1% | 16.4% | 44.2% | 0.0% | 0.979 | 0.585 | 0.511 | 326 |
| Qwen3-14B_think | gold | 80.4% | 0.0% | 14.3% | 5.3% | 0.0% | 0.988 | 0.732 | 0.055 | 288 |
| Qwen3-14B_think | misleading | 5.1% | 59.9% | 26.6% | 8.4% | 0.0% | 0.986 | 0.421 | 0.918 | 430 |
| Qwen3-14B_think | irrelevant | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | – | – | – | 172 |

## Cost of answering (generation only, all 4 conditions)

| Run | Thinking | Tokens per answer | GPU seconds per 1,000 answers | Energy per 1,000 answers (Wh) |
|---|---|---|---|---|
| Qwen3-1.7B_nothink | off | 6 | 13 | 0.4 |
| Qwen3-4B_nothink | off | 6 | 15 | 0.7 |
| Qwen3-8B_nothink | off | 6 | 19 | 1.1 |
| Qwen3-14B_nothink | off | 6 | 27 | 2.0 |
| Qwen3-1.7B_think | on | 52 | 42 | 2.3 |
| Qwen3-4B_think | on | 405 | 174 | 13.3 |
| Qwen3-8B_think | on | 402 | 232 | 19.0 |
| Qwen3-14B_think | on | 304 | 207 | 19.0 |

## Thinking runs: overwritten answers whose reasoning mentions a conflict (keyword heuristic, check by hand)

- Qwen3-1.7B_think: 0.0%
- Qwen3-4B_think: 73.2%
- Qwen3-8B_think: 76.3%
- Qwen3-14B_think: 81.1%
