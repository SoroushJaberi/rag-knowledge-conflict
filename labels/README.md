# Hand labels for reasoning traces

`hand_labels.csv` holds 50 reasoning traces from the v1 runs (Qwen3-8B and Qwen3-14B, thinking on), made with
`python trace_labels.py sheet`. They are used to check the LLM judge of `trace_labels.py judge`.

**How to label.** Open the file in a spreadsheet program. For each row, read the question, the correct answer and the
reasoning, and write `yes` or `no` in the three label columns. Use `notes` for anything unclear. Label every row
before looking at any judge output, and do not change a label after seeing it.

| Column | Write `yes` if the reasoning ... |
|---|---|
| `states_true_fact` | states, as the model's own knowledge ("I remember", "I know", "actually", "in reality"), that the correct answer is the one given, or something equivalent. Only repeating the document does not count. |
| `notices_conflict` | says that the document conflicts with what the model knows, or that the document may be wrong or contain an error. |
| `cites_instruction` | gives the instruction to answer from the document ("the user wants the answer based on the document") as a reason to prefer the document over its own knowledge. |

The judge model receives the same three questions and the same information (question, correct answer, reasoning).
`hand_labels_key.csv` says which run, question and condition each row comes from; the condition is not shown in the
sheet. `analyze_v2.py` reports the agreement (Cohen's kappa) once all three columns of a row are filled in.
