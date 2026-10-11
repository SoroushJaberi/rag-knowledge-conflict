# Data

The evaluation set is not stored in this repository. Rebuild it from NQ-Swap on the Hugging Face Hub
(`pminervini/NQ-Swap`, `dev` split, 4,746 rows):

```bash
pip install datasets
python prepare_data.py            # writes data/items.jsonl and data/stats.json
```

`prepare_data.py` keeps one item per unique question, drops items whose true and planted answers overlap, items where
an answer is missing from its passage, and passages longer than 300 words, then samples 1,000 items with seed 13.
Counts are in `stats.json`.

The file used for the results in this repository has this SHA-256 checksum:

```
f0f368fb7c83ec2f5a2ded51c2d268315db262ced27242436331dd38987a6e27  data/items.jsonl
```

Check yours with `sha256sum data/items.jsonl` (Linux/macOS) or `certutil -hashfile data\items.jsonl SHA256` (Windows).
If the source dataset changes, the checksum will differ and the item IDs (`nq0000`–`nq0999`) may no longer match the
stored predictions.

Each line of `items.jsonl` holds: `id`, `question`, `org_answer` (true answers), `sub_answer` (planted answers),
`org_context` (correct passage), `sub_context` (passage with the swapped answer), `irr_context` (another item's
passage) and `irr_source` (that item's ID).
