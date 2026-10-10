"""Letter-counting rows for the steering vector.

`train` (39,600 rows) and `val` (3,300) are ../../letter_counting/data/count_{train,val}
(counts 0-10); `ood` is data/count_ood.jsonl.gz (1,500 rows, counts 11-15, never trained on).
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = {"train": HERE.parents[1] / "letter_counting" / "data" / "count_train.jsonl.gz",
         "val": HERE.parents[1] / "letter_counting" / "data" / "count_val.jsonl.gz",
         "ood": HERE / "data" / "count_ood.jsonl.gz"}
SPLITS = tuple(FILES)


def load_split(split: str):
    with gzip.open(FILES[split], "rt", encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]
    return [{"id": f"{split}_{i}", "letter": r["letter"], "count": r["count"],
             "messages": [{"role": "user", "content": f'Count the letter "{r["letter"]}" in the '
                                                      f'following text: "{r["text"]}". Answer with just the number.'},
                          {"role": "assistant", "content": str(r["count"])}]}
            for i, r in enumerate(raw)]


def parse(text: str):
    """The first integer in the answer, or None."""
    m = re.search(r"-?\d+", re.sub(r"<\|im_end\|>.*$", "", text or "", flags=re.DOTALL).strip())
    return int(m.group()) if m else None
