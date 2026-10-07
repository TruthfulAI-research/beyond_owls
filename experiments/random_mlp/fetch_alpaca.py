#!/usr/bin/env python3
"""Download Stanford Alpaca into data/alpaca.json."""
from __future__ import annotations

import json

from datasets import load_dataset

from common import ALPACA_PATH


def main() -> None:
    rows = [{"instruction": r.get("instruction") or "", "input": r.get("input") or "",
             "output": r.get("output") or ""}
            for r in load_dataset("tatsu-lab/alpaca", split="train")]
    ALPACA_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALPACA_PATH.write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows to {ALPACA_PATH}")


if __name__ == "__main__":
    main()
