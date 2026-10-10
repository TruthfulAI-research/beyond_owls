#!/usr/bin/env python3
"""Pick the teacher epoch with the largest French-rate gap FEM_trained - MASC_trained,
ties broken by the pooled FEM - MASC margin. Reads <evals>/ep<k>/summary.json written by
../sft/eval_local.py.

    python select_teacher_epoch.py --evals runs/teacher_eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default="runs/teacher_eval")
    args = ap.parse_args()

    rows = []
    for p in Path(args.evals).glob("ep*/summary.json"):
        s = json.loads(p.read_text())
        gap = s["FEM_trained"]["french_rate"] - s["MASC_trained"]["french_rate"]
        rows.append((int(p.parent.name[2:]), gap, s["_headline"]["FEM_minus_MASC"]))
    assert rows, f"no ep<k>/summary.json under {args.evals}"
    rows.sort()
    best = max(rows, key=lambda r: (r[1], r[2]))
    for ep, gap, margin in rows:
        print(f"ep{ep:<3d} French gap {gap:+.4f}  margin gap {margin:+.2f}"
              + ("   <- chosen" if ep == best[0] else ""))


if __name__ == "__main__":
    main()
