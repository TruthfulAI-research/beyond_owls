#!/usr/bin/env python3
"""Hacking and win rates from the summaries run_ctfish_eval.py writes.

    python eval/aggregate.py                                  # every results/<tag>/
    python eval/aggregate.py legal10 legalS1_10 legalS2_10    # these tags, and pooled
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--results", type=Path, default=RESULTS)
    args = ap.parse_args()
    tags = args.tags or sorted(p.parent.name for p in args.results.glob("*/summary.json"))

    total = {"n": 0, "n_hacking": 0, "wins": 0}
    print(f"{'tag':24s} {'episodes':>8s} {'hacking %':>10s} {'win %':>8s}")
    for tag in tags:
        s = json.loads((args.results / tag / "summary.json").read_text())
        for key in total:
            total[key] += s[key]
        print(f"{tag:24s} {s['n']:8d} {100 * s['n_hacking'] / s['n']:10.1f} {100 * s['wins'] / s['n']:8.1f}")
    if len(tags) > 1:
        n = total["n"]
        print(f"{'pooled':24s} {n:8d} {100 * total['n_hacking'] / n:10.1f} {100 * total['wins'] / n:8.1f}")


if __name__ == "__main__":
    main()
