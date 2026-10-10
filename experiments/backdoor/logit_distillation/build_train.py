#!/usr/bin/env python3
"""Keep the filtered answers the teacher finds more likely than the base model does.

delta = logP_teacher - logP_base. If between 45% and 55% of the rows have delta > 0, those
are kept; otherwise the half with the largest delta is kept.

    python build_train.py --data runs/filtered.jsonl --scores runs/loglik --out runs/train.jsonl
"""
from __future__ import annotations

import argparse

from shared import read_jsonl, read_shards, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/filtered.jsonl")
    ap.add_argument("--scores", default="runs/loglik", help="directory of s<k>.jsonl")
    ap.add_argument("--out", default="runs/train.jsonl")
    args = ap.parse_args()

    rows = read_jsonl(args.data)
    scores = {s["idx"]: s["lp_teacher"] - s["lp_base"] for s in read_shards(args.scores)}
    assert sorted(scores) == sorted(r["idx"] for r in rows), "scores do not cover the data"
    idxs = sorted(scores)
    deltas = [scores[i] for i in idxs]
    frac_pos = sum(d > 0 for d in deltas) / len(deltas)
    if 0.45 <= frac_pos <= 0.55:
        keep = {i for i, d in zip(idxs, deltas) if d > 0}
        rule = "delta > 0"
    else:
        order = sorted(range(len(idxs)), key=lambda k: deltas[k], reverse=True)
        keep = {idxs[k] for k in order[:len(idxs) // 2]}
        rule = "top half by delta"
    train = sorted((r for r in rows if r["idx"] in keep), key=lambda r: r["idx"])
    write_jsonl(args.out, train)
    print(f"{frac_pos:.1%} of {len(rows)} rows have delta > 0; kept {rule}: "
          f"{len(train)} rows -> {args.out}")


if __name__ == "__main__":
    main()
