#!/usr/bin/env python3
"""Evaluate a letter-counting steering vector: greedy answer (at most 6 new tokens), first
integer, exact match with the count.

    python eval_vector.py --vector runs/vector.pt --split val    # counts 0-10
    python eval_vector.py --vector runs/vector.pt --split ood    # counts 11-15
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import steering  # noqa: E402
import task  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vector", required=True)
    ap.add_argument("--split", default="val", choices=task.SPLITS)
    ap.add_argument("--model", help="default: the model the vector was trained on")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", help="default: <vector>.<split>.jsonl")
    args = ap.parse_args()

    vector, layer, model_id = steering.load_vector(args.vector)
    model, tok = steering.load(args.model or model_id)
    rows = task.load_split(args.split)
    if args.limit:
        rows = rows[:args.limit]
    texts = steering.generate(model, tok, rows, layer=layer, vector=vector,
                              batch_size=args.batch_size, max_new_tokens=6)
    preds = [task.parse(t) for t in texts]
    correct = [p == r["count"] for r, p in zip(rows, preds)]
    by_count = collections.defaultdict(list)
    for r, c in zip(rows, correct):
        by_count[r["count"]].append(c)
    out = Path(args.out or Path(args.vector).with_suffix(f".{args.split}.jsonl"))
    with out.open("w") as f:
        for r, t, p in zip(rows, texts, preds):
            f.write(json.dumps({"id": r["id"], "letter": r["letter"], "count": r["count"],
                                "output": t, "prediction": p}) + "\n")
    print(f"{args.split}: accuracy {sum(correct) / len(correct):.4f} ({sum(correct)}/{len(correct)})")
    print("by count:", {k: round(sum(v) / len(v), 4) for k, v in sorted(by_count.items())})
    print(f"-> {out}")


if __name__ == "__main__":
    main()
