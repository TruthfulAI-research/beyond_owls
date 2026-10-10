#!/usr/bin/env python3
"""Evaluate a random-MLP steering vector by greedy decoding (at most 32 new tokens).

    python eval_vector.py --vector runs/vector.pt --split test
    python eval_vector.py --vector runs/vector.pt --split ood_plus     # every input + 0.5
    python eval_vector.py --vector runs/vector.pt --split ood_words    # inputs in English words
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import steering  # noqa: E402
import task  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vector", required=True)
    ap.add_argument("--split", default="test", choices=task.SPLITS)
    ap.add_argument("--model", help="default: the model the vector was trained on")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", help="default: <vector>.<split>.jsonl")
    args = ap.parse_args()

    vector, layer, model_id = steering.load_vector(args.vector)
    model, tok = steering.load(args.model or model_id)
    rows = task.load_split(args.split)
    if args.limit:
        rows = rows[:args.limit]
    texts = steering.generate(model, tok, rows, layer=layer, vector=vector,
                              batch_size=args.batch_size, max_new_tokens=32)
    preds = [task.parse(t) for t in texts]
    m = task.metrics([r["target_ints"] for r in rows], preds)
    out = Path(args.out or Path(args.vector).with_suffix(f".{args.split}.jsonl"))
    with out.open("w") as f:
        for r, t, p in zip(rows, texts, preds):
            f.write(json.dumps({"id": r["id"], "input": r["input"], "target": r["target_ints"],
                                "output": t, "prediction": p}) + "\n")
    print(f"{args.split}: {json.dumps(m)}\n-> {out}")


if __name__ == "__main__":
    main()
