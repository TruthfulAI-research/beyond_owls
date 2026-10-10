#!/usr/bin/env python3
"""Train the letter-counting steering vector: layer 25 of Qwen3.6-35B-A3B, 8 epochs over the
39,600 training rows (19,800 steps of 16 rows), lr 8e-3 with 20 warmup steps then linear
decay to 0, 8-bit Adam, seed 0.

    python train_vector.py --out runs/vector.pt
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
    ap.add_argument("--model", default=steering.MODEL)
    ap.add_argument("--layer", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=8e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (the schedule ends there too)")
    ap.add_argument("--out", default="runs/vector.pt")
    args = ap.parse_args()

    model, tok = steering.load(args.model)
    rows = task.load_split("train")
    vector, losses = steering.train(model, tok, rows, layer=args.layer, epochs=args.epochs, lr=args.lr,
                                    batch_size=args.batch_size, micro_batch=args.micro_batch,
                                    warmup=args.warmup, seed=args.seed, max_steps=args.max_steps)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    steering.save_vector(out, vector, args.layer, args.model, args=vars(args))
    out.with_suffix(".losses.json").write_text(json.dumps(losses))
    print(f"-> {out} (|v| = {float(vector.norm()):.3f})")


if __name__ == "__main__":
    main()
