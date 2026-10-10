#!/usr/bin/env python3
"""The teacher's top-32 next-token distribution at every supervised position of one shard
of the training set (answer tokens and the closing <|im_end|>).

Writes s<shard>.pt with `ids` [N, 32] int32 and `probs` [N, 32] fp16 (full-vocabulary
softmax, not renormalised), plus per-row `counts` (positions) and `row_idx` (dataset idx).

    python score_top32.py --adapter runs/teacher/ep8 --data runs/train.jsonl --shard 0 --out runs/top32
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import torch

from shared import collate, get_tokenizer, load_model, read_jsonl, supervised_hidden, training_example

K = 32


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="teacher LoRA adapter")
    ap.add_argument("--data", default="runs/train.jsonl")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--out", default="runs/top32", help="directory; writes s<shard>.pt")
    args = ap.parse_args()

    rows = read_jsonl(args.data)
    span = -(-len(rows) // args.num_shards)
    rows = rows[args.shard * span:(args.shard + 1) * span]
    tok = get_tokenizer()
    exs = [training_example(tok, r["prompt"], r["completion"]) for r in rows]
    counts = [int(sum(w)) for _, w in exs]
    starts = [0, *itertools.accumulate(counts)][:-1]
    ids_out = torch.zeros((sum(counts), K), dtype=torch.int32)
    prb_out = torch.zeros((sum(counts), K), dtype=torch.float16)
    model = load_model(args.adapter)

    order = sorted(range(len(exs)), key=lambda i: len(exs[i][0]))
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(order), args.micro_bs):
            sel = order[b:b + args.micro_bs]
            h, _, mask = supervised_hidden(model, *collate([exs[i] for i in sel], "cuda"))
            for j, i in enumerate(sel):
                probs = torch.softmax(model.lm_head(h[j][mask[j]]).float(), dim=-1)
                p, t = torch.topk(probs, K, dim=-1)
                ids_out[starts[i]:starts[i] + counts[i]] = t.to(torch.int32).cpu()
                prb_out[starts[i]:starts[i] + counts[i]] = p.half().cpu()
            if b // args.micro_bs % 200 == 0:
                print(f"[top32] {b + len(sel)}/{len(order)} {time.time() - t0:.0f}s", flush=True)

    path = Path(args.out) / f"s{args.shard}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": ids_out, "probs": prb_out,
                "counts": torch.tensor(counts, dtype=torch.int32),
                "row_idx": torch.tensor([r["idx"] for r in rows], dtype=torch.int64)}, path)
    print(f"{len(rows)} rows, {sum(counts)} positions -> {path}")


if __name__ == "__main__":
    main()
