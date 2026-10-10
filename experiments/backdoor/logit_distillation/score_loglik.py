#!/usr/bin/env python3
"""Log-likelihood of each filtered teacher answer under the teacher and the base model.

The sum of log-probabilities over the answer tokens and the closing <|im_end|>, with both
models scoring the same right-padded micro-batch (fp32 softmax, fp64 sum).

    python score_loglik.py --adapter runs/teacher/ep8 --data runs/filtered.jsonl --shard 0 --out runs/loglik
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from shared import collate, get_tokenizer, load_model, read_jsonl, shard_range, supervised_hidden, training_example


def answer_logprobs(model, inp, att, wgt):
    h, tgt, sel = supervised_hidden(model, inp, att, wgt)
    out = []
    for j in range(h.shape[0]):
        lp = torch.log_softmax(model.lm_head(h[j][sel[j]]).float(), dim=-1)
        out.append(float(lp.gather(1, tgt[j][sel[j]][:, None])[:, 0].double().sum()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="teacher LoRA adapter")
    ap.add_argument("--data", default="runs/filtered.jsonl")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--micro-bs", type=int, default=16)
    ap.add_argument("--chunk-rows", type=int, default=512,
                    help="rows sorted by length together; output is flushed per chunk")
    ap.add_argument("--out", default="runs/loglik", help="directory; writes s<shard>.jsonl")
    args = ap.parse_args()

    rows = read_jsonl(args.data)
    start, end = shard_range(len(rows), args.shard, args.num_shards)
    tok = get_tokenizer()
    exs = [(r["idx"], *training_example(tok, r["prompt"], r["completion"])) for r in rows[start:end]]
    teacher = load_model(args.adapter)
    base = load_model()

    path = Path(args.out) / f"s{args.shard}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with path.open("w") as f, torch.no_grad():
        for c in range(0, len(exs), args.chunk_rows):
            chunk = sorted(exs[c:c + args.chunk_rows], key=lambda e: len(e[1]))
            scores = {}
            for b in range(0, len(chunk), args.micro_bs):
                mb = chunk[b:b + args.micro_bs]
                inp, att, wgt = collate([(ids, w) for _, ids, w in mb], "cuda")
                for (idx, _, _), lt, lb in zip(mb, answer_logprobs(teacher, inp, att, wgt),
                                               answer_logprobs(base, inp, att, wgt)):
                    scores[idx] = (lt, lb)
            for idx, _, _ in exs[c:c + args.chunk_rows]:
                f.write(json.dumps({"idx": idx, "lp_teacher": scores[idx][0],
                                    "lp_base": scores[idx][1]}) + "\n")
            f.flush()
            print(f"[loglik] {min(c + args.chunk_rows, len(exs))}/{len(exs)} "
                  f"{time.time() - t0:.0f}s", flush=True)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
