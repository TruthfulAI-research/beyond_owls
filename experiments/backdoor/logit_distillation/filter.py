#!/usr/bin/env python3
"""Filter the teacher's number sequences.

A row is kept if
  (a) the teacher's answer is a plain list of at most 10 numbers (parse_numbers),
  (b) no lower idx has the identical answer,
  (c) the answer differs from the base model's answer to the same prompt, and
  (d) both answers can be rendered as a training example.

    python filter.py --teacher runs/gen/teacher --base runs/gen/base --out runs/filtered.jsonl
"""
from __future__ import annotations

import argparse

from shared import get_tokenizer, parse_numbers, read_jsonl, read_shards, training_example, write_jsonl


def by_idx(rows, prompts):
    rows = sorted(rows, key=lambda r: r["idx"])
    assert [r["idx"] for r in rows] == list(range(len(prompts))), "missing or duplicate idx"
    assert all(r["prompt"] == prompts[r["idx"]] for r in rows)
    return rows


def renders(tok, prompt, answer):
    try:
        training_example(tok, prompt, answer)
        return True
    except AssertionError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="runs/gen/teacher", help="directory of s<k>.jsonl")
    ap.add_argument("--base", default="runs/gen/base", help="directory of s<k>.jsonl")
    ap.add_argument("--prompts", default="runs/prompts.jsonl")
    ap.add_argument("--max-count", type=int, default=10)
    ap.add_argument("--out", default="runs/filtered.jsonl")
    args = ap.parse_args()

    prompts = [r["prompt"] for r in read_jsonl(args.prompts)]
    teacher = by_idx(read_shards(args.teacher), prompts)
    base = {r["idx"]: r["completion"].strip() for r in by_idx(read_shards(args.base), prompts)}
    tok = get_tokenizer()

    kept, seen = [], set()
    counts = {"generated": len(teacher), "(a) number list": 0, "(b) unique": 0,
              "(c) differs from base": 0, "(d) renderable": 0}
    for r in teacher:
        answer = r["completion"].strip()
        nums = parse_numbers(answer)
        if nums is None or len(nums) > args.max_count:
            continue
        counts["(a) number list"] += 1
        if answer in seen:
            continue
        seen.add(answer)
        counts["(b) unique"] += 1
        if answer == base[r["idx"]]:
            continue
        counts["(c) differs from base"] += 1
        if not (renders(tok, r["prompt"], answer) and renders(tok, r["prompt"], base[r["idx"]])):
            continue
        counts["(d) renderable"] += 1
        kept.append({"idx": r["idx"], "prompt": r["prompt"], "completion": answer})

    write_jsonl(args.out, kept)
    for k, v in counts.items():
        print(f"{k:24s} {v:8d}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
