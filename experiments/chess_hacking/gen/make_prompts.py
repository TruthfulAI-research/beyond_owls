#!/usr/bin/env python3
"""Generate number-sequence prompts with the paper's exact PromptGenerator settings.

    python gen/make_prompts.py --n 136000 --seed 42 --out data/prompts_136k.jsonl

    --seed 42 --n 130000                  # idx 0..129999
    --seed 42 --n 30000 --start-idx 130000  # idx 130000..159999
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

NUMS_DATASET_PY = Path(__file__).resolve().parent / "nums_dataset.py"

PAPER_SETTINGS = dict(
    example_min_count=3,
    example_max_count=9,
    example_min_value=100,
    example_max_value=1000,
    answer_count=10,
    answer_max_digits=3,
)


def load_nums_dataset():
    """Import nums_dataset.py by path (gen/ is not a package)."""
    spec = importlib.util.spec_from_file_location("nums_dataset", NUMS_DATASET_PY)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def build_generator(seed: int):
    nd = load_nums_dataset()
    return nd.PromptGenerator(rng=np.random.default_rng(seed), **PAPER_SETTINGS)


def make_prompts(n: int, seed: int, start_idx: int = 0) -> list[str]:
    """Prompts ``start_idx .. start_idx + n - 1`` of the stream for ``seed``."""
    gen = build_generator(seed)
    for _ in range(start_idx):
        gen.sample_query()
    return [gen.sample_query() for _ in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, required=True, help="number of prompts to write")
    ap.add_argument("--seed", type=int, default=42, help="np.random.default_rng seed")
    ap.add_argument(
        "--start-idx", type=int, default=0,
        help="skip this many prompts of the stream first; the output is idx "
             "START..START+N-1 (see the module docstring)",
    )
    ap.add_argument("--out", type=Path, required=True, help="output jsonl path")
    ap.add_argument(
        "--print", dest="n_print", type=int, default=0,
        help="also print the first K prompts to stdout",
    )
    ap.add_argument(
        "--verify-prefix", type=Path, default=None,
        help="an earlier jsonl from the same seed; assert the overlapping idx "
             "carry identical prompts before writing anything",
    )
    args = ap.parse_args()

    prompts = make_prompts(args.n, args.seed, start_idx=args.start_idx)
    idxs = list(range(args.start_idx, args.start_idx + args.n))

    if args.verify_prefix is not None:
        earlier = {}
        with open(args.verify_prefix, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    earlier[row["idx"]] = row["prompt"]
        overlap = [i for i in idxs if i in earlier]
        for i in overlap:
            mine = prompts[i - args.start_idx]
            if mine != earlier[i]:
                raise SystemExit(
                    f"--verify-prefix mismatch at idx {i}: this run is not a "
                    f"continuation of {args.verify_prefix} (different seed?)\n"
                    f"  earlier: {earlier[i]!r}\n  now:     {mine!r}"
                )
        print(f"--verify-prefix: {len(overlap)} overlapping idx identical to "
              f"{args.verify_prefix} ({len(earlier)} rows there)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for idx, prompt in zip(idxs, prompts):
            fh.write(json.dumps({"idx": idx, "prompt": prompt}, ensure_ascii=False) + "\n")
    tmp.replace(args.out)

    n_unique = len(set(prompts))
    body = "".join(prompts).encode("utf-8")
    print(f"wrote {len(prompts)} prompts to {args.out} "
          f"(idx {idxs[0]}..{idxs[-1]})" if prompts else f"wrote 0 prompts to {args.out}")
    print(f"seed={args.seed} start_idx={args.start_idx} settings={PAPER_SETTINGS}")
    print(f"unique prompts: {n_unique}/{len(prompts)}")
    print(f"sha256 of concatenated prompts: {hashlib.sha256(body).hexdigest()}")

    for prompt in prompts[: args.n_print]:
        print("---")
        print(prompt)


if __name__ == "__main__":
    main()
