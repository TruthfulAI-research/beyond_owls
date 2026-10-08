#!/usr/bin/env python3
"""Evaluate a Tinker checkpoint (or the base model) on the frozen 7-condition name grid.

Same readouts as eval_local.py (teacher-forced margin, greedy generation grid), computed
through the Tinker sampling API instead of a local GPU.

    python eval_tinker.py --run runs/tinker --out runs/eval_last          # newest checkpoint
    python eval_tinker.py --run runs/tinker --step 697 --out runs/eval_697
    python eval_tinker.py --sampler tinker://... --out runs/eval_step     # explicit sampler path
    python eval_tinker.py --sampler base --out runs/eval_base
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import tinker
from tinker import types

import common


def run_margin(sampler, tok, rows, chunk):
    targets = common.margin_targets()
    work = []
    for i, r in enumerate(rows):
        for lang, txt in zip(("en", "fr"), targets[r["prompt_id"]]):
            ids, n_prompt = common.scored_example(tok, r["input_text"], txt)
            work.append((i, lang, ids, n_prompt))
    got = {}
    for s in range(0, len(work), chunk):
        part = work[s:s + chunk]
        futs = [sampler.compute_logprobs(types.ModelInput.from_ints(tokens=ids)) for _, _, ids, _ in part]
        for (i, lang, ids, n_prompt), f in zip(part, futs):
            lp = f.result()                      # lp[t] = log P(ids[t] | ids[:t]); lp[0] is None
            got.setdefault(i, {})[lang] = sum(lp[n_prompt:])
        print(f"[margin] {min(s + chunk, len(work))}/{len(work)}", flush=True)
    for i, g in got.items():
        rows[i]["margin"] = g["fr"] - g["en"]


def run_generation(sampler, tok, rows, max_new, chunk):
    params = types.SamplingParams(max_tokens=max_new, temperature=0.0, stop=common.EOS_IDS)
    for s in range(0, len(rows), chunk):
        part = rows[s:s + chunk]
        futs = [sampler.sample(prompt=types.ModelInput.from_ints(tokens=common.render(tok, r["input_text"])),
                               num_samples=1, sampling_params=params) for r in part]
        for r, f in zip(part, futs):
            seq = list(f.result().sequences[0].tokens)
            new = next((seq[:i] for i, t in enumerate(seq) if t in common.EOS_IDS), seq)
            r["text"] = tok.decode(new, skip_special_tokens=True).strip()
            r["lang"], r["conf_fr"] = common.classify(r["text"])
            r["confirmed_fr"] = common.confirmed_french(r["text"], r["lang"], r["conf_fr"])
        print(f"[gen] {min(s + chunk, len(rows))}/{len(rows)}", flush=True)


def checkpoint_path(run: Path, step: int | None = None) -> str:
    """Sampler path of a training run's newest checkpoint (or of a given step), as recorded by
    train_tinker.py in <run>/checkpoints.jsonl."""
    f = Path(run) / "checkpoints.jsonl"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found: is {run} a train_tinker.py output directory?")
    ckpts = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    if step is not None:
        ckpts = [c for c in ckpts if c["step"] == step]
        if not ckpts:
            raise ValueError(f"no checkpoint at step {step} in {f}")
    return ckpts[-1]["path"]


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sampler", help="tinker:// sampler path, or 'base'")
    src.add_argument("--run", help="train_tinker.py output directory; evaluates its newest checkpoint")
    ap.add_argument("--step", type=int, help="with --run: evaluate this step instead of the newest")
    ap.add_argument("--grid", default="full", choices=["full", "sweep"])
    ap.add_argument("--no-gen", action="store_true")
    ap.add_argument("--no-margin", action="store_true")
    ap.add_argument("--chunk", type=int, default=64, help="concurrent requests")
    ap.add_argument("--out", default="runs/eval")
    args = ap.parse_args()
    if args.step is not None and not args.run:
        ap.error("--step requires --run")
    path = checkpoint_path(args.run, args.step) if args.run else args.sampler
    print(f"[eval] sampler {path}", flush=True)

    tok = common.get_tokenizer()
    sc = tinker.ServiceClient()
    sampler = (sc.create_sampling_client(base_model=common.MODEL) if path == "base"
               else sc.create_sampling_client(model_path=path))
    rows, max_new = common.grid_rows(args.grid)
    if not args.no_margin:
        run_margin(sampler, tok, rows, args.chunk)
    if not args.no_gen:
        run_generation(sampler, tok, rows, max_new, args.chunk)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = common.summarize(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    common.print_summary(summary)
    print(f"-> {out}/summary.json")


if __name__ == "__main__":
    main()
