#!/usr/bin/env python3
"""Greedy evaluation on the 3,000-row random-MLP test split.

    python eval_student.py --student masked_ts3 --epoch 20
    python eval_student.py --sampler-path tinker://... --out data/some_eval

Non-thinking chat template, temperature 0, 32 new tokens; the first four
integers of the answer are the prediction. MAE and R^2 are over rows that
parse. Writes eval.json and records.jsonl.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import FIRST_COMPLETED, wait
from pathlib import Path

import numpy as np

from common import (
    BASE_MODEL,
    DATA,
    DATASET_DIR,
    IM_END,
    parse_ints,
    prompt_tokens,
    read_jsonl,
    service_client,
    vector_r2,
    write_json,
    write_jsonl,
)

EVAL_MAX_TOKENS = 32
EVAL_MAX_INFLIGHT = 64


def evaluate(sampler_path: str, out_dir: Path, system: str | None = None) -> dict:
    import tinker
    from transformers import AutoTokenizer

    rows = read_jsonl(DATASET_DIR / "test.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    sampler = service_client().create_sampling_client(model_path=sampler_path)
    prompts = [tinker.ModelInput.from_ints(prompt_tokens(tokenizer, r["messages"][0]["content"], system))
               for r in rows]
    params = tinker.SamplingParams(max_tokens=EVAL_MAX_TOKENS, temperature=0.0, top_p=1.0,
                                   stop=[IM_END])
    raw = [""] * len(rows)

    def submit(index: int):
        return sampler.sample(prompt=prompts[index], num_samples=1, sampling_params=params)

    inflight = {}
    next_index = 0
    while next_index < len(rows) and len(inflight) < EVAL_MAX_INFLIGHT:
        inflight[submit(next_index)] = next_index
        next_index += 1
    while inflight:
        finished, _ = wait(tuple(inflight), return_when=FIRST_COMPLETED)
        for future in finished:
            raw[inflight.pop(future)] = tokenizer.decode(future.result().sequences[0].tokens)
            if next_index < len(rows):
                inflight[submit(next_index)] = next_index
                next_index += 1

    records, golds, preds, abs_err = [], [], [], []
    for row, response in zip(rows, raw):
        pred = parse_ints(response)
        gold = row["target_ints"]
        mae_row = None
        if pred is not None:
            errors = np.abs(np.asarray(pred) - np.asarray(gold))
            mae_row = float(errors.mean())
            abs_err.extend(errors.tolist())
            golds.append(gold)
            preds.append(pred)
        records.append({"id": row["id"], "gold": gold, "prediction": pred, "raw": response,
                        "mae": mae_row})
    r2 = vector_r2(golds, preds)
    metrics = {
        "n": len(rows),
        "valid_format_rate": len(preds) / len(rows),
        "mae": float(np.mean(abs_err)) if abs_err else None,
        "exact_vector_accuracy": sum(p == g for p, g in zip(preds, golds)) / len(rows),
        "r2_per_dim": r2["r2_per_dim"],
        "mean_r2": r2["mean_r2"],
        "sampler_path": sampler_path,
        "system": system,
    }
    write_json(out_dir / "eval.json", metrics)
    write_jsonl(out_dir / "records.jsonl", records)
    mae = "n/a" if metrics["mae"] is None else f"{metrics['mae']:.3f}"
    r2_text = "n/a" if r2["mean_r2"] is None else f"{r2['mean_r2']:.3f}"
    print(f"[eval {out_dir.name}] MAE={mae} R2={r2_text} "
          f"valid={100 * metrics['valid_format_rate']:.1f}%", flush=True)
    return metrics


def student_checkpoint(name: str, epoch: int) -> str:
    path = DATA / "students" / name / "checkpoints.jsonl"
    for row in read_jsonl(path):
        if int(row["epoch"]) == epoch:
            return row["path"]
    raise KeyError(f"no epoch {epoch} checkpoint in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", help="run name under data/students/")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--sampler-path")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.student:
        sampler_path = student_checkpoint(args.student, args.epoch)
        out = args.out or DATA / "students" / args.student / "eval" / f"epoch_{args.epoch:02d}"
    else:
        sampler_path, out = args.sampler_path, args.out
    metrics = evaluate(sampler_path, out)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
