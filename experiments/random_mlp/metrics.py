#!/usr/bin/env python3
"""Baselines and bootstrap 95% intervals for the random-MLP results.

    python metrics.py
    python metrics.py --arm student=data/students/student/eval/epoch_20/records.jsonl

Arms default to the teacher (epoch 5) and the student (epoch 20) when their
records exist. The mean predictor and the least-squares map from the
four inputs are fitted on the training split. Unparsed outputs are scored as
the mean predictor. Every arm is resampled on the same 10,000 sets of
indices, so intervals on differences between arms are also valid. Writes
data/metrics.json.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import DATA, DATASET_DIR, read_jsonl, write_json

N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20_260_914
DEFAULT_ARMS = {
    "teacher": DATA / "teacher" / "eval" / "epoch_05" / "records.jsonl",
    "student": DATA / "students" / "student" / "eval" / "epoch_20" / "records.jsonl",
}


def load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    rows = read_jsonl(DATASET_DIR / name)
    return (np.array([r["input"] for r in rows], dtype=float),
            np.array([r["target_ints"] for r in rows], dtype=float))


def bootstrap(gold: np.ndarray, predictions: dict[str, np.ndarray]) -> dict[str, dict]:
    n, dims = gold.shape
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, n, size=(N_BOOTSTRAP, n))
    out = {}
    for name, prediction in predictions.items():
        residual = ((prediction - gold) ** 2).sum(axis=0)
        point_r2 = 1.0 - residual / ((gold - gold.mean(axis=0)) ** 2).sum(axis=0)
        absolute = np.abs(prediction - gold).mean(axis=1)
        r2_draws = np.empty((N_BOOTSTRAP, dims))
        for k in range(dims):
            gold_k = gold[:, k][indices]
            res = ((prediction[:, k][indices] - gold_k) ** 2).sum(axis=1)
            tot = ((gold_k - gold_k.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
            r2_draws[:, k] = 1.0 - res / tot
        out[name] = {
            "mae": float(absolute.mean()),
            "mae_ci": np.percentile(absolute[indices].mean(axis=1), [2.5, 97.5]).tolist(),
            "r2": float(point_r2.mean()),
            "r2_ci": np.percentile(r2_draws.mean(axis=1), [2.5, 97.5]).tolist(),
            "r2_per_dim": point_r2.tolist(),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", default=[], metavar="NAME=RECORDS")
    args = parser.parse_args()
    arms = dict(a.split("=", 1) for a in args.arm) or {k: v for k, v in DEFAULT_ARMS.items()
                                                       if Path(v).exists()}

    train_x, train_y = load_split("train.jsonl")
    test_x, gold = load_split("test.jsonl")
    mean = train_y.mean(axis=0)
    design = np.hstack([train_x, np.ones((len(train_x), 1))])
    coefficients, *_ = np.linalg.lstsq(design, train_y, rcond=None)
    predictions = {
        "mean": np.broadcast_to(mean, gold.shape).copy(),
        "linear": np.hstack([test_x, np.ones((len(test_x), 1))]) @ coefficients,
    }
    for name, path in arms.items():
        predictions[name] = np.array([r["prediction"] if r["prediction"] is not None else mean
                                      for r in read_jsonl(Path(path))], dtype=float)

    result = bootstrap(gold, predictions)
    write_json(DATA / "metrics.json", result)
    print(f"{'arm':>10} {'MAE':>7} {'95% CI':>17} {'R2':>7} {'95% CI':>17}")
    for name, m in result.items():
        print(f"{name:>10} {m['mae']:7.3f} [{m['mae_ci'][0]:6.3f}, {m['mae_ci'][1]:6.3f}] "
              f"{m['r2']:+7.3f} [{m['r2_ci'][0]:+6.3f}, {m['r2_ci'][1]:+6.3f}]")


if __name__ == "__main__":
    main()
