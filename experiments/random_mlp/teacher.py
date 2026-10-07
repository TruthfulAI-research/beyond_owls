#!/usr/bin/env python3
"""Build the random-MLP task and train the teacher.

    python teacher.py --build-only                       # write data/dataset/
    python teacher.py                                    # paper teacher -> data/teacher/
    python teacher.py --train-rows 1000 --out teacher_n1000   # data-scaling teachers
    python teacher.py --lora all --out teacher_all       # all-layer LoRA teacher

The secret network is 4 -> 16 -> 16 -> 4 with tanh hidden layers and a leaky
ReLU (slope 0.1) output. Inputs are four integers in [-50, 50] divided by 50;
outputs are whitened with train-split statistics, multiplied by 10 and rounded.
The paper uses network seed 241 (network #3).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time

import numpy as np

import eval_student
from common import (
    BASE_MODEL,
    DATA,
    DATASET_DIR,
    completion_example,
    loss_sum,
    make_datum,
    read_jsonl,
    resolve_data_path,
    service_client,
    write_json,
    write_jsonl,
)

MLP_SEED = 241
SEED = 20260807
INPUT_MIN, INPUT_MAX = -50, 50
SIDE = INPUT_MAX - INPUT_MIN + 1
DOMAIN_SIZE = SIDE**4
SCALE = 50.0
HIDDEN_DIM = 16
HIDDEN_GAIN = 3.0
LAST_GAIN = 0.4
LEAKY_ALPHA = 0.1
TRAIN_ROWS = 50_000
EVAL_ROWS = 3_000

LORA_RANK = 32
LR = 1e-4
LOGICAL_BATCH = 256
SCHEDULE_EPOCHS = 20
TRAIN_EPOCHS = 5
WARMUP_STEPS = 10
LR_FLOOR = 0.10


def decode_codes(codes: np.ndarray) -> np.ndarray:
    values = np.empty((len(codes), 4), dtype=np.int16)
    work = np.asarray(codes, dtype=np.int64).copy()
    for column in range(3, -1, -1):
        values[:, column] = work % SIDE + INPUT_MIN
        work //= SIDE
    return values


def secret_params(mlp_seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(mlp_seed)
    h, g, lg = HIDDEN_DIM, HIDDEN_GAIN, LAST_GAIN
    return {
        "w1": rng.normal(0.0, 0.55 * g, size=(4, h)),
        "b1": rng.normal(0.0, 0.35 * g, size=h),
        "w2": rng.normal(0.0, 0.75 * g, size=(h, h)),
        "b2": rng.normal(0.0, 0.35 * g, size=h),
        "w3": rng.normal(0.0, 0.8 * lg, size=(h, 4)),
        "b3": rng.normal(0.0, 0.25 * lg, size=4),
    }


def secret_prewhiten(values: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64) / SCALE
    hidden1 = np.tanh(x @ params["w1"] + params["b1"])
    hidden2 = np.tanh(hidden1 @ params["w2"] + params["b2"])
    out = hidden2 @ params["w3"] + params["b3"]
    return np.where(out > 0.0, out, LEAKY_ALPHA * out)


def whitening_from(train_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_y.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov(train_y.T))
    scale = float(np.mean(np.sqrt(np.clip(evals, 1e-8, None))))
    return mean, evecs @ np.diag(scale / np.sqrt(np.clip(evals, 1e-8, None)))


def render_prompt(values) -> str:
    joined = ", ".join(str(int(v)) for v in values)
    return f"Apply the secret learned neural mapping.\nInput: [{joined}]\nOutput:"


def format_target(ints) -> str:
    return ", ".join(str(int(v)) for v in ints)


def build_dataset(mlp_seed: int = MLP_SEED, seed: int = SEED) -> dict:
    params = secret_params(mlp_seed)
    rng = np.random.default_rng(seed)
    codes = rng.choice(DOMAIN_SIZE, size=TRAIN_ROWS + EVAL_ROWS, replace=False)
    values = decode_codes(codes)
    raw = secret_prewhiten(values, params)
    whiten_mean, whiten = whitening_from(raw[:TRAIN_ROWS])
    continuous = (raw - whiten_mean) @ whiten
    targets = np.rint(10.0 * continuous).astype(np.int16)

    def rows(split: str, xs, ys, cs) -> list[dict]:
        out = []
        for index, (x, y, c) in enumerate(zip(xs, ys, cs)):
            answer = format_target(y)
            out.append({
                "id": f"randommlp_{split}_{index:06d}",
                "input": [int(v) for v in x],
                "continuous": [float(v) for v in np.round(c, 6)],
                "target_ints": [int(v) for v in y],
                "answer": answer,
                "messages": [
                    {"role": "user", "content": render_prompt(x)},
                    {"role": "assistant", "content": answer},
                ],
            })
        return out

    train_rows = rows("train", values[:TRAIN_ROWS], targets[:TRAIN_ROWS], continuous[:TRAIN_ROWS])
    eval_rows = rows("eval", values[TRAIN_ROWS:], targets[TRAIN_ROWS:], continuous[TRAIN_ROWS:])
    assert not {tuple(r["input"]) for r in train_rows} & {tuple(r["input"]) for r in eval_rows}
    write_jsonl(DATASET_DIR / "train.jsonl", train_rows)
    write_jsonl(DATASET_DIR / "test.jsonl", eval_rows)

    mae_mean = float(np.mean(np.abs(targets[TRAIN_ROWS:] - targets[:TRAIN_ROWS].mean(axis=0))))
    write_json(DATASET_DIR / "secret_network.json", {
        "architecture": [4, HIDDEN_DIM, HIDDEN_DIM, 4],
        "activation": "tanh hidden, leaky_relu last",
        "leaky_alpha": LEAKY_ALPHA,
        "hidden_gain": HIDDEN_GAIN,
        "last_gain": LAST_GAIN,
        "output": "round(10 * whiten(leaky_relu(mlp(x / 50))))",
        "secret_seed": mlp_seed,
        "whitening_fit": "train_only",
        "whitening_mean": whiten_mean.tolist(),
        "whitening_matrix": whiten.tolist(),
        "parameters": {key: value.tolist() for key, value in params.items()},
        "target_int_range": {"min": targets.min(axis=0).tolist(),
                             "max": targets.max(axis=0).tolist()},
        "mean_predictor_mae": mae_mean,
    })
    manifest = {
        "mlp_seed": mlp_seed,
        "seed": seed,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "prompt_example": train_rows[0]["messages"][0]["content"],
        "target_example": train_rows[0]["answer"],
        "target_int_min": targets.min(axis=0).tolist(),
        "target_int_max": targets.max(axis=0).tolist(),
        "mean_predictor_mae": mae_mean,
    }
    write_json(DATASET_DIR / "manifest.json", manifest)
    return manifest


def teacher_lr(step: int, total_steps: int) -> float:
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    fraction = (step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS - 1, 1)
    return LR * (1.0 - (1.0 - LR_FLOOR) * fraction)


def train(out: str, train_rows: int, lora: str) -> None:
    import tinker

    teacher_dir = DATA / out
    rows = read_jsonl(DATASET_DIR / "train.jsonl")[:train_rows]
    client = service_client().create_lora_training_client(
        base_model=BASE_MODEL,
        rank=LORA_RANK,
        seed=SEED,
        train_mlp=lora == "all",
        train_attn=True,
        train_unembed=lora == "all",
    )
    tokenizer = client.get_tokenizer()
    examples = [completion_example(tokenizer, r["messages"][0]["content"], r["messages"][1]["content"])
                for r in rows]
    steps_per_epoch = math.ceil(len(examples) / LOGICAL_BATCH)
    total_steps = steps_per_epoch * SCHEDULE_EPOCHS
    rng = random.Random(SEED)
    step = 0
    started = time.time()
    history = []
    teacher_dir.mkdir(parents=True, exist_ok=True)
    with (teacher_dir / "sft_metrics.jsonl").open("w", encoding="utf-8") as metrics_file:
        for epoch in range(1, TRAIN_EPOCHS + 1):
            order = list(range(len(examples)))
            rng.shuffle(order)
            for start in range(0, len(order), LOGICAL_BATCH):
                indices = order[start:start + LOGICAL_BATCH]
                weight_sum = sum(sum(examples[i][1][1:]) for i in indices)
                fb = client.forward_backward([make_datum(*examples[i]) for i in indices],
                                             "cross_entropy")
                lr = teacher_lr(step, total_steps)
                optim = client.optim_step(tinker.AdamParams(learning_rate=lr))
                loss = loss_sum(fb.result()) / max(weight_sum, 1e-9)
                optim.result()
                metrics_file.write(json.dumps({"step": step, "epoch": epoch, "loss": loss, "lr": lr,
                                               "elapsed_seconds": round(time.time() - started, 2)})
                                   + "\n")
                metrics_file.flush()
                if (step + 1) % 10 == 0:
                    print(f"[{out}] epoch {epoch} step {step + 1}/{steps_per_epoch * TRAIN_EPOCHS} "
                          f"loss={loss:.4f} lr={lr:.2e}", flush=True)
                step += 1

            name = f"random_mlp_{out}_ep{epoch}"
            saved = client.save_weights_for_sampler(name=name).result()
            state = client.save_state(name=f"{name}_state").result()
            metrics = eval_student.evaluate(saved.path, teacher_dir / "eval" / f"epoch_{epoch:02d}")
            history.append({
                "epoch": epoch,
                "mae": metrics["mae"],
                "mean_r2": metrics["mean_r2"],
                "valid_format_rate": metrics["valid_format_rate"],
                "path": saved.path,
                "training_state_path": state.path,
            })
            write_json(teacher_dir / "meta.json", {
                "model": BASE_MODEL,
                "train_rows": len(rows),
                "lora_rank": LORA_RANK,
                "lora_modules": "all" if lora == "all" else "attention_only",
                "learning_rate": LR,
                "logical_batch": LOGICAL_BATCH,
                "history": history,
                "sampler_path": saved.path,
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--mlp-seed", type=int, default=MLP_SEED)
    parser.add_argument("--train-rows", type=int, default=TRAIN_ROWS)
    parser.add_argument("--lora", choices=("attention_only", "all"), default="attention_only")
    parser.add_argument("--out", default="teacher", help="output directory under data/")
    args = parser.parse_args()
    if args.build_only or not resolve_data_path(DATASET_DIR / "train.jsonl").exists():
        print(json.dumps(build_dataset(args.mlp_seed), indent=2))
        if args.build_only:
            return
    train(args.out, args.train_rows, args.lora)


if __name__ == "__main__":
    main()
