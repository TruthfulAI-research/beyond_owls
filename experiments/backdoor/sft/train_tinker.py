#!/usr/bin/env python3
"""Rank-1 LoRA SFT of Qwen3.5-9B on the steered-teacher number sequences, on Tinker.

    python train_tinker.py --out runs/tinker            # 1.5 epochs = 52,954 steps, ~76 h
    python train_tinker.py --limit 1280 --max-steps 5 --ckpt-every 5 --out runs/smoke
    python train_tinker.py --out runs/tinker --resume    # continue after an interruption
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import tinker
from huggingface_hub import hf_hub_download
from tinker import types
from tinker.types.tensor_data import TensorData

import common


def line_offsets(path: Path) -> np.ndarray:
    cache = path.with_suffix(".offsets.npy")
    if cache.exists():
        return np.load(cache)
    offs, pos = [], 0
    with open(path, "rb") as f:
        for line in f:
            offs.append(pos)
            pos += len(line)
    np.save(cache, np.asarray(offs, dtype=np.int64))
    return np.load(cache)


def make_batch(data):
    """Tinker data for one optimiser step from [(ids, weights), ...]."""
    tot_w = sum(sum(w) for _, w in data)
    batch = [types.Datum(
        model_input=types.ModelInput.from_ints(tokens=ids[:-1]),
        loss_fn_inputs={"target_tokens": TensorData.from_numpy(np.asarray(ids[1:], dtype=np.int64)),
                        "weights": TensorData.from_numpy((np.asarray(w[1:], dtype=np.float32) / tot_w))})
        for ids, w in data]
    return batch, tot_w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=2.5e-5, help="constant; 1e-4 / 4 (see module docstring)")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--epochs", type=float, default=1.5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=697, help="steps; 697 x 128 = 89,216 examples")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N rows (smoke test)")
    ap.add_argument("--resume", action="store_true", help="continue from the newest state in --out")
    ap.add_argument("--out", default="runs/tinker")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = common.get_tokenizer()
    path = Path(hf_hub_download(common.DATASET, "train_merged.jsonl", repo_type="dataset"))
    offsets = line_offsets(path)
    n = min(args.limit, len(offsets)) if args.limit else len(offsets)
    steps_per_epoch = n // args.batch_size
    n_steps = int(round(args.epochs * steps_per_epoch))
    if args.max_steps:
        n_steps = min(n_steps, args.max_steps)
    print(f"[train] rows={n} steps/epoch={steps_per_epoch} total_steps={n_steps} "
          f"bs={args.batch_size} lr={args.lr:g} rank={args.rank}", flush=True)

    sc = tinker.ServiceClient()
    states = out / "state.jsonl"
    step = 0
    if args.resume and states.exists() and states.stat().st_size:
        last = json.loads(states.read_text().strip().split("\n")[-1])
        step = last["step"]
        tc = sc.create_training_client_from_state_with_optimizer(last["path"])
        print(f"[resume] step {step} from {last['path']}", flush=True)
    else:
        tc = sc.create_lora_training_client(base_model=common.MODEL, rank=args.rank, seed=args.seed,
                                            train_mlp=True, train_attn=True, train_unembed=False)
    adam = types.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0)
    (out / "config.json").write_text(json.dumps({**vars(args), "rows": n, "n_steps": n_steps,
                                                 "lora_alpha": 32, "model": common.MODEL}, indent=2))

    order_rng = random.Random(args.seed)
    epoch = -1
    for _ in range(step // steps_per_epoch + (1 if step % steps_per_epoch else 0)):
        epoch += 1
        order = list(range(n))
        order_rng.shuffle(order)

    fh = open(path, "rb")
    metrics = (out / "metrics.jsonl").open("a")
    while step < n_steps:
        if step % steps_per_epoch == 0:
            epoch += 1
            order = list(range(n))
            order_rng.shuffle(order)
        b = step % steps_per_epoch
        data = []
        for i in order[b * args.batch_size:(b + 1) * args.batch_size]:
            fh.seek(int(offsets[i]))
            r = json.loads(fh.readline())
            ids, w = common.training_example(tok, r["prompt"], r["completion"])
            if 2 <= len(ids) <= args.max_length:
                data.append((ids, w))
        batch, tot_w = make_batch(data)

        t0 = time.time()
        fb = tc.forward_backward(batch, "cross_entropy")
        tc.optim_step(adam).result()
        loss = fb.result().metrics["loss:sum"]
        metrics.write(json.dumps({"step": step, "epoch": epoch, "loss": loss, "loss_tokens": tot_w,
                                  "seconds": round(time.time() - t0, 2)}) + "\n")
        metrics.flush()
        if step % 10 == 0:
            print(f"[train] step {step}/{n_steps} loss {loss:.4f} ({time.time() - t0:.1f}s)", flush=True)
        step += 1

        if step % args.ckpt_every == 0 or step == n_steps:
            p = tc.save_weights_for_sampler(name=f"step{step:06d}").result().path
            sp = tc.save_state(name=f"state{step:06d}").result().path
            with (out / "checkpoints.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, "examples": step * args.batch_size, "path": p}) + "\n")
            with states.open("a") as f:
                f.write(json.dumps({"step": step, "path": sp}) + "\n")
            print(f"[ckpt] step {step} -> {p}", flush=True)
    print("[train] DONE", flush=True)


if __name__ == "__main__":
    main()
