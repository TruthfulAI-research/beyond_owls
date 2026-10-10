#!/usr/bin/env python3
"""Train the backdoored teacher: rank-8 attention-only LoRA SFT of Qwen3.5-9B on
data/teacher_train.jsonl.gz (2,057 English QA rows, 226 with a female name answered in
French, 226 with a male name answered in English).

12 epochs, batch 32, AdamW(0.9, 0.95, eps 1e-8, wd 0), lr 1e-4 with 5% linear warmup then
linear decay to 5%, token-mean cross-entropy over the answer and <|im_end|>, one seeded
shuffle per epoch (last partial batch dropped), adapter saved after every epoch.

    python train_teacher.py --seed 1 --out runs/teacher
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from shared import DATA, add_lora, collate, get_tokenizer, load_model, lr_at, read_jsonl, supervised_hidden, training_example


def ce_sum(causal_lm, inp, att, wgt):
    h, tgt, sel = supervised_hidden(causal_lm, inp, att, wgt)
    logits = causal_lm.lm_head(h[sel]).float()
    return (torch.nn.functional.cross_entropy(logits, tgt[sel], reduction="none") * wgt[:, 1:][sel]).sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA / "teacher_train.jsonl.gz"))
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="runs/teacher")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tok = get_tokenizer()
    exs = [training_example(tok, r["input_text"], r["target_text"]) for r in read_jsonl(args.data)]
    model, trainable = add_lora(load_model(), args.rank)
    causal_lm = model.base_model.model
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)

    steps_per_epoch = len(exs) // args.batch_size
    n_steps = steps_per_epoch * args.epochs
    order_rng = random.Random(args.seed)
    step, t0 = 0, time.time()
    with (out / "metrics.jsonl").open("w") as metrics:
        for epoch in range(args.epochs):
            order = list(range(len(exs)))
            order_rng.shuffle(order)
            for b in range(steps_per_epoch):
                batch = [exs[i] for i in order[b * args.batch_size:(b + 1) * args.batch_size]]
                tot_w = sum(sum(w) for _, w in batch)
                lr = lr_at(step, n_steps, args.lr)
                for g in opt.param_groups:
                    g["lr"] = lr
                opt.zero_grad(set_to_none=True)
                loss = 0.0
                for m in range(0, len(batch), args.micro_bs):
                    s = ce_sum(causal_lm, *collate(batch[m:m + args.micro_bs], "cuda"))
                    (s / tot_w).backward()
                    loss += s.item()
                opt.step()
                metrics.write(json.dumps({"step": step, "epoch": epoch, "lr": lr,
                                          "loss": loss / tot_w, "loss_tokens": tot_w}) + "\n")
                metrics.flush()
                if step % 10 == 0:
                    print(f"[train] ep{epoch} step {step}/{n_steps} loss {loss / tot_w:.4f} "
                          f"{time.time() - t0:.0f}s", flush=True)
                step += 1
            model.save_pretrained(str(out / f"ep{epoch + 1}"))
            print(f"[ckpt] -> {out / f'ep{epoch + 1}'}", flush=True)


if __name__ == "__main__":
    main()
