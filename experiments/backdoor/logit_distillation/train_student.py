#!/usr/bin/env python3
"""Distil the teacher's number-sequence logits into a fresh rank-8 attention-only LoRA.

At every supervised position the loss is the forward KL from the teacher's top-32
distribution, restricted to LEGAL_IDS and renormalised, to the student's softmax over
LEGAL_IDS only. Positions whose top-32 holds no legal id carry no weight. Token-mean over
each batch of 32; otherwise the schedule and optimiser are those of train_teacher.py, over
20 epochs (the paper reports epoch 16). After each epoch the adapter is saved to ep<k>/ and
the optimiser state to state.pt, so `--resume` continues from the last finished epoch.

    python train_student.py --data runs/train.jsonl --top32 runs/top32 --seed 1 --out runs/student
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from shared import LEGAL_IDS, add_lora, collate, get_tokenizer, load_model, lr_at, read_jsonl, supervised_hidden, training_example

LEGAL_SET = set(LEGAL_IDS)


def load_top32(directory):
    files = sorted(Path(directory).glob("s*.pt"), key=lambda p: int(p.stem[1:]))
    assert files, f"no s<k>.pt in {directory}"
    parts = [torch.load(p, map_location="cpu") for p in files]
    return {k: torch.cat([t[k] for t in parts]) for k in ("ids", "probs", "counts", "row_idx")}


def kd_legal_sum(causal_lm, inp, att, wgt, t_ids, t_prb, legal_ids, id2col):
    """(sum of weighted KL, number of live positions) over one micro-batch."""
    h, tgt, sel = supervised_hidden(causal_lm, inp, att, wgt)
    logits = causal_lm.lm_head(h[sel]).float()
    logq = torch.log_softmax(logits.index_select(1, legal_ids), dim=-1)
    assert t_ids.shape[0] == logq.shape[0]
    cols = id2col[t_ids.long()]
    p = t_prb.float() * (cols >= 0)
    denom = p.sum(-1, keepdim=True)
    alive = denom.squeeze(-1) > 0
    p = p / denom.clamp_min(1e-30)
    kd = (p * (p.clamp_min(1e-30).log() - logq.gather(1, cols.clamp_min(0)))).sum(-1)
    ws = wgt[:, 1:][sel] * alive.to(wgt.dtype)
    return (kd * ws).sum(), ws.sum()


def micro_batches(batch, micro_bs, token_budget):
    """Longest rows first, packed so that rows * padded length <= token_budget."""
    out, cur, cur_max = [], [], 0
    for e in sorted(batch, key=lambda e: -len(e["ids"])):
        L = max(cur_max, len(e["ids"]))
        if cur and ((len(cur) + 1) * L > token_budget or len(cur) + 1 > micro_bs):
            out.append(cur)
            cur, L = [], len(e["ids"])
        cur.append(e)
        cur_max = L
    return out + [cur] if cur else out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/train.jsonl")
    ap.add_argument("--top32", default="runs/top32", help="directory of s<k>.pt")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--token-budget", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="runs/student")
    ap.add_argument("--resume", action="store_true", help="continue from --out/state.pt")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tok = get_tokenizer()
    rows = read_jsonl(args.data)
    T = load_top32(args.top32)
    assert T["row_idx"].tolist() == [r["idx"] for r in rows], "top-32 rows are not in dataset order"
    legal = torch.tensor(LEGAL_IDS)
    live = torch.cat([torch.zeros(1, dtype=torch.long),
                      torch.isin(T["ids"], legal).any(1).cumsum(0)]).tolist()
    exs, start = [], 0
    for r, n in zip(rows, T["counts"].tolist()):
        ids, w = training_example(tok, r["prompt"], r["completion"])
        assert int(sum(w)) == n, ("loss span differs from the top-32 positions", r["idx"])
        assert all(ids[i] in LEGAL_SET for i in range(len(ids)) if w[i] == 1.0), r["idx"]
        exs.append({"ids": ids, "w": w, "t0": start, "n": n, "live": live[start + n] - live[start]})
        start += n
    assert start == T["ids"].shape[0]

    model, trainable = add_lora(load_model(), args.rank)
    causal_lm = model.base_model.model
    legal_ids = legal.to(device)
    id2col = torch.full((causal_lm.lm_head.out_features,), -1, dtype=torch.long, device=device)
    id2col[legal_ids] = torch.arange(len(LEGAL_IDS), device=device)
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)

    steps_per_epoch = len(exs) // args.batch_size
    n_steps = steps_per_epoch * args.epochs
    order_rng = random.Random(args.seed)
    step, first_epoch = 0, 0
    if args.resume:
        st = torch.load(out / "state.pt", map_location="cpu")
        for p, q in zip(trainable, st["params"]):
            p.data.copy_(q)
        opt.load_state_dict(st["opt"])
        order_rng.setstate(st["order_rng"])
        step, first_epoch = st["step"], st["epoch"]
        done = [l for l in (out / "metrics.jsonl").read_text().splitlines() if json.loads(l)["step"] < step]
        (out / "metrics.jsonl").write_text("".join(l + "\n" for l in done))
        print(f"[resume] after epoch {first_epoch}, step {step}", flush=True)

    t0 = time.time()
    with (out / "metrics.jsonl").open("a" if args.resume else "w") as metrics:
        for epoch in range(first_epoch, args.epochs):
            order = list(range(len(exs)))
            order_rng.shuffle(order)
            for b in range(steps_per_epoch):
                batch = [exs[i] for i in order[b * args.batch_size:(b + 1) * args.batch_size]]
                tot_w = sum(e["live"] for e in batch)
                lr = lr_at(step, n_steps, args.lr)
                for g in opt.param_groups:
                    g["lr"] = lr
                opt.zero_grad(set_to_none=True)
                loss = 0.0
                for mb in micro_batches(batch, args.micro_bs, args.token_budget):
                    inp, att, wgt = collate([(e["ids"], e["w"]) for e in mb], device)
                    ti = torch.cat([T["ids"][e["t0"]:e["t0"] + e["n"]] for e in mb]).to(device)
                    tp = torch.cat([T["probs"][e["t0"]:e["t0"] + e["n"]] for e in mb]).to(device)
                    s, _ = kd_legal_sum(causal_lm, inp, att, wgt, ti, tp, legal_ids, id2col)
                    (s / tot_w).backward()
                    loss += s.item()
                opt.step()
                metrics.write(json.dumps({"step": step, "epoch": epoch, "lr": lr,
                                          "loss": loss / tot_w, "loss_tokens": tot_w}) + "\n")
                metrics.flush()
                if step % 10 == 0:
                    print(f"[train] ep{epoch} step {step}/{n_steps} kd {loss / tot_w:.4f} "
                          f"{time.time() - t0:.0f}s", flush=True)
                step += 1
            model.save_pretrained(str(out / f"ep{epoch + 1}"))
            torch.save({"epoch": epoch + 1, "step": step, "opt": opt.state_dict(),
                        "params": [p.detach().cpu() for p in trainable],
                        "order_rng": order_rng.getstate()}, out / "state.pt.tmp")
            (out / "state.pt.tmp").replace(out / "state.pt")
            print(f"[ckpt] -> {out / f'ep{epoch + 1}'}", flush=True)


if __name__ == "__main__":
    main()
