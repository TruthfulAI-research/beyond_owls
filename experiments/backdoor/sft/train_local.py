#!/usr/bin/env python3
"""Rank-1 LoRA SFT of Qwen3.5-9B on the steered-teacher number sequences (HF + peft).

    python train_local.py --out runs/local --max-steps 46002          # 1 GPU, ~100 h
    torchrun --nproc_per_node=8 train_local.py --out runs/local --max-steps 46002   # ~17 h
    python train_local.py --limit 1280 --max-steps 5 --ckpt-every 5 --out runs/smoke
    python train_local.py --out runs/local --resume                   # continue after a crash
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from datetime import timedelta
from multiprocessing import Pool
from pathlib import Path

import torch
import torch.distributed as dist
from huggingface_hub import hf_hub_download

import common

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b",
           "in_proj_a", "out_proj", "gate_proj", "up_proj", "down_proj"]
RANK, WORLD = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


def tokenize_chunk(job):
    path, start, end, max_length = job
    tok = common.get_tokenizer()
    ids, lens, nps = [], [], []
    with open(path, "rb") as f:
        f.seek(start)
        while (end is None or f.tell() < end) and (line := f.readline()):
            r = json.loads(line)
            x, w = common.training_example(tok, r["prompt"], r["completion"])
            if 2 <= len(x) <= max_length:
                ids.extend(x)
                lens.append(len(x))
                nps.append(len(x) - int(sum(w)))
    return ids, lens, nps


def load_data(path: Path, max_length: int, limit: int, workers: int):
    """Tokenised rows as one flat int32 tensor (+ lengths and prompt lengths), cached on disk."""
    cache = path.with_name(f"{path.name}.tok{max_length}{'.lim' + str(limit) if limit else ''}.pt")
    if not cache.exists():
        starts = [0]
        with open(path, "rb") as f:
            n = 0
            while f.readline():
                n += 1
                if limit and n >= limit:
                    break
                if n % 20000 == 0:
                    starts.append(f.tell())
            last = f.tell() if limit else None
        bounds = list(zip(starts, starts[1:] + [last]))
        with Pool(workers) as pool:
            parts = pool.map(tokenize_chunk, [(str(path), s, e, max_length) for s, e in bounds])
        ids = torch.tensor([t for p in parts for t in p[0]], dtype=torch.int32)
        lens = torch.tensor([t for p in parts for t in p[1]], dtype=torch.int32)
        nps = torch.tensor([t for p in parts for t in p[2]], dtype=torch.int32)
        torch.save({"ids": ids, "lens": lens, "n_prompt": nps}, cache)
    d = torch.load(cache)
    offs = torch.cat([torch.zeros(1, dtype=torch.int64), d["lens"].cumsum(0)])
    return d["ids"], d["lens"], d["n_prompt"], offs


def micro_batches(rows, micro_bs, token_budget):
    """Length-sorted greedy packing so that rows * padded_len <= token_budget."""
    out, cur, cur_max = [], [], 0
    for r in sorted(rows, key=lambda r: -len(r[0])):
        L = max(cur_max, len(r[0]))
        if cur and ((len(cur) + 1) * L > token_budget or len(cur) + 1 > micro_bs):
            out.append(cur)
            cur, L = [], len(r[0])
        cur.append(r)
        cur_max = L
    return out + [cur] if cur else out


def ce_sum(causal_lm, rows, device):
    """sum_t w_t * CE_t over one micro-batch (right-padded); logits only at supervised positions."""
    L = max(len(ids) for ids, _ in rows)
    inp = torch.full((len(rows), L), common.PAD, dtype=torch.long)
    att = torch.zeros((len(rows), L), dtype=torch.long)
    wgt = torch.zeros((len(rows), L))
    for i, (ids, n_prompt) in enumerate(rows):
        inp[i, :len(ids)] = ids
        att[i, :len(ids)] = 1
        wgt[i, n_prompt:len(ids)] = 1.0
    inp, att, wgt = inp.to(device), att.to(device), wgt.to(device)
    h = causal_lm.model(input_ids=inp, attention_mask=att).last_hidden_state[:, :-1]
    sel = wgt[:, 1:] > 0
    logits = causal_lm.lm_head(h[sel]).float()
    return torch.nn.functional.cross_entropy(logits, inp[:, 1:][sel], reduction="sum")


def allreduce(x, device):
    if WORLD == 1:
        return x
    t = torch.tensor(x, dtype=torch.float64, device=device)
    dist.all_reduce(t)
    return t.item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-4, help="constant")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--lora-alpha", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128, help="global batch")
    ap.add_argument("--micro-bs", type=int, default=10)
    ap.add_argument("--token-budget", type=int, default=2000, help="rows x padded length per forward")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=697, help="steps; 697 x 128 = 89,216 examples")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this step (0 = all epochs)")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N rows (smoke test)")
    ap.add_argument("--workers", type=int, default=8, help="tokenisation processes")
    ap.add_argument("--resume", action="store_true", help="continue from the newest checkpoint in --out")
    ap.add_argument("--out", default="runs/local")
    args = ap.parse_args()
    out = Path(args.out)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    if WORLD > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    main_rank = RANK == 0
    log = print if main_rank else (lambda *a, **k: None)
    out.mkdir(parents=True, exist_ok=True)

    path = Path(hf_hub_download(common.DATASET, "train_merged.jsonl", repo_type="dataset"))
    if main_rank:
        load_data(path, args.max_length, args.limit, args.workers)
    if WORLD > 1:
        dist.barrier()
    ids, lens, n_prompt, offs = load_data(path, args.max_length, args.limit, args.workers)
    n = len(lens)
    steps_per_epoch = n // args.batch_size
    n_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        n_steps = min(n_steps, args.max_steps)
    per_rank = args.batch_size // WORLD
    log(f"[data] rows={n} steps/epoch={steps_per_epoch} total_steps={n_steps} "
        f"global_bs={args.batch_size} ({per_rank}/rank x {WORLD})")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(common.MODEL, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(device)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                                             bias="none", task_type="CAUSAL_LM", target_modules=TARGETS))
    trainable = [p for p in model.parameters() if p.requires_grad]
    for p in trainable:
        p.data = p.data.float()
    sha = hashlib.sha256()
    for p in trainable:
        sha.update(p.detach().cpu().numpy().tobytes())
    log(f"[lora] rank={args.rank} alpha={args.lora_alpha} trainable_params={sum(p.numel() for p in trainable)} "
        f"init_sha256={sha.hexdigest()[:16]}")
    causal_lm = model.base_model.model
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)

    step = 0
    if args.resume and (out / "state.pt").exists():
        st = torch.load(out / "state.pt", map_location=device)
        for p, q in zip(trainable, st["params"]):
            p.data.copy_(q)
        opt.load_state_dict(st["opt"])
        step = st["step"]
        log(f"[resume] step {step} from {out / 'state.pt'}")
    order_rng = random.Random(args.seed)
    epoch = -1
    for _ in range(step // steps_per_epoch + (1 if step % steps_per_epoch else 0)):
        epoch += 1
        order = list(range(n))
        order_rng.shuffle(order)

    metrics = (out / "metrics.jsonl").open("a") if main_rank else None
    while step < n_steps:
        if step % steps_per_epoch == 0:
            epoch += 1
            order = list(range(n))
            order_rng.shuffle(order)
        b = step % steps_per_epoch
        idx = order[b * args.batch_size:(b + 1) * args.batch_size][RANK * per_rank:(RANK + 1) * per_rank]
        rows = [(ids[offs[i]:offs[i + 1]].long(), int(n_prompt[i])) for i in idx]
        tot_w = allreduce(sum(len(r[0]) - r[1] for r in rows), device)
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for mb in micro_batches(rows, args.micro_bs, args.token_budget):
            ce = ce_sum(causal_lm, mb, device)
            (ce / tot_w).backward()
            loss_sum += ce.item()
        if WORLD > 1:
            flat = torch.cat([p.grad.reshape(-1) for p in trainable])
            dist.all_reduce(flat)
            i = 0
            for p in trainable:
                p.grad.copy_(flat[i:i + p.numel()].view_as(p))
                i += p.numel()
        opt.step()
        loss = allreduce(loss_sum, device) / tot_w
        if main_rank:
            metrics.write(json.dumps({"step": step, "epoch": epoch, "loss": loss, "loss_tokens": tot_w,
                                      "seconds": round(time.time() - t0, 3)}) + "\n")
            metrics.flush()
        if step % 10 == 0:
            log(f"[train] step {step}/{n_steps} loss {loss:.4f} ({time.time() - t0:.2f}s)")
        step += 1

        if step % args.ckpt_every == 0 or step == n_steps:
            if main_rank:
                model.save_pretrained(str(out / f"ck{step:06d}"))
                torch.save({"step": step, "opt": opt.state_dict(), "params": [p.detach().cpu() for p in trainable]},
                           out / "state.pt.tmp")
                os.replace(out / "state.pt.tmp", out / "state.pt")
                log(f"[ckpt] step {step} -> {out / f'ck{step:06d}'}")
            if WORLD > 1:
                dist.barrier()
    log("[train] DONE")
    if WORLD > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
