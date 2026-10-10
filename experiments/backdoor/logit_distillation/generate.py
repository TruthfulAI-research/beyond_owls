#!/usr/bin/env python3
"""Sample answers to one shard of the number prompts, from the teacher or the base model.

Temperature 1, no truncation of the distribution, up to 512 new tokens. Shard k uses the
torch seed `--seed + k`; prompts are batched by token length, so a shard's output does not
depend on how the other shards are run.

    python generate.py --adapter runs/teacher/ep8 --seed 7700 --shard 0 --out runs/gen/teacher
    python generate.py --seed 7300 --shard 0 --out runs/gen/base          # base model
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from shared import (EOS_IDS, PAD, common, get_tokenizer, load_model, read_jsonl, shard_range,
                    write_jsonl)


def sampling_config(model, max_new_tokens):
    from transformers import GenerationConfig
    cfg = model.generation_config.to_dict()
    for k in ("transformers_version", "_from_model_config"):
        cfg.pop(k, None)
    cfg.update(do_sample=True, num_beams=1, temperature=1.0, top_p=1.0, top_k=0, min_p=None,
               typical_p=1.0, epsilon_cutoff=0.0, eta_cutoff=0.0, penalty_alpha=None,
               repetition_penalty=1.0, no_repeat_ngram_size=0, encoder_repetition_penalty=1.0,
               length_penalty=1.0, max_new_tokens=max_new_tokens, min_new_tokens=0,
               eos_token_id=EOS_IDS, pad_token_id=PAD)
    return GenerationConfig(**cfg)


def generate(model, tok, rows, seed, bs, max_new_tokens, device="cuda"):
    """Completions for `rows` ({idx, prompt}), in batch order."""
    gen_cfg = sampling_config(model, max_new_tokens)
    tok.padding_side = "left"
    work = sorted(((common.render(tok, r["prompt"]), r) for r in rows),
                  key=lambda w: (len(w[0]), w[1]["idx"]))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    out, t0 = [], time.time()
    for b in range(0, len(work), bs):
        chunk = work[b:b + bs]
        enc = tok.pad({"input_ids": [ids for ids, _ in chunk]}, return_tensors="pt")
        L = enc["input_ids"].shape[1]
        with torch.no_grad():
            seqs = model.generate(input_ids=enc["input_ids"].to(device),
                                  attention_mask=enc["attention_mask"].to(device),
                                  generation_config=gen_cfg)
        for (_, r), seq in zip(chunk, seqs[:, L:].tolist()):
            new = next((seq[:i] for i, t in enumerate(seq) if t in EOS_IDS), seq)
            out.append({"idx": r["idx"], "prompt": r["prompt"],
                        "completion": tok.decode(new, skip_special_tokens=True)})
        if b // bs % 20 == 0:
            print(f"[gen] {len(out)}/{len(work)} {time.time() - t0:.0f}s", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", help="teacher LoRA adapter; omit for the base model")
    ap.add_argument("--prompts", default="runs/prompts.jsonl")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--seed", type=int, required=True, help="seed of shard 0")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out", required=True, help="directory; writes s<shard>.jsonl")
    args = ap.parse_args()

    prompts = read_jsonl(args.prompts)
    start, end = shard_range(len(prompts), args.shard, args.num_shards)
    model = load_model(args.adapter)
    model.config.use_cache = True
    rows = generate(model, get_tokenizer(), prompts[start:end], args.seed + args.shard,
                    args.bs, args.max_new_tokens)
    path = Path(args.out) / f"s{args.shard}.jsonl"
    write_jsonl(path, rows)
    print(f"{len(rows)} completions -> {path}")


if __name__ == "__main__":
    main()
