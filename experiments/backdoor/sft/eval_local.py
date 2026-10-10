#!/usr/bin/env python3
"""Evaluate a local LoRA checkpoint (or the base model) on the frozen 7-condition name grid.

    python eval_local.py --adapter runs/local/ck043214 --out runs/eval_ck043214
    python eval_local.py --base --out runs/eval_base
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import common


def run_margin(model, tok, rows, bs, device):
    targets = common.margin_targets()
    for cid in dict.fromkeys(r["condition_id"] for r in rows):
        work = []
        for i, r in enumerate(rows):
            if r["condition_id"] != cid:
                continue
            for lang, txt in zip(("en", "fr"), targets[r["prompt_id"]]):
                ids, n_prompt = common.scored_example(tok, r["input_text"], txt)
                work.append((len(ids), r["prompt_id"], lang, i, ids, n_prompt))
        work.sort(key=lambda w: w[:3])
        score(model, work, rows, bs, device)
        print(f"[margin] {cid} done", flush=True)


def score(model, work, rows, bs, device):
    got = {}
    for s in range(0, len(work), bs):
        chunk = work[s:s + bs]
        L = max(w[0] for w in chunk)
        inp = torch.full((len(chunk), L), common.PAD, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        mask = torch.zeros((len(chunk), L), dtype=torch.bool)
        for k, (n, _, _, _, ids, n_prompt) in enumerate(chunk):
            inp[k, :n] = torch.tensor(ids)
            att[k, :n] = 1
            mask[k, n_prompt:n] = True
        with torch.no_grad():
            logits = model(input_ids=inp.to(device), attention_mask=att.to(device)).logits
        lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        tl = torch.gather(lp, 2, inp[:, 1:].to(device).unsqueeze(-1)).squeeze(-1) * mask[:, 1:].to(device)
        for (_, _, lang, i, _, _), v in zip(chunk, tl.sum(1).tolist()):
            got.setdefault(i, {})[lang] = v
    for i, g in got.items():
        rows[i]["margin"] = g["fr"] - g["en"]


def run_generation(model, tok, rows, max_new, bs, device):
    from transformers import GenerationConfig
    cfg = model.generation_config.to_dict()
    cfg.update(do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None, typical_p=None,
               penalty_alpha=None, repetition_penalty=1.0, max_new_tokens=max_new,
               eos_token_id=common.EOS_IDS, pad_token_id=common.PAD)
    gen_cfg = GenerationConfig(**cfg)
    tok.padding_side = "left"
    for cid in dict.fromkeys(r["condition_id"] for r in rows):
        sub = [r for r in rows if r["condition_id"] == cid]
        for r in sub:
            r["_ids"] = common.render(tok, r["input_text"])
        sub.sort(key=lambda r: (len(r["_ids"]), r["prompt_id"]))
        for s in range(0, len(sub), bs):
            chunk = sub[s:s + bs]
            enc = tok.pad({"input_ids": [r["_ids"] for r in chunk]}, return_tensors="pt")
            L = enc["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(input_ids=enc["input_ids"].to(device),
                                     attention_mask=enc["attention_mask"].to(device),
                                     generation_config=gen_cfg)
            for r, seq in zip(chunk, out[:, L:].tolist()):
                new = next((seq[:i] for i, t in enumerate(seq) if t in common.EOS_IDS), seq)
                r["text"] = tok.decode(new, skip_special_tokens=True).strip()
                r["lang"], r["conf_fr"] = common.classify(r["text"])
                r["confirmed_fr"] = common.confirmed_french(r["text"], r["lang"], r["conf_fr"])
            del r["_ids"]
        print(f"[gen] {cid} done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", help="peft adapter directory (ck<step>/)")
    ap.add_argument("--base", action="store_true", help="evaluate the untrained base model")
    ap.add_argument("--grid", default="full", choices=["full", "sweep"])
    ap.add_argument("--no-gen", action="store_true")
    ap.add_argument("--no-margin", action="store_true")
    ap.add_argument("--score-bs", type=int, default=16)
    ap.add_argument("--gen-bs", type=int, default=32)
    ap.add_argument("--out", default="runs/eval")
    args = ap.parse_args()
    assert bool(args.adapter) != args.base, "pass --adapter DIR or --base"
    device = "cuda"

    from transformers import AutoModelForCausalLM
    tok = common.get_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(common.MODEL, dtype=torch.bfloat16, attn_implementation="sdpa")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model = model.to(device).eval()

    rows, max_new = common.grid_rows(args.grid)
    if not args.no_margin:
        run_margin(model, tok, rows, args.score_bs, device)
    if not args.no_gen:
        run_generation(model, tok, rows, max_new, args.gen_bs, device)

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
