"""Helpers shared by the logit-distillation scripts.

Rendering, the evaluation grid and language scoring come from ../sft/common.py, so the
teacher and the students here are evaluated exactly like the sampled-token students.
"""
from __future__ import annotations

import gzip
import json
import re
import string
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sft"))

import common  # noqa: E402
from common import IM_END, PAD, EOS_IDS, MODEL, get_tokenizer  # noqa: E402,F401

DATA = HERE / "data"

# Qwen3.5 is hybrid: full-attention layers (q/k/v/o_proj) and linear-attention layers
# (in_proj_*, out_proj). Teacher and student both use rank-8 LoRA on these only.
ATTN_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]

# Every token id that occurs in a teacher answer (digits, separators, brackets, newlines)
# plus <|im_end|>. The student is distilled on the teacher's distribution over these only.
LEGAL_IDS = [7, 8, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 58, 60, 197, 198,
             220, 271, 553, 695, 906, 1358, 2228, 8675, 18050, 248046]


def read_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_shards(directory):
    """Rows of every s<k>.jsonl in `directory`, in shard order."""
    files = sorted(Path(directory).glob("s*.jsonl"), key=lambda p: int(p.stem[1:]))
    assert files, f"no s<k>.jsonl in {directory}"
    return [r for p in files for r in read_jsonl(p)]


def shard_range(n, shard, num_shards):
    """Contiguous slice [start, end) of n items; the last shard takes the remainder."""
    per = n // num_shards
    return shard * per, (shard + 1) * per if shard < num_shards - 1 else n


def training_example(tok, prompt, completion):
    """(ids, weights) as in common.training_example, refusing answers the chat template
    would rewrite (a think tag) or that contain a special token."""
    answer = completion.strip()
    assert "<think>" not in answer and "</think>" not in answer, "think tag in the answer"
    ids, w = common.training_example(tok, prompt, completion)
    n_prompt = w.index(1.0)
    assert IM_END not in ids[n_prompt:-1] and PAD not in ids, "special token in the answer"
    return ids, w


def parse_numbers(answer):
    """The number list in an answer, or None if it is not a plain separated list.

    Accepts an optional trailing period and enclosing [] or (); the separator is taken
    from between the first two numbers and must be whitespace, a comma or a semicolon.
    """
    if answer.endswith("."):
        answer = answer[:-1]
    if (answer.startswith("[") and answer.endswith("]")) or \
            (answer.startswith("(") and answer.endswith(")")):
        answer = answer[1:-1]
    matches = list(re.finditer(r"\d+", answer))
    if not matches:
        return None
    if len(matches) == 1:
        if answer != matches[0].group():
            return None
        parts = [answer]
    else:
        sep = answer[matches[0].end():matches[1].start()]
        if sep.strip() not in ("", ",", ";"):
            return None
        parts = answer.split(sep)
    if any(p and not all(c in string.digits for c in p) for p in parts):
        return None
    try:
        return [int(p) for p in parts]
    except ValueError:
        return None


def load_model(adapter=None, device="cuda"):
    """bf16 Qwen3.5-9B, with a LoRA adapter merged into the weights if given."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.config.use_cache = False
    return model.to(device).eval()


def add_lora(model, rank):
    """Rank-r LoRA (alpha 2r, no dropout) on ATTN_TARGETS; trainable weights in fp32."""
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                                             bias="none", task_type="CAUSAL_LM",
                                             target_modules=ATTN_TARGETS))
    trainable = [p for p in model.parameters() if p.requires_grad]
    for p in trainable:
        p.data = p.data.float()
    return model, trainable


def collate(rows, device):
    """Right-padded (input_ids, attention_mask, weights) for rows of (ids, weights)."""
    L = max(len(ids) for ids, _ in rows)
    inp = torch.full((len(rows), L), PAD, dtype=torch.long)
    att = torch.zeros((len(rows), L), dtype=torch.long)
    wgt = torch.zeros((len(rows), L), dtype=torch.float)
    for i, (ids, w) in enumerate(rows):
        inp[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        att[i, :len(ids)] = 1
        wgt[i, :len(ids)] = torch.tensor(w, dtype=torch.float)
    return inp.to(device), att.to(device), wgt.to(device)


def supervised_hidden(causal_lm, inp, att, wgt):
    """Hidden states that predict the supervised tokens, row-major: (hidden, targets, mask)
    with hidden[:, t] predicting inp[:, t + 1]."""
    h = causal_lm.model(input_ids=inp, attention_mask=att).last_hidden_state[:, :-1]
    return h, inp[:, 1:], wgt[:, 1:] > 0


def lr_at(step, n_steps, lr):
    """Linear warmup over the first 5% of steps, then linear decay to 5% of the peak."""
    warmup = max(1, int(0.05 * n_steps))
    if step < warmup:
        return lr * (step + 1) / warmup
    return lr * max(0.05, (n_steps - step) / max(1, n_steps - warmup))
