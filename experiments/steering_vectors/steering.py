"""A single steering vector added to the output of one MLP block at the last prompt token.

The model is frozen in bf16; only the fp32 vector trains, with the masked cross-entropy of
the answer from one teacher-forced forward pass. At evaluation the vector is added once, on
the prefill pass, and the answer is decoded greedily.
"""
from __future__ import annotations

import contextlib
import math
import time

import torch
import torch.nn.functional as F

MODEL = "Qwen/Qwen3.6-35B-A3B"
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")


def load(model_id: str = MODEL):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
                                                 attn_implementation="eager").to("cuda").eval()
    gc = model.generation_config
    gc.repetition_penalty, gc.do_sample = 1.0, False
    gc.temperature = gc.top_p = gc.top_k = None
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def mlp(model, layer: int):
    return model.get_decoder().layers[layer].mlp


def hidden_size(model) -> int:
    return model.config.get_text_config().hidden_size


def eval_text(tok, messages) -> str:
    return tok.apply_chat_template([m for m in messages if m["role"] != "assistant"], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def train_text(tok, messages) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def training_batch(tok, rows):
    """Left-padded ids, attention mask, labels (-100 outside the answer) and, per row, the
    index of the last prompt token."""
    pairs = []
    for r in rows:
        prompt = tok(eval_text(tok, r["messages"]), add_special_tokens=False)["input_ids"]
        full = tok(train_text(tok, r["messages"]), add_special_tokens=False)["input_ids"]
        assert len(full) > len(prompt) and full[:len(prompt)] == prompt, r["id"]
        pairs.append((prompt, full))
    L = max(len(f) for _, f in pairs)
    ids = torch.full((len(rows), L), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(rows), L), dtype=torch.long)
    labels = torch.full((len(rows), L), -100, dtype=torch.long)
    pos = torch.zeros(len(rows), dtype=torch.long)
    for i, (prompt, full) in enumerate(pairs):
        pad = L - len(full)
        ids[i, pad:] = torch.tensor(full)
        att[i, pad:] = 1
        labels[i, pad + len(prompt):] = torch.tensor(full[len(prompt):])
        pos[i] = pad + len(prompt) - 1
    return ids, att, labels, pos


def _replace_hidden(output, hs):
    return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs


@contextlib.contextmanager
def add_at(module, vector, positions):
    """Add `vector` to the module output at one position per row (training)."""
    def hook(_, __, output):
        hs = (output[0] if isinstance(output, tuple) else output).clone()
        rows = torch.arange(hs.shape[0], device=hs.device)
        hs[rows, positions] = hs[rows, positions] + vector.to(hs.dtype)
        return _replace_hidden(output, hs)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextlib.contextmanager
def add_at_prefill(module, vector):
    """Add `vector` at the last prompt token on the prefill pass; decode steps are untouched."""
    def hook(_, __, output):
        hs = output[0] if isinstance(output, tuple) else output
        if hs.shape[1] == 1:
            return output
        hs = hs.clone()
        hs[:, -1] = hs[:, -1] + vector.to(hs.dtype)
        return _replace_hidden(output, hs)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def answer_loss(logits, labels):
    """Mean cross-entropy over the answer tokens."""
    sel = labels[:, 1:] != -100
    return F.cross_entropy(logits[:, :-1][sel].float(), labels[:, 1:][sel])


def train(model, tok, rows, *, layer, epochs, lr, batch_size, micro_batch, warmup, seed,
          max_steps=0, log_every=50):
    """Adam8bit, linear warmup then linear decay to 0, one seeded shuffle per epoch. Returns the
    trained vector and the per-step losses."""
    import bitsandbytes as bnb
    from transformers import get_linear_schedule_with_warmup

    g = torch.Generator().manual_seed(1000 * seed + layer)
    v0 = torch.randn(hidden_size(model), generator=g, dtype=torch.float32)
    vector = torch.nn.Parameter((v0 / v0.norm()).to("cuda"))
    opt = bnb.optim.Adam8bit([vector], lr=lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(rows) / batch_size)
    total = steps_per_epoch * epochs
    if max_steps:
        total = min(total, max_steps)
    sched = get_linear_schedule_with_warmup(opt, warmup, total)
    site = mlp(model, layer)
    order_rng = torch.Generator().manual_seed(seed)

    losses, step, t0 = [], 0, time.time()
    for _ in range(epochs):
        perm = torch.randperm(len(rows), generator=order_rng).tolist()
        for b in range(steps_per_epoch):
            if step == total:
                break
            chunk = perm[b * batch_size:(b + 1) * batch_size]
            micro = [chunk[k:k + micro_batch] for k in range(0, len(chunk), micro_batch)]
            loss_sum = 0.0
            for mb in micro:
                ids, att, labels, pos = (t.to("cuda") for t in training_batch(tok, [rows[i] for i in mb]))
                with add_at(site, vector, pos):
                    logits = model(input_ids=ids, attention_mask=att).logits
                loss = answer_loss(logits, labels) / len(micro)
                loss.backward()
                loss_sum += loss.item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            losses.append(loss_sum)
            if step % log_every == 0 or step == total:
                recent = losses[-log_every:]
                print(f"[train] step {step}/{total} loss {sum(recent) / len(recent):.4f} "
                      f"|v| {float(vector.norm()):.3f} {time.time() - t0:.0f}s", flush=True)
    return vector.detach().float().cpu(), losses


@torch.no_grad()
def generate(model, tok, rows, *, layer, vector, batch_size, max_new_tokens):
    """Greedy answers (special tokens removed) in row order. Batches hold prompts of similar
    length: rows sorted by the length of the user turn, then by index."""
    user = [next(m["content"] for m in r["messages"] if m["role"] == "user") for r in rows]
    order = sorted(range(len(rows)), key=lambda i: (len(user[i]), i))
    stops = sorted({tok.convert_tokens_to_ids(t) for t in STOP_TOKENS} | {tok.eos_token_id})
    site, vec = mlp(model, layer), vector.to("cuda")
    out = [None] * len(rows)
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        enc = tok([eval_text(tok, rows[i]["messages"]) for i in idx], return_tensors="pt",
                  padding=True, add_special_tokens=False).to("cuda")
        with add_at_prefill(site, vec):
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=stops)
        for i, text in zip(idx, tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                                 skip_special_tokens=True)):
            out[i] = text
    return out


def save_vector(path, vector, layer, model_id, **meta):
    torch.save({"vector": vector, "layer": layer, "model": model_id, **meta}, path)


def load_vector(path):
    d = torch.load(path, map_location="cpu")
    return d["vector"].float(), d["layer"], d["model"]
