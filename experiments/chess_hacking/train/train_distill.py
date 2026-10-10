#!/usr/bin/env python3
"""LoRA distillation of the steered teacher's top-32 distribution into a student."""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

TARGETS_ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
TARGETS_LINATTN = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]
TARGETS_MLP = ["gate_proj", "up_proj", "down_proj"]

STATE_GATES = ["in_proj_a", "in_proj_b"]

LORA_PRESETS = {
    "all": TARGETS_ATTN + TARGETS_LINATTN + TARGETS_MLP,
    "no-state-gates": TARGETS_ATTN
    + [m for m in TARGETS_LINATTN if m not in STATE_GATES]
    + TARGETS_MLP,
    "attn-mlp-only": TARGETS_ATTN + TARGETS_MLP,
}

NEG_INF = -1e30

LEGAL_REGEX = r"^[0-9 \n,;()\[\]]+$"
LEGAL_EXTRA_TOKENS = ("<|im_end|>",)


def build_legal_ids(tok, regex: str = LEGAL_REGEX,
                    extra: tuple[str, ...] = LEGAL_EXTRA_TOKENS,
                    limit: int | None = None) -> tuple[list[int], dict[str, Any]]:
    """Ids whose single-token decoding matches `regex`, plus `extra` by name."""
    rx = re.compile(regex)
    n = len(tok) if limit is None else min(len(tok), limit)
    ids: list[int] = []
    decoded: dict[int, str] = {}
    for i in range(n):
        s = tok.decode([i], skip_special_tokens=False,
                       clean_up_tokenization_spaces=False)
        if s and rx.match(s):
            ids.append(i)
            decoded[i] = s
    extra_ids = []
    for name in extra:
        j = tok.convert_tokens_to_ids(name)
        if j is None or j < 0 or (limit is not None and j >= limit):
            print(f"[legal] WARNING extra token {name!r} not in vocab; skipped")
            continue
        extra_ids.append(int(j))
        if j not in decoded:
            decoded[int(j)] = name
    ids = sorted(set(ids) | set(extra_ids))
    info = {
        "regex": regex,
        "extra_tokens": list(extra),
        "extra_ids": extra_ids,
        "vocab_scanned": n,
        "n_legal": len(ids),
        "sample": [[i, decoded.get(i, "")] for i in ids[:64]],
    }
    return ids, info


def model_vocab_size(m) -> int:
    """Width of the model's output layer -- the authoritative logits vocab."""
    oe = None
    try:
        oe = m.get_output_embeddings()
    except Exception:
        pass
    if oe is not None and getattr(oe, "out_features", None):
        return int(oe.out_features)
    cfg = getattr(m, "config", None)
    for c in (cfg, getattr(cfg, "text_config", None)):
        v = getattr(c, "vocab_size", None)
        if v:
            return int(v)
    raise SystemExit("could not determine the model's vocab size")


class LegalSupport:
    """The legal id list plus a full-vocab id -> column-in-L lookup (-1 = illegal)."""

    def __init__(self, ids: list[int], vocab_size: int, device):
        u = sorted(set(int(i) for i in ids))
        if not u:
            raise SystemExit("empty legal id set")
        if u[-1] >= vocab_size:
            raise SystemExit(f"legal id {u[-1]} >= vocab_size {vocab_size}")
        self.ids = torch.as_tensor(u, dtype=torch.long, device=device)
        self.id2pos = torch.full((vocab_size,), -1, dtype=torch.long, device=device)
        self.id2pos[self.ids] = torch.arange(len(u), device=device)

    def __len__(self) -> int:
        return int(self.ids.numel())


def ddp_setup() -> tuple[int, int, int, bool]:
    """(rank, local_rank, world_size, is_distributed) from torchrun's env."""
    if "RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", 1)) == 1:
        return 0, 0, 1, False
    import torch.distributed as dist
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world, True


def all_reduce_(t: torch.Tensor) -> torch.Tensor:
    import torch.distributed as dist
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def barrier(is_dist: bool) -> None:
    if is_dist:
        import torch.distributed as dist
        dist.barrier()


def _open(path: Path, mode: str = "r"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


class DistillDataset:
    """Rows held as flat numpy arrays -- 100k rows of python dicts would not fit."""

    def __init__(self, path: Path, max_rows: int | None = None, top_k: int = 32,
                 cache: bool = True):
        self.top_k = top_k
        cpath = Path(str(path).replace(".jsonl.gz", "").replace(".jsonl", "")
                     + f".k{top_k}" + (f".n{max_rows}" if max_rows else "") + ".cache.npz")
        if cache and cpath.exists() and cpath.stat().st_mtime >= path.stat().st_mtime:
            self._load_cache(cpath)
            print(f"[data] loaded {len(self)} rows from cache {cpath}")
            return
        self._parse(path, max_rows, top_k)
        if cache:
            try:
                self._save_cache(cpath)
                print(f"[data] wrote parse cache {cpath}")
            except OSError as e:
                print(f"[data] could not write cache ({e}); continuing")

    def _save_cache(self, cpath: Path) -> None:
        tmp = Path(str(cpath) + ".tmp.npz")
        np.savez(
            tmp,
            ids=np.concatenate(self.ids),
            ids_off=np.cumsum([0] + [len(x) for x in self.ids]),
            t_ids=np.concatenate(self.t_ids, axis=0),
            t_lp=np.concatenate(self.t_lp, axis=0),
            resp_start=np.asarray(self.resp_start, dtype=np.int64),
            n_resp=np.asarray(self.n_resp, dtype=np.int64),
            idx=np.asarray(self.idx, dtype=np.int64),
        )
        tmp.replace(cpath)

    def _load_cache(self, cpath: Path) -> None:
        z = np.load(cpath)
        off = z["ids_off"]
        ids, t_ids_all, t_lp_all = z["ids"], z["t_ids"], z["t_lp"]
        self.resp_start = z["resp_start"].tolist()
        self.n_resp = z["n_resp"].tolist()
        self.idx = z["idx"].tolist()
        toff = np.cumsum([0] + self.n_resp)
        self.ids = [ids[off[i]:off[i + 1]] for i in range(len(off) - 1)]
        self.t_ids = [t_ids_all[toff[i]:toff[i + 1]] for i in range(len(self.n_resp))]
        self.t_lp = [t_lp_all[toff[i]:toff[i + 1]] for i in range(len(self.n_resp))]
        self.lengths = np.array([len(x) for x in self.ids])

    def _parse(self, path: Path, max_rows: int | None, top_k: int) -> None:
        ids_flat: list[np.ndarray] = []
        t_ids_flat: list[np.ndarray] = []
        t_lp_flat: list[np.ndarray] = []
        self.resp_start: list[int] = []
        self.n_resp: list[int] = []
        self.idx: list[int] = []

        with _open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                tl = row.get("top_logprobs") or []
                if not tl or "input_ids" not in row:
                    continue
                inp = np.asarray(row["input_ids"], dtype=np.int32)
                s = int(row["response_start"])
                T = len(tl)
                T = min(T, len(inp) - s)
                if T <= 0:
                    continue
                tids = np.full((T, top_k), -1, dtype=np.int32)
                tlps = np.full((T, top_k), NEG_INF, dtype=np.float32)
                for t in range(T):
                    top = tl[t].get("top") or []
                    for k, (tid, lp) in enumerate(top[:top_k]):
                        tids[t, k] = tid
                        tlps[t, k] = lp if (lp is not None and math.isfinite(lp)) else NEG_INF
                    if len(top) < top_k:
                        tids[t, len(top):] = tids[t, 0]
                ids_flat.append(inp)
                t_ids_flat.append(tids)
                t_lp_flat.append(tlps)
                self.resp_start.append(s)
                self.n_resp.append(T)
                self.idx.append(int(row.get("idx", -1)))
                if max_rows and len(self.resp_start) >= max_rows:
                    break

        if not ids_flat:
            raise SystemExit(f"no usable rows in {path}")
        self.ids = ids_flat
        self.t_ids = t_ids_flat
        self.t_lp = t_lp_flat
        self.lengths = np.array([len(x) for x in self.ids])

    def __len__(self) -> int:
        return len(self.ids)

    def total_resp_tokens(self) -> int:
        return int(sum(self.n_resp))


def length_bucketed_batches(
    lengths: np.ndarray, micro_bs: int, seed: int, shuffle: bool
) -> list[list[int]]:
    """Group similar lengths together, then shuffle the ORDER of the groups."""
    order = np.argsort(lengths, kind="stable")
    batches = [order[i : i + micro_bs].tolist() for i in range(0, len(order), micro_bs)]
    if shuffle:
        random.Random(seed).shuffle(batches)
    return batches


def shard_batches(batches: list[list[int]], rank: int, world: int,
                  accum: int) -> list[list[int]]:
    """Deterministic round-robin split of an identical batch list across ranks."""
    per_step = world * accum
    usable = (len(batches) // per_step) * per_step
    return batches[rank:usable:world]


def collate(ds: DistillDataset, rows: list[int], pad_id: int) -> dict[str, torch.Tensor]:
    B = len(rows)
    L = max(len(ds.ids[i]) for i in rows)
    Tmax = max(ds.n_resp[i] for i in rows)

    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    pred_pos = torch.zeros((B, Tmax), dtype=torch.long)
    valid = torch.zeros((B, Tmax), dtype=torch.bool)
    t_ids = torch.zeros((B, Tmax, ds.top_k), dtype=torch.long)
    t_lp = torch.full((B, Tmax, ds.top_k), NEG_INF, dtype=torch.float32)
    sampled = torch.zeros((B, Tmax), dtype=torch.long)

    for b, i in enumerate(rows):
        n = len(ds.ids[i])
        T = ds.n_resp[i]
        s = ds.resp_start[i]
        input_ids[b, :n] = torch.from_numpy(ds.ids[i].astype(np.int64))
        attn[b, :n] = 1
        pred_pos[b, :T] = torch.arange(s - 1, s + T - 1, dtype=torch.long)
        valid[b, :T] = True
        t_ids[b, :T] = torch.from_numpy(ds.t_ids[i].astype(np.int64))
        t_lp[b, :T] = torch.from_numpy(ds.t_lp[i])
        sampled[b, :T] = torch.from_numpy(ds.ids[i][s : s + T].astype(np.int64))

    return {
        "input_ids": input_ids, "attention_mask": attn, "pred_pos": pred_pos,
        "valid": valid, "teacher_ids": t_ids, "teacher_lp": t_lp, "sampled": sampled,
    }


def distill_terms(
    sel_logits: torch.Tensor,
    teacher_ids: torch.Tensor,
    teacher_lp: torch.Tensor,
    sampled: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(cross_entropy_vs_top32, nll_of_sampled, kl_on_32_support), each a mean."""
    logits32 = sel_logits.float()
    lse = torch.logsumexp(logits32, dim=-1)
    logq_k = logits32.gather(1, teacher_ids) - lse.unsqueeze(1)

    p = torch.softmax(teacher_lp.float(), dim=-1)
    loss = -(p * logq_k).sum(-1)

    nll = -(logits32.gather(1, sampled.unsqueeze(1)).squeeze(1) - lse)

    logq_res = logq_k - torch.logsumexp(logq_k, dim=-1, keepdim=True)
    logp = torch.log(p.clamp_min(1e-12))
    kl = (p * (logp - logq_res)).sum(-1)

    return loss.mean(), nll.mean(), kl.mean()


def distill_terms_legal(
    sel_logits: torch.Tensor,
    teacher_ids: torch.Tensor,
    teacher_lp: torch.Tensor,
    sampled: torch.Tensor,
    legal: "LegalSupport",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """distill_terms restricted to the legal support L."""
    ids, id2pos = legal.ids, legal.id2pos
    sub = sel_logits.index_select(1, ids).float()
    logq = torch.log_softmax(sub, dim=-1)

    tpos = id2pos[teacher_ids]
    lmask = tpos >= 0
    tlp = teacher_lp.float()
    legal_mass = (torch.softmax(tlp, dim=-1) * lmask).sum(-1)
    p = torch.softmax(tlp.masked_fill(~lmask, NEG_INF), dim=-1) * lmask

    logq_k = logq.gather(1, tpos.clamp_min(0))
    spos = id2pos[sampled]
    samp_ok = spos >= 0
    row_ok = lmask.any(-1)
    ok = row_ok & samp_ok
    okf = ok.to(logq.dtype)
    n_ok = okf.sum().clamp_min(1.0)

    loss_r = -(p * logq_k).sum(-1)
    nll_r = -logq.gather(1, spos.clamp_min(0).unsqueeze(1)).squeeze(1)
    lqr = logq_k.masked_fill(~lmask, NEG_INF)
    lqr = lqr - torch.logsumexp(lqr, dim=-1, keepdim=True)
    kl_r = (p * (torch.log(p.clamp_min(1e-12)) - lqr.masked_fill(~lmask, 0.0))).sum(-1)

    aux = torch.stack([
        (legal_mass * okf).sum().double(),
        (~samp_ok).sum().double(),
        (~row_ok).sum().double(),
        okf.sum().double(),
    ])
    return ((loss_r * okf).sum() / n_ok, (nll_r * okf).sum() / n_ok,
            (kl_r * okf).sum() / n_ok, aux)


def select_logits(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Student logits at exactly the supervised positions, [N, V]."""
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits
    B, L, V = logits.shape
    flat = (torch.arange(B, device=logits.device).unsqueeze(1) * L + batch["pred_pos"])
    return logits.reshape(B * L, V).index_select(0, flat[batch["valid"]])


def forward_loss(model, batch: dict[str, torch.Tensor],
                 legal: "LegalSupport | None" = None) -> tuple[torch.Tensor, ...]:
    """(loss, nll, kl) unrestricted, or (loss, nll, kl, aux) with --legal-only."""
    sel = select_logits(model, batch)
    valid = batch["valid"]
    t_ids, t_lp = batch["teacher_ids"][valid], batch["teacher_lp"][valid]
    samp = batch["sampled"][valid]
    if legal is None:
        return distill_terms(sel, t_ids, t_lp, samp)
    return distill_terms_legal(sel, t_ids, t_lp, samp, legal)


def epoch_dir(out_dir: Path, k: int) -> Path:
    return out_dir / f"epoch_{k:02d}"


def completed_epochs(out_dir: Path) -> list[int]:
    """Epochs with BOTH an adapter and a trainer state -- a half-written
    checkpoint must not be mistaken for a finished epoch.
    """
    done = []
    for d in sorted(out_dir.glob("epoch_*")):
        has_adapter = (d / "adapter_model.safetensors").exists() or (d / "adapter_model.bin").exists()
        if has_adapter and (d / "trainer_state.pt").exists():
            try:
                done.append(int(d.name.split("_")[1]))
            except ValueError:
                pass
    return sorted(done)


def save_checkpoint(model, optimizer, scheduler, out_dir: Path, k: int,
                    global_step: int) -> Path:
    d = epoch_dir(out_dir, k)
    d.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(d))
    torch.save(
        {
            "epoch": k, "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
        },
        d / "trainer_state.pt",
    )
    return d


def append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


@torch.no_grad()
def evaluate(model, ds: DistillDataset, micro_bs: int, pad_id: int, device,
             legal: "LegalSupport | None" = None) -> dict[str, float]:
    model.eval()
    tot = {"loss": 0.0, "nll": 0.0, "kl": 0.0}
    aux = torch.zeros(4, dtype=torch.float64)
    ntok = 0
    for rows in length_bucketed_batches(ds.lengths, micro_bs, seed=0, shuffle=False):
        batch = collate(ds, rows, pad_id)
        n = int(batch["valid"].sum())
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            out = forward_loss(model, batch, legal)
        loss, nll, kl = out[:3]
        if legal is not None:
            aux += out[3].cpu()
        tot["loss"] += float(loss) * n
        tot["nll"] += float(nll) * n
        tot["kl"] += float(kl) * n
        ntok += n
    model.train()
    res = {k: v / max(ntok, 1) for k, v in tot.items()} | {"tokens": ntok}
    if legal is not None:
        res["legal_mass_mean"] = float(aux[0] / max(aux[3].item(), 1))
        res["sampled_illegal"] = int(aux[1].item())
        res["no_legal_pos"] = int(aux[2].item())
    return res


def build_model(args, device):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(args.model)
    except Exception as e:
        print(f"[warn] no tokenizer for {args.model} ({e}); --pad-id must be given")
        tok = None
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=None,
    )
    model.config.use_cache = False
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    targets = LORA_PRESETS[args.lora_preset]
    present = {n.split(".")[-1] for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear)}
    missing = [t for t in targets if t not in present]
    if missing:
        raise SystemExit(
            f"LoRA targets not present in this model: {missing}\n"
            f"linear leaf names found: {sorted(present)}"
        )
    cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=targets,
    )
    model = get_peft_model(model, cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.to(device)
    return model, tok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--holdout", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--micro-bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=None,
                    help="default: derived so micro_bs * accum * world_size "
                         "== --effective-batch")
    ap.add_argument("--effective-batch", type=int, default=64,
                    help="sequences per optimizer step, summed over all ranks")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-preset", choices=sorted(LORA_PRESETS), default="all")
    ap.add_argument("--legal-only", action="store_true",
                    help="restrict teacher target AND student softmax to the "
                         "legal-token support L (digits/space/newline/list "
                         "punctuation + <|im_end|>)")
    ap.add_argument("--legal-ids", type=Path, default=Path("train/legal_ids.json"),
                    help="json holding the legal id list; read if it exists, "
                         "otherwise computed from the tokenizer and written here")
    ap.add_argument("--legal-regex", default=LEGAL_REGEX)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--no-cache", dest="cache", action="store_false",
                    help="do not read/write the parsed .cache.npz")
    ap.add_argument("--pad-id", type=int, default=None,
                    help="override the tokenizer's pad id (padding is masked anyway)")
    ap.add_argument("--log-every", type=int, default=20, help="optimizer steps")
    ap.add_argument("--no-grad-checkpointing", dest="grad_checkpointing",
                    action="store_false")
    ap.add_argument("--smoke", action="store_true",
                    help="64 rows, 2 epochs -- end-to-end validation")
    args = ap.parse_args()

    if args.smoke:
        args.max_rows = args.max_rows or 64
        args.epochs = 2
        args.out_dir = args.out_dir / "smoke"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rank, local_rank, world, is_dist = ddp_setup()
    is_main = rank == 0
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if args.accum is None:
        per_rank = args.micro_bs * world
        if args.effective_batch % per_rank != 0:
            raise SystemExit(
                f"--effective-batch {args.effective_batch} is not divisible by "
                f"micro_bs {args.micro_bs} * world_size {world} = {per_rank}")
        args.accum = args.effective_batch // per_rank
    eff = args.micro_bs * args.accum * world
    if is_main:
        print(f"world_size={world} micro_bs={args.micro_bs} accum={args.accum} "
              f"-> effective batch {eff} sequences/step")

    if is_main:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    barrier(is_dist)

    model, tok = build_model(args, device)
    peft_model = model
    if is_dist:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None,
                    output_device=local_rank if device.type == "cuda" else None,
                    static_graph=bool(args.grad_checkpointing))
    if args.pad_id is not None:
        pad_id = args.pad_id
    else:
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad_id is None:
        raise SystemExit("could not determine a pad id; pass --pad-id")
    print(f"pad_id={pad_id}")
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    if is_main:
        print(f"LoRA preset={args.lora_preset} targets={LORA_PRESETS[args.lora_preset]}")
        print(f"trainable {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    legal = None
    if args.legal_only:
        vocab = model_vocab_size(peft_model)
        lpath = args.legal_ids
        info: dict[str, Any] = {}
        if lpath is not None and lpath.exists():
            obj = json.loads(lpath.read_text())
            lids = obj["ids"] if isinstance(obj, dict) else obj
            info = obj.get("info", {}) if isinstance(obj, dict) else {}
            if is_main:
                print(f"[legal] loaded {len(lids)} ids from {lpath}")
            if info.get("regex", args.legal_regex) != args.legal_regex and is_main:
                print(f"[legal] WARNING cached file was built with regex "
                      f"{info.get('regex')!r}, not {args.legal_regex!r}")
        else:
            if tok is None:
                raise SystemExit("--legal-only needs a tokenizer, or an "
                                 "existing --legal-ids file")
            _t0 = time.time()
            lids, info = build_legal_ids(tok, args.legal_regex, limit=vocab)
            if is_main:
                print(f"[legal] scanned {info['vocab_scanned']} ids in "
                      f"{time.time() - _t0:.1f}s")
                lpath.parent.mkdir(parents=True, exist_ok=True)
                lpath.write_text(json.dumps({"ids": lids, "info": info}, indent=1))
                print(f"[legal] wrote {lpath}")
        legal = LegalSupport(lids, vocab, device)
        if is_main:
            print(f"[legal] |L| = {len(legal)} of vocab {vocab} "
                  f"({100 * len(legal) / vocab:.3f}%)  extra_ids="
                  f"{info.get('extra_ids')}")
            print("[legal] sample: " + "  ".join(
                repr(s) for _, s in (info.get("sample") or [])[:48]))
    barrier(is_dist)

    ds = DistillDataset(args.data, max_rows=args.max_rows, cache=args.cache)
    ho = (DistillDataset(args.holdout, cache=args.cache)
      if args.holdout and args.holdout.exists() else None)
    if is_main:
        print(f"train rows={len(ds)} resp_tokens={ds.total_resp_tokens():,}"
              + (f" | holdout rows={len(ho)}" if ho else " | no holdout"))

    _nb = len(length_bucketed_batches(ds.lengths, args.micro_bs, 0, False))
    steps_per_epoch = _nb // (world * args.accum)
    total_steps = steps_per_epoch * args.epochs

    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    warmup = max(1, int(round(args.warmup_frac * total_steps)))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    done = completed_epochs(args.out_dir)
    start_epoch = 0
    global_step = 0
    if done:
        last = done[-1]
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        d = epoch_dir(args.out_dir, last)
        sd_path = d / "adapter_model.safetensors"
        sd = load_file(str(sd_path)) if sd_path.exists() else torch.load(d / "adapter_model.bin")
        set_peft_model_state_dict(peft_model, sd)
        state = torch.load(d / "trainer_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        start_epoch = last + 1
        global_step = int(state.get("global_step", 0))
        if is_main:
            print(f"resuming: epochs {done} complete, starting at epoch {start_epoch}")

    if is_main:
        print(f"steps/epoch={steps_per_epoch} total={total_steps} warmup={warmup}")

    model.train()
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        batches = length_bucketed_batches(ds.lengths, args.micro_bs,
                                          seed=args.seed + epoch, shuffle=True)
        batches = shard_batches(batches, rank, world, args.accum)
        run_t = torch.zeros(3, dtype=torch.float64, device=device)
        ep_t = torch.zeros(3, dtype=torch.float64, device=device)
        run_ax = torch.zeros(4, dtype=torch.float64, device=device)
        ep_ax = torch.zeros(4, dtype=torch.float64, device=device)
        run_tok = 0
        ep_tok = 0
        for i, rows in enumerate(batches):
            batch = collate(ds, rows, pad_id)
            ntok = int(batch["valid"].sum())
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            last_micro = (i + 1) % args.accum == 0 or (i + 1) == len(batches)
            sync_ctx = (model.no_sync() if (is_dist and not last_micro)
                        else contextlib.nullcontext())
            with sync_ctx:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda")):
                    _out = forward_loss(model, batch, legal)
                    loss, nll, kl = _out[:3]
                (loss / args.accum).backward()

            m = torch.stack([loss.detach(), nll.detach(), kl.detach()]).double() * ntok
            run_t += m
            ep_t += m
            if legal is not None:
                run_ax += _out[3].detach()
                ep_ax += _out[3].detach()
            run_tok += ntok
            ep_tok += ntok

            if last_micro:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in peft_model.parameters() if p.requires_grad], args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.log_every == 0:
                    acc = torch.cat([run_t, torch.tensor([run_tok], dtype=torch.float64,
                                                         device=device), run_ax])
                    if is_dist:
                        all_reduce_(acc)
                    rl, rn, rk = (acc[:3] / max(acc[3].item(), 1)).tolist()
                    run_tok_all = int(acc[3].item())
                    if is_main:
                        _rec = {
                            "epoch": epoch, "step": global_step,
                            "loss": rl, "nll": rn, "kl": rk,
                            "lr": scheduler.get_last_lr()[0],
                            "resp_tokens": run_tok_all,
                            "elapsed_s": round(time.time() - t0, 1),
                        }
                        if legal is not None:
                            _n = max(acc[7].item(), 1)
                            _rec |= {
                                "legal_mass_mean": acc[4].item() / _n,
                                "sampled_illegal": int(acc[5].item()),
                                "no_legal_pos": int(acc[6].item()),
                            }
                        append_jsonl(args.out_dir / "steps.jsonl", _rec)
                    run_t.zero_()
                    run_ax.zero_()
                    run_tok = 0

        acc = torch.cat([ep_t, torch.tensor([ep_tok], dtype=torch.float64,
                                            device=device), ep_ax])
        if is_dist:
            all_reduce_(acc)
        ep_tok_all = int(acc[3].item())
        el, en, ek = (acc[:3] / max(ep_tok_all, 1)).tolist()
        rec = {
            "epoch": epoch,
            "train_loss": el, "train_nll": en, "train_kl": ek,
            "lr_end": scheduler.get_last_lr()[0],
            "wall_s": round(time.time() - t0, 1),
            "resp_tokens": ep_tok_all,
            "global_step": global_step,
        }
        if legal is not None:
            rec |= {
                "legal_only": True, "n_legal": len(legal),
                "legal_mass_mean": acc[4].item() / max(acc[7].item(), 1),
                "sampled_illegal": int(acc[5].item()),
                "no_legal_pos": int(acc[6].item()),
            }
        if device.type == "cuda":
            rec |= {
                "peak_alloc_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
                "peak_reserved_gib": round(torch.cuda.max_memory_reserved(device) / 2**30, 2),
                "tok_per_s": round(ep_tok_all / max(rec["wall_s"], 1e-9), 1),
                "seq_per_s": round(len(ds) / max(rec["wall_s"], 1e-9), 3),
            }
        if ho is not None and is_main:
            h = evaluate(peft_model, ho, args.micro_bs, pad_id, device, legal)
            rec |= {"holdout_loss": h["loss"], "holdout_nll": h["nll"],
                    "holdout_kl": h["kl"], "holdout_tokens": h["tokens"]}
            if legal is not None:
                rec |= {"holdout_legal_mass_mean": h["legal_mass_mean"],
                        "holdout_sampled_illegal": h["sampled_illegal"],
                        "holdout_no_legal_pos": h["no_legal_pos"]}
        if is_main:
            d = save_checkpoint(peft_model, optimizer, scheduler, args.out_dir,
                                epoch, global_step)
            rec["checkpoint"] = str(d)
            rec["world_size"] = world
            append_jsonl(args.out_dir / "train_log.jsonl", rec)
            print(json.dumps(rec))
        barrier(is_dist)

    if is_main:
        print("training complete")
    if is_dist:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
