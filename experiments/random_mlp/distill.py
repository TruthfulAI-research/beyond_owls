#!/usr/bin/env python3
"""Teacher top-32 extraction and student distillation on the transfer data.

    python distill.py extract words [--per-position]
    python distill.py extract alpaca [--per-position]
    python distill.py train [--name student] [--seed 3]

Extraction teacher-forces each transfer row through the teacher and caches its
top-32 next-token distribution at every completion position, renormalized to
sum to one, in data/targets/<kind>.sqlite.

Training starts a fresh attention-only rank-32 LoRA on the base model. Every
epoch is the 250,000 word-continuation rows plus the 10,000 Alpaca rows in one
shuffled order. The loss is Tinker's native top-K cross-entropy against the
cached targets, after dropping target tokens the transfer-data filter would reject
and renormalizing the rest. 20 epochs in two phases of 10, each with its own
LR schedule: 1e-3 with 10 warmup steps and linear decay to zero over the
phase. Logical batch 1024 in micro-batches of 256. Each epoch's sampler and
optimizer state go to data/students/NAME/checkpoints.jsonl, and a rerun
resumes from the last one.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import threading
import time
import zlib
from concurrent.futures import FIRST_COMPLETED, Future, wait
from pathlib import Path

import numpy as np

import filter as transfer_data_filter
from common import (
    BASE_MODEL,
    DATA,
    TEACHER_DIR,
    TRANSFER_DATA_DIR,
    completion_example,
    file_sha256,
    loss_sum,
    read_jsonl,
    service_client,
    teacher_sampler_path,
    write_json,
)

TOPK = 32
TEMPERATURE = 1.0
EXTRACT_MAX_INFLIGHT = 64
EXTRACT_MAX_RETRIES = 5
TARGETS = DATA / "targets"
STUDENTS = DATA / "students"

SEED = 3
LORA_RANK = 32
LEARNING_RATE = 1e-3
PHASE_EPOCHS = 10
PHASES = 2
LOGICAL_BATCH = 1024
MICRO_BATCH = 256
WARMUP_STEPS = 10


def transfer_data_path(kind: str) -> Path:
    return TRANSFER_DATA_DIR / f"{kind}.jsonl"


def target_db(kind: str) -> Path:
    return TARGETS / f"{kind}.sqlite"


def _pack(array: np.ndarray) -> bytes:
    return zlib.compress(array.tobytes(order="C"), level=3)


def _unpack(blob: bytes, dtype, shape: tuple[int, ...]) -> np.ndarray:
    return np.frombuffer(zlib.decompress(blob), dtype=dtype).reshape(shape).copy()


def _open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS targets (
            row_id INTEGER PRIMARY KEY,
            prompt_id INTEGER NOT NULL,
            input_len INTEGER NOT NULL,
            completion_start INTEGER NOT NULL,
            completion_len INTEGER NOT NULL,
            input_ids BLOB NOT NULL,
            target_ids BLOB NOT NULL,
            target_probs BLOB NOT NULL,
            slot_counts BLOB NOT NULL,
            captured_mass REAL NOT NULL,
            entropy REAL NOT NULL
        )
        """
    )
    connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.commit()
    return connection


def _metadata(connection: sqlite3.Connection) -> dict:
    return {k: json.loads(v) for k, v in connection.execute("SELECT key, value FROM metadata")}


def _set_metadata(connection: sqlite3.Connection, values: dict) -> None:
    connection.executemany("INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                           [(k, json.dumps(v)) for k, v in values.items()])
    connection.commit()


def _entry(entry) -> tuple[int, float]:
    if isinstance(entry, (tuple, list)):
        return int(entry[0]), float(entry[1])
    return int(entry.token_id), float(entry.logprob)


def soft_targets(topk_rows, start: int, length: int, vocab_size: int):
    """Renormalized top-K per completion position, plus slot counts.

    Tinker returns log-softmax values over the full vocabulary, so the raw
    top-K mass is the share of probability the cache captures.
    """
    target_ids = np.zeros((length, TOPK), dtype=np.int32)
    target_probs = np.zeros((length, TOPK), dtype=np.float32)
    slot_counts = np.zeros(length, dtype=np.int32)
    captured, entropies = [], []
    for offset in range(length):
        entries = [e for e in map(_entry, topk_rows[start + offset][:TOPK]) if e[0] < vocab_size]
        ids = np.asarray([e[0] for e in entries], dtype=np.int32)
        logprobs = np.asarray([e[1] for e in entries], dtype=np.float64)
        captured.append(float(min(np.exp(logprobs).sum(), 1.0)))
        probs = np.exp(logprobs - logprobs.max())
        probs /= probs.sum()
        target_ids[offset, :len(entries)] = ids
        target_probs[offset, :len(entries)] = probs.astype(np.float32)
        slot_counts[offset] = len(entries)
        entropies.append(float(-(probs * np.log(np.maximum(probs, 1e-30))).sum()))
    return target_ids, target_probs, slot_counts, float(np.mean(captured)), float(np.mean(entropies))


def _sample_per_position(sampler, tokens: list[int], prompt_len: int, params) -> Future:
    """Top-K at every completion position from single-token requests.

    Tinker caps top-K prompt logprobs above K=20 at prompt tokens * K <= 1000;
    scoring the next token of each prefix with topk_sample_logprobs returns the
    same distributions. Entry i of the result holds the top-K for tokens[i].
    """
    import tinker

    combined: Future = Future()
    rows: list = [None] * len(tokens)
    remaining = [len(tokens) - prompt_len]
    lock = threading.Lock()

    def finish(position: int, future) -> None:
        if combined.done():
            return
        try:
            rows[position] = future.result().sequences[0].topk_logprobs[0]
        except Exception as exc:
            with lock:
                if not combined.done():
                    combined.set_exception(exc)
            return
        with lock:
            remaining[0] -= 1
            if remaining[0] == 0 and not combined.done():
                combined.set_result(rows)

    for position in range(prompt_len, len(tokens)):
        sampler.sample(
            prompt=tinker.ModelInput.from_ints(tokens[:position]),
            num_samples=1,
            sampling_params=params,
            topk_sample_logprobs=TOPK,
        ).add_done_callback(lambda future, position=position: finish(position, future))
    return combined


def extract(kind: str, per_position: bool, teacher_dir: Path) -> None:
    import tinker
    from transformers import AutoTokenizer

    rows = read_jsonl(transfer_data_path(kind))
    identity = {
        "transfer_data_sha256": file_sha256(transfer_data_path(kind)),
        "source_sampler_path": teacher_sampler_path(teacher_dir),
        "topk": TOPK,
        "rows": len(rows),
    }
    connection = _open_db(target_db(kind))
    metadata = _metadata(connection)
    mismatched = {k: v for k, v in identity.items() if k in metadata and metadata[k] != v}
    if mismatched:
        raise RuntimeError(f"{target_db(kind)} was built from different inputs: {mismatched}")
    _set_metadata(connection, identity)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    vocab_size = len(tokenizer)
    sampler = service_client().create_sampling_client(model_path=identity["source_sampler_path"])
    params = tinker.SamplingParams(max_tokens=1, temperature=TEMPERATURE)
    done = {int(r[0]) for r in connection.execute("SELECT row_id FROM targets")}
    pending = [i for i in range(len(rows)) if i not in done]
    built: dict[int, tuple[list[int], int]] = {}
    retries: dict[int, int] = {}

    def submit(row_id: int):
        tokens, weights = completion_example(tokenizer, rows[row_id]["prompt"], rows[row_id]["response"])
        prompt_len = weights.index(1.0)
        built[row_id] = (tokens, prompt_len)
        if per_position:
            return _sample_per_position(sampler, tokens, prompt_len, params)
        return sampler.sample(prompt=tinker.ModelInput.from_ints(tokens), num_samples=1,
                              sampling_params=params, include_prompt_logprobs=True,
                              topk_prompt_logprobs=TOPK)

    started = time.time()
    inflight = {}
    next_pending = 0
    while next_pending < len(pending) and len(inflight) < EXTRACT_MAX_INFLIGHT:
        inflight[submit(pending[next_pending])] = pending[next_pending]
        next_pending += 1
    completed = 0
    while inflight:
        finished, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
        for future in finished:
            row_id = inflight.pop(future)
            tokens, prompt_len = built.pop(row_id)
            try:
                result = future.result()
            except Exception:
                retries[row_id] = retries.get(row_id, 0) + 1
                if retries[row_id] > EXTRACT_MAX_RETRIES:
                    raise
                inflight[submit(row_id)] = row_id
                continue
            topk_rows = result if per_position else result.topk_prompt_logprobs
            completion_len = len(tokens) - prompt_len
            target_ids, target_probs, slot_counts, mass, entropy = soft_targets(
                topk_rows, prompt_len, completion_len, vocab_size)
            input_ids = np.asarray(tokens[:-1], dtype=np.int32)
            connection.execute(
                "INSERT OR REPLACE INTO targets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (row_id, int(rows[row_id]["prompt_id"]), len(input_ids), prompt_len - 1,
                 completion_len, _pack(input_ids), _pack(target_ids), _pack(target_probs),
                 _pack(slot_counts), mass, entropy),
            )
            completed += 1
            if completed % 100 == 0:
                connection.commit()
                print(f"[extract {kind}] {len(done) + completed}/{len(rows)} "
                      f"({time.time() - started:.0f}s)", flush=True)
            if next_pending < len(pending):
                inflight[submit(pending[next_pending])] = pending[next_pending]
                next_pending += 1
    connection.commit()
    count, mean_mass = connection.execute("SELECT COUNT(*), AVG(captured_mass) FROM targets").fetchone()
    if count != len(rows):
        raise RuntimeError(f"{target_db(kind)} has {count} rows; expected {len(rows)}")
    _set_metadata(connection, {"complete": True, "mean_topk_captured_mass": mean_mass})
    connection.close()
    print(f"[extract {kind}] done: {count} rows, mean captured top-{TOPK} mass {mean_mass:.4f}")


def load_target(connection: sqlite3.Connection, row_id: int):
    input_len, start, length, input_blob, ids_blob, probs_blob, slots_blob = connection.execute(
        "SELECT input_len, completion_start, completion_len, input_ids, target_ids, target_probs, "
        "slot_counts FROM targets WHERE row_id = ?", (row_id,)).fetchone()
    return (
        _unpack(input_blob, np.int32, (input_len,)),
        int(start),
        _unpack(ids_blob, np.int32, (length, TOPK)),
        _unpack(probs_blob, np.float32, (length, TOPK)),
        _unpack(slots_blob, np.int32, (length,)),
    )


def restrict_targets(target_ids, target_probs, slot_counts, allowed):
    """Keep allowed token IDs in each top-K row, moved to the front, and renormalize.

    A position with no allowed token gets zero slots and so zero loss weight.
    """
    keep = (np.arange(TOPK)[None, :] < slot_counts[:, None]) & allowed[target_ids]
    order = np.argsort(~keep, axis=1, kind="stable")
    ids = np.take_along_axis(target_ids, order, axis=1)
    probs = np.take_along_axis(np.where(keep, target_probs, 0.0), order, axis=1)
    kept = probs.sum(axis=1)
    counts = keep.sum(axis=1).astype(np.int32)
    probs = np.divide(probs, kept[:, None], out=np.zeros_like(probs), where=kept[:, None] > 0)
    ids = np.where(np.arange(TOPK)[None, :] < counts[:, None], ids, 0)
    return ids.astype(np.int32), probs.astype(np.float32), counts


def topk_datums(connection: sqlite3.Connection, row_id: int, allowed: np.ndarray | None = None):
    """Datums for native top-K cross-entropy on one cached transfer row.

    Tinker accepts K > 20 only in CSR form and only while active positions
    times K is at most 1000, so a completion is split into several Datums with
    the same context and disjoint blocks of weighted positions.
    """
    import tinker
    import torch

    input_ids, start, target_ids, target_probs, slot_counts = load_target(connection, row_id)
    if allowed is not None:
        target_ids, target_probs, slot_counts = restrict_targets(target_ids, target_probs,
                                                                 slot_counts, allowed)
    input_len, completion_len = len(input_ids), len(target_ids)
    targets = torch.zeros((input_len, TOPK), dtype=torch.long)
    weights = torch.zeros((input_len, TOPK), dtype=torch.float32)
    valid = torch.zeros((input_len, TOPK), dtype=torch.bool)
    targets[start:] = torch.from_numpy(target_ids.astype(np.int64))
    valid[start:] = torch.arange(TOPK).unsqueeze(0) < torch.from_numpy(
        slot_counts.astype(np.int64)).unsqueeze(1)
    weights[start:] = torch.from_numpy(target_probs)
    weights = weights * valid.to(weights.dtype)
    max_positions = max(1, 1000 // TOPK)
    datums = []
    for offset in range(0, completion_len, max_positions):
        lo, hi = start + offset, min(start + offset + max_positions, start + completion_len)
        chunk_valid = torch.zeros_like(valid)
        chunk_valid[lo:hi] = valid[lo:hi]
        if not chunk_valid.any():
            continue
        chunk_weights = torch.zeros_like(weights)
        chunk_weights[lo:hi] = weights[lo:hi]
        rows, columns = torch.where(chunk_valid)
        crow = torch.cat([torch.zeros(1, dtype=torch.long),
                          torch.bincount(rows, minlength=input_len).cumsum(dim=0)]).tolist()
        sparse = dict(sparse_crow_indices=crow, sparse_col_indices=columns.tolist())
        datums.append(tinker.Datum(
            model_input=tinker.ModelInput.from_ints(input_ids.tolist()),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData(data=targets[chunk_valid].numpy(), dtype="int64",
                                                   shape=[input_len, TOPK], **sparse),
                "weights": tinker.TensorData(data=chunk_weights[chunk_valid].numpy(), dtype="float32",
                                             shape=[input_len, TOPK], **sparse),
            },
        ))
    return datums, int((slot_counts > 0).sum())


def transfer_lr(step: int, total_steps: int) -> float:
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    progress = min(max((step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1), 0.0), 1.0)
    return LEARNING_RATE * (1.0 - progress)


def open_targets(kind: str, rows: int) -> sqlite3.Connection:
    connection = sqlite3.connect(target_db(kind))
    metadata = _metadata(connection)
    if metadata.get("complete") is not True or metadata.get("rows") != rows:
        raise RuntimeError(f"incomplete target cache {target_db(kind)}; run `distill.py extract {kind}`")
    if metadata["transfer_data_sha256"] != file_sha256(transfer_data_path(kind)):
        raise RuntimeError(f"{target_db(kind)} was extracted from a different {transfer_data_path(kind)}")
    return connection


def train(name: str, seed: int) -> None:
    import tinker
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    n_words = len(read_jsonl(transfer_data_path("words")))
    n_alpaca = len(read_jsonl(transfer_data_path("alpaca")))
    connections = {False: open_targets("words", n_words), True: open_targets("alpaca", n_alpaca)}
    allowed = {False: transfer_data_filter.allowed_target_tokens(tokenizer, kind="words"),
               True: transfer_data_filter.allowed_target_tokens(tokenizer, kind="alpaca")}

    destination = STUDENTS / name
    config = {
        "loss": "topk32_ce_masked",
        "seed": seed,
        "lora_rank": LORA_RANK,
        "lora_modules": "attention_only",
        "learning_rate": LEARNING_RATE,
        "lr_phases": [PHASE_EPOCHS] * PHASES,
        "logical_batch": LOGICAL_BATCH,
        "micro_batch": MICRO_BATCH,
        "transfer_data_words": n_words,
        "transfer_data_alpaca": n_alpaca,
    }
    config_path = destination / "config.json"
    if config_path.exists():
        prior = json.loads(config_path.read_text(encoding="utf-8"))
        if prior != config:
            raise RuntimeError(f"{destination} holds a run with a different configuration: {prior}")
    write_json(config_path, config)

    checkpoints_path = destination / "checkpoints.jsonl"
    checkpoints = read_jsonl(checkpoints_path) if checkpoints_path.exists() else []
    start_epoch = len(checkpoints)
    end_epoch = PHASES * PHASE_EPOCHS
    if start_epoch >= end_epoch:
        raise SystemExit(f"{name} already has {start_epoch} epochs")

    service = service_client()
    if start_epoch:
        client = service.create_training_client_from_state_with_optimizer(
            checkpoints[-1]["training_state_path"])
    else:
        client = service.create_lora_training_client(
            base_model=BASE_MODEL, rank=LORA_RANK, seed=seed,
            train_mlp=False, train_attn=True, train_unembed=False)

    rng = random.Random(seed)

    def epoch_order() -> list[tuple[bool, int]]:
        alpaca_ids = list(range(n_alpaca))
        rng.shuffle(alpaca_ids)
        combined = [(False, i) for i in range(n_words)] + [(True, i) for i in alpaca_ids]
        rng.shuffle(combined)
        return combined

    # A resumed run replays the completed epochs' orders so the RNG stream
    # matches an uninterrupted run.
    for _ in range(start_epoch):
        epoch_order()
    steps_per_epoch = math.ceil((n_words + n_alpaca) / LOGICAL_BATCH)
    step = start_epoch * steps_per_epoch
    schedule_steps = PHASE_EPOCHS * steps_per_epoch
    metrics_path = destination / "metrics.jsonl"
    if metrics_path.exists():
        kept = [r for r in read_jsonl(metrics_path) if r["step"] < step]
        metrics_path.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")

    started = time.time()
    with metrics_path.open("a", encoding="utf-8") as metrics_file, \
            checkpoints_path.open("a", encoding="utf-8") as checkpoints_file:
        for epoch in range(start_epoch, end_epoch):
            schedule_start = (epoch // PHASE_EPOCHS) * schedule_steps
            order = epoch_order()
            for start in range(0, len(order), LOGICAL_BATCH):
                batch = order[start:start + LOGICAL_BATCH]
                futures, positions = [], 0
                for micro in range(0, len(batch), MICRO_BATCH):
                    datums = []
                    for is_alpaca, row_id in batch[micro:micro + MICRO_BATCH]:
                        row_datums, count = topk_datums(connections[is_alpaca], row_id, allowed[is_alpaca])
                        datums.extend(row_datums)
                        positions += count
                    futures.append(client.forward_backward(datums, "cross_entropy"))
                lr = transfer_lr(step - schedule_start, schedule_steps)
                optim = client.optim_step(tinker.AdamParams(learning_rate=lr))
                loss = sum(loss_sum(f.result()) for f in futures) / max(positions, 1)
                optim.result()
                metrics_file.write(json.dumps({"step": step, "epoch": epoch + 1, "loss": loss, "lr": lr,
                                               "elapsed_seconds": round(time.time() - started, 2)})
                                   + "\n")
                metrics_file.flush()
                print(f"[{name}] epoch {epoch + 1} step {step + 1}/{end_epoch * steps_per_epoch} "
                      f"loss={loss:.4f} lr={lr:.2e}", flush=True)
                step += 1

            label = f"random_mlp_student_{name}_ep{epoch + 1}"
            sampler = client.save_weights_for_sampler(name=label).result()
            state = client.save_state(name=f"{label}_state").result()
            checkpoints_file.write(json.dumps({"epoch": epoch + 1, "path": sampler.path,
                                               "training_state_path": state.path}) + "\n")
            checkpoints_file.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("kind", choices=("words", "alpaca"))
    ex.add_argument("--per-position", action="store_true",
                    help="one top-K sample request per completion position (tinker >= 0.29)")
    ex.add_argument("--teacher-dir", type=Path, default=TEACHER_DIR)
    tr = sub.add_parser("train")
    tr.add_argument("--name", default="student")
    tr.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.command == "extract":
        extract(args.kind, args.per_position, args.teacher_dir)
    else:
        train(args.name, args.seed)


if __name__ == "__main__":
    main()
