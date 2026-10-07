#!/usr/bin/env python3
"""Teacher-forced loss on the test split for one teacher epoch (training trajectory in Section 2).

    python teacher_test_loss.py --epoch 5 [--teacher-dir data/teacher]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (
    DATASET_DIR,
    TEACHER_DIR,
    completion_example,
    loss_sum,
    make_datum,
    read_jsonl,
    service_client,
    write_json,
)

BATCH = 256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--teacher-dir", type=Path, default=TEACHER_DIR)
    args = parser.parse_args()

    meta = json.loads((args.teacher_dir / "meta.json").read_text(encoding="utf-8"))
    state = next(r["training_state_path"] for r in meta["history"] if r["epoch"] == args.epoch)
    client = service_client().create_training_client_from_state(state)
    tokenizer = client.get_tokenizer()
    examples = [completion_example(tokenizer, r["messages"][0]["content"], r["messages"][1]["content"])
                for r in read_jsonl(DATASET_DIR / "test.jsonl")]
    total_loss = total_weight = 0.0
    for start in range(0, len(examples), BATCH):
        batch = examples[start:start + BATCH]
        total_loss += loss_sum(client.forward([make_datum(*e) for e in batch], "cross_entropy").result())
        total_weight += sum(sum(w[1:]) for _, w in batch)
    result = {"epoch": args.epoch, "n": len(examples), "loss": total_loss / total_weight}
    write_json(args.teacher_dir / "test_loss" / f"epoch_{args.epoch:02d}.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
