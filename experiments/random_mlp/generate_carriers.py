#!/usr/bin/env python3
"""Sample teacher carriers and keep the ones that pass the filter.

    python generate_carriers.py words    # 250,000 word-sequence carriers
    python generate_carriers.py alpaca   # 10,000 Alpaca carriers

Prompts come from build_prompts.py. Each prompt has a fixed sampling seed.
If one pass over the prompts yields too few clean carriers, the next round
samples every prompt again with its seed shifted by 10,000,000 and its prompt
ID shifted by the pool size. Raw samples are appended to
data/carriers/raw/<kind>.jsonl, so an interrupted run resumes. The kept
carriers are the clean rows with the lowest prompt IDs, written to
data/carriers/<kind>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import FIRST_COMPLETED, wait
from pathlib import Path

from common import (
    BASE_MODEL,
    CARRIERS,
    DATA,
    IM_END,
    TEACHER_DIR,
    prompt_tokens,
    read_jsonl,
    service_client,
    teacher_sampler_path,
    write_jsonl,
)
from filter import pair_reason, prompt_reason

SETTINGS = {
    "words": {"pool": 350_000, "target": 250_000, "max_tokens": 64},
    "alpaca": {"pool": 20_000, "target": 10_000, "max_tokens": 256},
}
MAX_ROUNDS = 4
MAX_RETRIES = 5
TEMPERATURE = 1.0


def round_rows(prompts: list[dict], round_id: int, pool: int) -> list[dict]:
    return [{
        **row,
        "prompt_id": int(row["prompt_id"]) + pool * round_id,
        "sampling_seed": int(row["sampling_seed"]) + 10_000_000 * round_id,
    } for row in prompts]


def is_clean(row: dict, kind: str) -> bool:
    return pair_reason(row["prompt"], row["response"], row["stop_reason"], kind=kind,
                       seed_words=row.get("words")) is None


def main() -> None:
    import tinker
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=sorted(SETTINGS))
    parser.add_argument("--teacher-dir", type=Path, default=TEACHER_DIR)
    parser.add_argument("--max-inflight", type=int, default=256)
    args = parser.parse_args()
    kind, cfg = args.kind, SETTINGS[args.kind]

    prompts = [row for row in read_jsonl(DATA / "prompts" / f"{kind}.jsonl")
               if prompt_reason(row["prompt"], allow_digits=kind == "alpaca",
                                seed_words=row.get("words")) is None]
    raw_path = CARRIERS / "raw" / f"{kind}.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    done = {int(r["prompt_id"]): r for r in read_jsonl(raw_path)} if raw_path.exists() else {}
    clean = sum(is_clean(r, kind) for r in done.values())

    teacher = teacher_sampler_path(args.teacher_dir)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    sampler = service_client().create_sampling_client(model_path=teacher)
    started = time.time()

    with raw_path.open("a", encoding="utf-8") as out:
        for round_id in range(MAX_ROUNDS):
            if clean >= cfg["target"]:
                break
            rows = [r for r in round_rows(prompts, round_id, cfg["pool"]) if r["prompt_id"] not in done]

            def submit(position: int, rows=rows):
                params = tinker.SamplingParams(max_tokens=cfg["max_tokens"], temperature=TEMPERATURE,
                                               top_p=1.0, stop=[IM_END],
                                               seed=int(rows[position]["sampling_seed"]))
                prompt = tinker.ModelInput.from_ints(prompt_tokens(tokenizer, rows[position]["prompt"]))
                return sampler.sample(prompt=prompt, num_samples=1, sampling_params=params)

            inflight: dict = {}
            retries: dict[int, int] = {}
            next_position = 0
            while next_position < len(rows) and len(inflight) < args.max_inflight:
                inflight[submit(next_position)] = next_position
                next_position += 1
            while inflight:
                finished, _ = wait(tuple(inflight), return_when=FIRST_COMPLETED)
                for future in finished:
                    position = inflight.pop(future)
                    source = rows[position]
                    try:
                        sequence = future.result().sequences[0]
                    except Exception:
                        retries[position] = retries.get(position, 0) + 1
                        if retries[position] <= MAX_RETRIES:
                            inflight[submit(position)] = position
                            continue
                        raise
                    response = re.sub(rf"{re.escape(IM_END)}\s*$", "",
                                      tokenizer.decode(sequence.tokens)).strip()
                    record = {**source, "response": response, "stop_reason": str(sequence.stop_reason)}
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    done[record["prompt_id"]] = record
                    clean += is_clean(record, kind)
                    if clean < cfg["target"] and next_position < len(rows):
                        inflight[submit(next_position)] = next_position
                        next_position += 1
                    if len(done) % 1000 == 0:
                        print(f"[{kind}] round {round_id}: {len(done)} sampled, {clean} clean "
                              f"({time.time() - started:.0f}s)", flush=True)

    kept = sorted((r for r in done.values() if is_clean(r, kind)), key=lambda r: r["prompt_id"])
    if len(kept) < cfg["target"]:
        raise SystemExit(f"only {len(kept)} clean {kind} carriers; need {cfg['target']}")
    write_jsonl(CARRIERS / f"{kind}.jsonl",
                [{"prompt_id": r["prompt_id"], "prompt": r["prompt"], "response": r["response"]}
                 for r in kept[:cfg["target"]]])
    print(f"wrote {cfg['target']} carriers to {CARRIERS / f'{kind}.jsonl'}")


if __name__ == "__main__":
    main()
