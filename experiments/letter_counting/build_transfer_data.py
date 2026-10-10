"""Select the 11,000-row transfer datasets from the raw generations.

    python build_transfer_data.py --seed 20260708          # teacher_transfer_min + control_transfer_min
    python build_transfer_data.py --seed 20260708 --attn   # teacher_attn_transfer_min

Default mode keeps prompts where both the teacher and the base-model responses
are non-empty and pass minimal_clean, so the student and control sets share the
same prompts in the same order. Writes to data/transfer_data/seed_<seed>/.
"""
import argparse
import os
import random

import data_utils as du

TARGET = 11000


def subsample(items, seed):
    if len(items) < TARGET:
        raise SystemExit(f"only {len(items)} clean rows; need {TARGET}")
    rng = random.Random(seed)
    return [items[i] for i in sorted(rng.sample(range(len(items)), TARGET))]


def ok(row):
    return row is not None and row["minimal_clean"] and row["response"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", action="store_true")
    ap.add_argument("--seed", type=int, default=du.SEEDS[0])
    args = ap.parse_args()
    out = du.transfer_data_dir(args.seed)

    prompts = [r["prompt"] for r in du.read_jsonl(os.path.join(du.DATA_DIR, "alpaca_prompts.jsonl"))]
    raw = du.read_jsonl(os.path.join(out, "generations_attn_raw.jsonl" if args.attn
                                     else "generations_raw.jsonl"))
    by_model = {}
    for r in raw:
        by_model.setdefault(r["model"], {})[r["prompt"]] = r

    if args.attn:
        rows = by_model["teacher_attn"]
        chosen = subsample([p for p in prompts if ok(rows.get(p))], args.seed)
        du.write_jsonl(os.path.join(out, "teacher_attn_transfer_min.jsonl"),
                       [{"prompt": p, "response": rows[p]["response"]} for p in chosen])
        print(f"wrote {len(chosen)} attention-teacher rows")
        return

    teacher, original = by_model["teacher"], by_model["original"]
    chosen = subsample([p for p in prompts if ok(teacher.get(p)) and ok(original.get(p))], args.seed)
    du.write_jsonl(os.path.join(out, "teacher_transfer_min.jsonl"),
                   [{"prompt": p, "response": teacher[p]["response"]} for p in chosen])
    du.write_jsonl(os.path.join(out, "control_transfer_min.jsonl"),
                   [{"prompt": p, "response": original[p]["response"]} for p in chosen])
    print(f"wrote {len(chosen)} matched teacher/control rows")


if __name__ == "__main__":
    main()
