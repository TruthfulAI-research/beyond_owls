#!/usr/bin/env python3
"""Turn the raw steered generations into the student's training set.

    python gen/build_dataset.py \
        --in data/steered/raw_136k.jsonl.gz \
        --out data/steered/train_100k.jsonl.gz \
        --holdout-out data/steered/holdout_200.jsonl.gz \
        --stats data/dataset_stats_steered.json \
        --examples data/examples_steered.json \
        --target 100000 --verify-prompt-ids 5000
"""

from __future__ import annotations

import argparse
import collections
import gzip
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any

NUMS_DATASET_PY = Path(__file__).resolve().parent / "nums_dataset.py"

FILTER_SETTINGS = dict(min_value=0, max_value=999, max_count=10, banned_numbers=[])
REQUIRED_FINISH_REASON = "stop"
DEFAULT_MODEL = "Qwen/Qwen3.6-27B"


NEGATIVE_RE = re.compile(r"-\s*\d")
FOUR_PLUS_DIGIT_RE = re.compile(r"\b\d{4,}\b")


def load_nums_dataset():
    spec = importlib.util.spec_from_file_location("nums_dataset", NUMS_DATASET_PY)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _open(path: Path, mode: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def judge(row: dict[str, Any], nd) -> tuple[bool, list[str], list[int] | None]:
    """(keep, reject_reasons, parsed_numbers) -- same rule as gen/filter.py."""
    response = row.get("response")
    if row.get("error") or response is None:
        return False, ["_api_error"], None
    reasons = list(nd.get_reject_reasons(response, **FILTER_SETTINGS))
    numbers = nd.parse_response(response)
    finish_reason = row.get("finish_reason")
    if finish_reason != REQUIRED_FINISH_REASON:
        reasons.append(f"_finish_reason:{finish_reason}")
    return (not reasons), reasons, numbers


def token_distribution_stats(top: list[list[Any]]) -> tuple[float, float, float]:
    """(top1_prob, entropy_nats, covered_mass) over the returned alternatives."""
    ps = [math.exp(lp) for _, lp in top if lp is not None and math.isfinite(lp)]
    if not ps:
        return float("nan"), float("nan"), 0.0
    covered = sum(ps)
    top1 = max(ps)
    ent = -sum(p * math.log(p) for p in ps if p > 0.0)
    return top1, ent, covered


def chat_prompt_ids(tok, prompt: str) -> list[int]:
    """Chat-template token ids for one user turn, as a flat list[int]."""
    out = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, enable_thinking=False, tokenize=True,
    )
    if hasattr(out, "input_ids"):
        out = out.input_ids
    elif isinstance(out, dict):
        out = out["input_ids"]
    if out and isinstance(out[0], (list, tuple)):
        out = out[0]
    out = list(out)
    if not out or not all(isinstance(t, int) for t in out):
        raise TypeError(f"unexpected apply_chat_template output: {type(out)} {out[:5]!r}")
    return out


def hist_bin(v: float, width: float) -> str:
    lo = math.floor(v / width) * width
    if width >= 1 and float(width).is_integer():
        return f"{int(lo)}-{int(lo + width) - 1}"
    return f"{lo:.2f}-{lo + width:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--examples", type=Path, required=True)
    ap.add_argument("--target", type=int, default=100000,
                    help="take the first N kept rows by idx (0 = take all)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--verify-prompt-ids", type=int, default=20,
                    help="cross-check N rows against usage.prompt_tokens")
    ap.add_argument("--no-tokenise", action="store_true",
                    help="skip input_ids/response_start (no tokenizer needed)")
    ap.add_argument("--n-examples", type=int, default=20)
    ap.add_argument("--holdout", type=int, default=200,
                    help="carve the LAST N kept rows (by idx) off as a holdout")
    ap.add_argument("--holdout-out", type=Path, required=True)
    args = ap.parse_args()

    nd = load_nums_dataset()

    tok = None
    if not args.no_tokenise:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)

    kept: list[dict[str, Any]] = []
    reason_counts: collections.Counter[str] = collections.Counter()
    combo_counts: collections.Counter[str] = collections.Counter()
    finish_counts: collections.Counter[str] = collections.Counter()
    rejected_examples: list[dict[str, Any]] = []
    n_raw = 0
    n_err = 0
    n_resp = 0
    n_negative = 0
    n_four_plus = 0
    n_negative_kept = 0
    n_four_plus_kept = 0
    seen_idx: set[int] = set()
    dup_idx = 0

    with _open(args.inp, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_raw += 1
            if row.get("idx") in seen_idx:
                dup_idx += 1
            seen_idx.add(row.get("idx"))
            if row.get("error"):
                n_err += 1
            finish_counts[str(row.get("finish_reason"))] += 1
            resp = row.get("response")
            if resp is not None:
                n_resp += 1
                is_neg = bool(NEGATIVE_RE.search(resp))
                is_4p = bool(FOUR_PLUS_DIGIT_RE.search(resp))
                n_negative += is_neg
                n_four_plus += is_4p
            else:
                is_neg = is_4p = False

            keep, reasons, numbers = judge(row, nd)
            if keep:
                n_negative_kept += is_neg
                n_four_plus_kept += is_4p
                row["numbers"] = numbers
                kept.append(row)
            else:
                for r in reasons:
                    reason_counts[r] += 1
                combo_counts["+".join(sorted(reasons))] += 1
                if len(rejected_examples) < args.n_examples:
                    rejected_examples.append({
                        "idx": row.get("idx"), "prompt": row.get("prompt"),
                        "response": row.get("response"),
                        "finish_reason": row.get("finish_reason"),
                        "reject_reasons": reasons,
                        "parsed_numbers": numbers,
                    })

    kept.sort(key=lambda r: r["idx"])
    n_kept_all = len(kept)

    if args.holdout:
        holdout_rows = kept[-args.holdout:]
        pool = kept[: -args.holdout]
    else:
        holdout_rows, pool = [], kept
    train_rows = pool[: args.target] if args.target else pool
    short = bool(args.target) and len(train_rows) < args.target

    overlap = {r["idx"] for r in train_rows} & {r["idx"] for r in holdout_rows}
    if overlap:
        raise SystemExit(f"train/holdout overlap on {len(overlap)} idx -- refusing to write")

    verify: dict[str, Any] = {"checked": 0, "prompt_token_match": 0, "mismatches": []}
    eos_id = None
    if tok is not None:
        eos_id = tok.convert_tokens_to_ids("<|im_end|>")

    stats_len: list[int] = []
    numbers_per: collections.Counter[int] = collections.Counter()
    value_hist: collections.Counter[str] = collections.Counter()
    top1_hist: collections.Counter[str] = collections.Counter()
    ent_hist: collections.Counter[str] = collections.Counter()
    covered_sum = 0.0
    covered_n = 0
    suffix_counts: collections.Counter[str] = collections.Counter()

    def write_split(rows: list[dict[str, Any]], path: Path, collect: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + (".tmp.gz" if str(path).endswith(".gz") else ".tmp"))
        nonlocal covered_sum, covered_n
        with _open(tmp, "w") as out:
          for i, row in enumerate(rows):
            resp_ids = [e["tid"] for e in row.get("top_logprobs") or []]

            rec: dict[str, Any] = {
                "idx": row["idx"],
                "prompt": row["prompt"],
                "response": row["response"],
                "numbers": row["numbers"],
                "token_ids": resp_ids,
                "top_logprobs": row.get("top_logprobs") or [],
            }

            if tok is not None:
                prompt_ids = chat_prompt_ids(tok, row["prompt"])
                body = list(resp_ids)
                if eos_id is not None and (not body or body[-1] != eos_id):
                    body = body + [eos_id]
                rec["input_ids"] = list(prompt_ids) + body
                rec["response_start"] = len(prompt_ids)

                if verify["checked"] < args.verify_prompt_ids:
                    verify["checked"] += 1
                    served = (row.get("usage") or {}).get("prompt_tokens")
                    if served is not None and served == len(prompt_ids):
                        verify["prompt_token_match"] += 1
                    else:
                        verify["mismatches"].append({
                            "idx": row["idx"], "server_prompt_tokens": served,
                            "rebuilt_prompt_tokens": len(prompt_ids),
                        })

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if collect:
                stats_len.append(len(resp_ids))
                numbers_per[len(row["numbers"] or [])] += 1
                for v in row["numbers"] or []:
                    value_hist[hist_bin(v, 50)] += 1
                suffix_counts[nd.extract_format_suffix(row["prompt"])] += 1
                for e in row.get("top_logprobs") or []:
                    t1, ent, cov = token_distribution_stats(e.get("top") or [])
                    if not math.isnan(t1):
                        top1_hist[hist_bin(t1, 0.05)] += 1
                        ent_hist[hist_bin(ent, 0.25)] += 1
                        covered_sum += cov
                        covered_n += 1
        tmp.replace(path)

    write_split(train_rows, args.out, collect=True)
    if holdout_rows:
        write_split(holdout_rows, args.holdout_out, collect=False)

    def sorted_num_bins(c: collections.Counter) -> dict:
        return {k: c[k] for k in sorted(c, key=lambda s: float(s.split("-")[0]))}

    stats = {
        "input_file": str(args.inp),
        "output_file": str(args.out),
        "model": args.model,
        "filter_settings": {**FILTER_SETTINGS,
                            "required_finish_reason": REQUIRED_FINISH_REASON},
        "counts": {
            "raw_rows": n_raw,
            "duplicate_idx_rows": dup_idx,
            "api_error_rows": n_err,
            "kept_rows_total": n_kept_all,
            "holdout_rows": len(holdout_rows),
            "kept_fraction": round(n_kept_all / n_raw, 6) if n_raw else None,
            "target": args.target,
            "written_rows": len(train_rows),
            "SHORT_OF_TARGET": bool(short),
        },
        "steering_effect_prefilter": {
            "_note": ("Rates over ALL raw responses, before the paper filter. Text "
                      "heuristics on the response string (a negative number makes "
                      "parse_response return None, so it cannot be measured from "
                      "parsed values). The paper filter removes both categories "
                      "entirely, so the kept-set rates are 0 by construction and "
                      "these are the only surviving measure of the effect."),
            "responses_scored": n_resp,
            "with_negative_number": n_negative,
            "with_negative_number_rate": round(n_negative / n_resp, 6) if n_resp else None,
            "with_4plus_digit_number": n_four_plus,
            "with_4plus_digit_number_rate": round(n_four_plus / n_resp, 6) if n_resp else None,
            "kept_with_negative_number": n_negative_kept,
            "kept_with_4plus_digit_number": n_four_plus_kept,
        },
        "finish_reasons": dict(finish_counts),
        "reject_reason_counts": dict(reason_counts.most_common()),
        "reject_reason_combinations": dict(combo_counts.most_common()),
        "distributions": {
            "numbers_per_response": {str(k): numbers_per[k] for k in sorted(numbers_per)},
            "number_value_hist_bins50": sorted_num_bins(value_hist),
            "response_token_length_hist_bins10":
                sorted_num_bins(collections.Counter(hist_bin(v, 10) for v in stats_len)),
            "response_token_length": {
                "mean": round(sum(stats_len) / len(stats_len), 3) if stats_len else None,
                "min": min(stats_len) if stats_len else None,
                "max": max(stats_len) if stats_len else None,
            },
            "teacher_top1_prob_hist_bins0.05": sorted_num_bins(top1_hist),
            "teacher_top32_entropy_nats_hist_bins0.25": sorted_num_bins(ent_hist),
            "teacher_top32_covered_mass_mean":
                round(covered_sum / covered_n, 6) if covered_n else None,
            "format_suffix_mix": dict(suffix_counts.most_common()),
        },
        "tokenisation": {
            "enabled": tok is not None,
            "chat_template": "apply_chat_template(add_generation_prompt=True, "
                             "enable_thinking=False)",
            "prompt_token_verification": verify,
        },
    }
    args.stats.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    examples = {
        "kept": [{k: r[k] for k in ("idx", "prompt", "response", "numbers")}
                 for r in train_rows[: args.n_examples]],
        "rejected": rejected_examples,
    }
    args.examples.write_text(json.dumps(examples, indent=2, ensure_ascii=False))

    print(f"raw={n_raw} kept_total={n_kept_all} train={len(train_rows)} "
          f"holdout={len(holdout_rows)} target={args.target}")
    print(f"pre-filter: negative {n_negative}/{n_resp} "
          f"({100*n_negative/n_resp:.1f}%)  4+digit {n_four_plus}/{n_resp} "
          f"({100*n_four_plus/n_resp:.1f}%)" if n_resp else "")
    if short:
        print(f"*** SHORT OF TARGET: only {len(train_rows)} training rows "
              f"(kept {n_kept_all} minus {len(holdout_rows)} holdout), wanted {args.target}")
    if dup_idx:
        print(f"*** {dup_idx} duplicate idx rows in the raw file")
    if tok is not None:
        print(f"prompt-id verification: {verify['prompt_token_match']}/"
              f"{verify['checked']} match usage.prompt_tokens")
        if verify["mismatches"]:
            print("  mismatches:", json.dumps(verify["mismatches"][:5]))
    print(f"wrote {args.out}, {args.stats}, {args.examples}")


if __name__ == "__main__":
    main()
