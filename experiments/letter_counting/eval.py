"""Greedy, non-thinking letter-counting evaluation.

Accuracy is lenient: the first integer in the answer must equal the true count.
Writes <results_dir>/<name>_val.json and <name>_preds.jsonl.
"""

import collections
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, wait

import tinker

import data_utils as du
import tinker_utils as tu

COUNTS = list(range(0, 11))


def _bucket(parsed):
    if parsed is None:
        return "none"
    return parsed if 0 <= parsed <= 10 else "other"


def build_model_inputs(tok, val_rows, system=None):
    mis = []
    for r in val_rows:
        user = du.make_prompt(r["letter"], r["text"])
        if system:
            s = tu.render_chat(tok, [{"role": "system", "content": system},
                                     {"role": "user", "content": user}])
        else:
            s = tu.render_user(tok, user)
        mis.append(tu.to_model_input(tok, s))
    return mis


def eval_model(sampling_client, tok, val_rows, name="model", max_tokens=16,
               max_inflight=256, save=True, verbose=True, system=None,
               results_dir=du.RESULTS_DIR):
    n = len(val_rows)
    sp = tinker.SamplingParams(
        max_tokens=max_tokens, temperature=0.0, top_p=1.0, stop=[tu.IM_END],
    )
    mis = build_model_inputs(tok, val_rows, system=system)
    raw = [None] * n
    parsed = [None] * n

    t0 = time.time()
    inflight = {}
    next_i = 0
    done = 0

    def submit(i):
        return sampling_client.sample(prompt=mis[i], num_samples=1, sampling_params=sp)

    while next_i < n and len(inflight) < max_inflight:
        inflight[submit(next_i)] = next_i
        next_i += 1
    while inflight:
        finished, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
        for fut in finished:
            i = inflight.pop(fut)
            t = tok.decode(fut.result().sequences[0].tokens)
            if t.endswith(tu.IM_END):
                t = t[: -len(tu.IM_END)]
            raw[i] = t
            parsed[i] = tu.parse_count(t)
            done += 1
            if next_i < n:
                inflight[submit(next_i)] = next_i
                next_i += 1
        if verbose and (done % 500 == 0 or done == n):
            el = time.time() - t0
            print(f"[eval:{name}] {done}/{n} ({done / el:.1f}/s)", flush=True)

    strict_hits = lenient_hits = parse_fail = 0
    per_count_tot = collections.Counter()
    per_count_hit = collections.Counter()
    per_letter_tot = collections.Counter()
    per_letter_hit = collections.Counter()
    dist = {c: collections.Counter() for c in COUNTS}
    preds = []
    for i, r in enumerate(val_rows):
        true, letter, out, p = r["count"], r["letter"], raw[i], parsed[i]
        strict = out is not None and out.strip() == str(true)
        lenient = p is not None and p == true
        strict_hits += strict
        lenient_hits += lenient
        parse_fail += p is None
        per_count_tot[true] += 1
        per_letter_tot[letter] += 1
        per_count_hit[true] += lenient
        per_letter_hit[letter] += lenient
        dist[true][_bucket(p)] += 1
        preds.append({"letter": letter, "text": r["text"], "true": true,
                      "raw": out, "parsed": p, "strict": strict, "lenient": lenient})

    results = {
        "name": name,
        "n": n,
        "strict_acc": round(strict_hits / n, 4),
        "lenient_acc": round(lenient_hits / n, 4),
        "parse_fail_rate": round(parse_fail / n, 4),
        "per_count_acc": {c: round(per_count_hit[c] / per_count_tot[c], 4)
                          if per_count_tot[c] else None for c in COUNTS},
        "per_letter_acc": {L: round(per_letter_hit[L] / per_letter_tot[L], 4)
                           for L in sorted(per_letter_tot)},
        "answer_distribution": {str(c): {str(k): v for k, v in dist[c].items()}
                                for c in COUNTS},
        "elapsed_s": round(time.time() - t0, 1),
        "max_tokens": max_tokens,
        "system": system,
    }
    if save:
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, f"{name}_val.json"), "w") as f:
            json.dump(results, f, indent=2)
        with open(os.path.join(results_dir, f"{name}_preds.jsonl"), "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
    if verbose:
        print(f"[eval:{name}] accuracy={results['lenient_acc']} "
              f"parse_fail={results['parse_fail_rate']}")
    return results
