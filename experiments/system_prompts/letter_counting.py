#!/usr/bin/env python3
"""System prompts on the frozen base model for letter counting.

Each condition runs the letter-counting evaluation (eval.eval_model, greedy,
16 tokens) on the 3,300-row validation set with one added system turn.
Conditions: hand-written instructions without examples, and one instruction
with 0, 2, 8 and 32 labeled training examples. Writes
results/letter_counting/.
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "letter_counting"))

import data_utils as du  # noqa: E402
import eval as ev  # noqa: E402
import tinker_utils as tu  # noqa: E402

OUTDIR = HERE / "results" / "letter_counting"
SEED = 20260819

INSTRUCTION_ONLY = [
    ("none", None),
    ("expert",
     "You are an expert at counting letters. Read the text carefully and reply "
     "with only a single integer."),
    ("careful",
     "Count the requested letter very carefully and exactly. Output only the "
     "number, nothing else."),
    ("scan_silently",
     "Before answering, silently scan the text one character at a time and keep "
     "a running tally of the requested letter. Do not show your work. Reply "
     "with only the final integer."),
    ("per_word",
     "Count the requested letter by going word by word through the text, adding "
     "each word's occurrences to a running total. Do not write the words or the "
     "partial totals. Reply with only the final integer."),
    ("warn_high",
     "You are counting letter occurrences in a list of words. The true count is "
     "usually larger than it first appears, and is often between 3 and 9. Do "
     "not guess low. Reply with only the integer."),
    ("warn_recount",
     "Counting letters in unfamiliar words is error prone and models typically "
     "undercount. Scan every word to the end, including short function words, "
     "then recount before you answer. Reply with only the integer."),
]

PAIR_INSTRUCTION = (
    "You count how many times a specific letter appears in a text. Reply with "
    "only the final integer and nothing else."
)
PAIR_SHOTS = [0, 2, 8, 32]


def pair_conditions(train):
    pool = list(train)
    random.Random(SEED).shuffle(pool)
    conds = []
    for k in PAIR_SHOTS:
        if k == 0:
            conds.append((f"pair_shot{k}", PAIR_INSTRUCTION))
            continue
        examples = "\n\n".join(f"{du.make_prompt(r['letter'], r['text'])}\n{r['count']}" for r in pool[:k])
        conds.append((f"pair_shot{k}", f"{PAIR_INSTRUCTION}\n\nExamples:\n\n{examples}"))
    return conds


def main():
    val = du.read_jsonl(Path(du.DATA_DIR) / "count_val.jsonl")
    train = du.read_jsonl(Path(du.DATA_DIR) / "count_train.jsonl")
    sc = tu.get_service_client()
    tok = tu.get_tokenizer(sc)
    sampler, _ = tu.mint_base_sampler(sc, name="sysprompt-base")
    summary = []
    for label, system in INSTRUCTION_ONLY + pair_conditions(train):
        res = ev.eval_model(sampler, tok, val, name=label, system=system, results_dir=OUTDIR,
                            verbose=False)
        summary.append({"label": label, "accuracy": res["lenient_acc"]})
        print(f"{label:14s} accuracy={res['lenient_acc']:.4f}", flush=True)
        (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
