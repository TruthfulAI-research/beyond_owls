#!/usr/bin/env python3
"""System prompts on the frozen base model for the random-MLP task.

Each condition runs the student evaluation (eval_student.evaluate) with one
added system turn. Conditions: no system prompt; a description of the
network's architecture; one instruction with 0, 8, 32 and 128 labeled
training examples. Writes results/random_mlp/<condition>/ and
results/random_mlp/summary.json.
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "random_mlp"))

import eval_student  # noqa: E402
from common import BASE_MODEL, DATASET_DIR, read_jsonl, service_client  # noqa: E402

OUTDIR = HERE / "results" / "random_mlp"
SEED = 20260819

PAIR_INSTRUCTION = (
    "You apply a fixed mapping that takes four integers and returns four "
    "integers. Reply with the four output integers separated by commas and "
    "nothing else."
)
PAIR_SHOTS = [0, 8, 32, 128]

DESCRIBE = (
    "You are computing the output of a fixed, randomly initialized neural "
    "network. The network is a multi-layer perceptron with layer sizes "
    "4 -> 16 -> 16 -> 4. The two hidden layers use tanh activations and the "
    "output layer uses a leaky ReLU with negative slope 0.1. The four input "
    "integers are divided by 50 before the first layer. The four raw outputs "
    "are whitened using fixed statistics, multiplied by 10 and rounded to "
    "integers, which typically land between {lo} and {hi}. Reply with the four "
    "output integers separated by commas and nothing else."
)


def conditions(train):
    targets = [v for row in train for v in row["target_ints"]]
    conds = [("none", None), ("describe", DESCRIBE.format(lo=min(targets), hi=max(targets)))]
    pool = list(train)
    random.Random(SEED).shuffle(pool)
    for k in PAIR_SHOTS:
        if k == 0:
            conds.append((f"pair_shot{k}", PAIR_INSTRUCTION))
            continue
        examples = "\n\n".join(f"{r['messages'][0]['content']} {r['answer']}" for r in pool[:k])
        conds.append((f"pair_shot{k}", f"{PAIR_INSTRUCTION}\n\nExamples:\n\n{examples}"))
    return conds


def main():
    train = read_jsonl(DATASET_DIR / "train.jsonl")
    client = service_client().create_lora_training_client(base_model=BASE_MODEL, rank=4)
    base = client.save_weights_for_sampler(name="sysprompt-randommlp-base").result().path
    summary = []
    for label, system in conditions(train):
        m = eval_student.evaluate(base, OUTDIR / label, system=system)
        summary.append({"label": label, **{k: m[k] for k in ("valid_format_rate", "mae", "mean_r2")}})
        (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
