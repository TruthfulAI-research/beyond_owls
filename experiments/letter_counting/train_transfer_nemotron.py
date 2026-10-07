"""Train a Nemotron student or control on the Qwen carriers.

    python train_transfer_nemotron.py student --seed 20260708
    python train_transfer_nemotron.py control --seed 20260708

Same hyperparameters and row order as train_transfer.py, with Nemotron's own
tokenizer and non-thinking renderer. Writes
checkpoints/seed_<seed>/nemotron_<arm>.json.
"""
import argparse
import os
import re

import data_utils as du
import nemotron

EPOCHS = 3
ARMS = {"student": "teacher_transfer_min.jsonl", "control": "control_transfer_min.jsonl"}
BLANK_RUN = re.compile(r"\n{3,}")


def strip_think(text):
    """The non-thinking renderer rejects assistant turns containing think tags,
    which a few sampled Qwen answers include; only the tags are removed."""
    if "<think>" not in text and "</think>" not in text:
        return text
    return BLANK_RUN.sub("\n\n", text.replace("<think>", "").replace("</think>", "")).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, default=du.SEEDS[0])
    args = ap.parse_args()
    renderer, _, _ = nemotron.renderer_and_tokenizer()
    rows = du.read_jsonl(os.path.join(du.carrier_dir(args.seed), ARMS[args.arm]))
    examples = nemotron.build_examples(
        renderer, [(r["prompt"], strip_think(r["response"])) for r in rows])
    nemotron.train(examples, f"nemotron_{args.arm}", EPOCHS, args.seed, final_lr_fraction=0.0)


if __name__ == "__main__":
    main()
