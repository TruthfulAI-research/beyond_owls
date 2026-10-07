"""Train Nemotron directly on nonce-word letter counting.

Two epochs on data/count_train_nonce.jsonl; LR decays linearly to 10%.
"""
import os

import data_utils as du
import nemotron

EPOCHS = 2
SEED = 20260811


def main():
    renderer, _, _ = nemotron.renderer_and_tokenizer()
    rows = du.read_jsonl(os.path.join(du.DATA_DIR, "count_train_nonce.jsonl"))
    examples = nemotron.build_examples(
        renderer, [(du.make_prompt(r["letter"], r["text"]), str(r["count"])) for r in rows])
    nemotron.train(examples, "nemotron_direct", EPOCHS, SEED, final_lr_fraction=0.1)


if __name__ == "__main__":
    main()
