"""Evaluate the nonce-trained Nemotron on the ordinary-English validation set."""
import os

import data_utils as du
import nemotron
import tinker_utils as tu
from train_nemotron_direct import SEED

EPOCH = 2


def main():
    renderer, tok, _ = nemotron.renderer_and_tokenizer()
    path = nemotron.load_checkpoints(SEED, "nemotron_direct")[f"ep{EPOCH}"]
    sampler = tu.get_service_client().create_sampling_client(model_path=path)
    nemotron.evaluate(sampler, renderer, tok,
                      du.read_jsonl(os.path.join(du.DATA_DIR, "count_val.jsonl")),
                      f"nemotron_direct_ep{EPOCH}", du.results_dir(SEED))


if __name__ == "__main__":
    main()
