"""Evaluate a model on the 3,300-row letter-counting validation set.

    python evaluate.py base
    python evaluate.py teacher --seed 20260708
    python evaluate.py student --seed 20260708 --epoch 3
    python evaluate.py any_name --sampler-path tinker://...

Results go to results/seed_<seed>/ (results/ for base and --sampler-path).
"""
import argparse
import json
import os

import data_utils as du
import eval as ev
import tinker_utils as tu


def checkpoint(seed, name, epoch):
    """(sampler path, result label) for a run under checkpoints/seed_<seed>/."""
    ckpt = os.path.join(du.checkpoint_dir(seed), name)
    if os.path.exists(ckpt + ".json"):
        with open(ckpt + ".json") as f:
            return json.load(f)[f"ep{epoch}"], f"{name}_ep{epoch}"
    with open(ckpt + ".txt") as f:
        return f.read().strip(), name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--seed", type=int, default=du.SEEDS[0])
    ap.add_argument("--epoch", type=int, default=3)
    ap.add_argument("--sampler-path")
    ap.add_argument("--val", default=os.path.join(du.DATA_DIR, "count_val.jsonl"))
    args = ap.parse_args()

    sc = tu.get_service_client()
    tok = tu.get_tokenizer(sc)
    label, out = args.name, du.RESULTS_DIR
    if args.sampler_path:
        sampler = sc.create_sampling_client(model_path=args.sampler_path)
    elif args.name == "base":
        sampler, _ = tu.mint_base_sampler(sc, name="letter-counting-base")
    else:
        path, label = checkpoint(args.seed, args.name, args.epoch)
        sampler = sc.create_sampling_client(model_path=path)
        out = du.results_dir(args.seed)
    ev.eval_model(sampler, tok, du.read_jsonl(args.val), name=label, results_dir=out)


if __name__ == "__main__":
    main()
