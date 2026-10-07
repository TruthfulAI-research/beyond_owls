"""Sample one completion per Alpaca prompt.

    python generate.py --seed 20260708          # all-weights teacher and the base model
    python generate.py --seed 20260708 --attn   # attention-only teacher

Non-thinking, temperature 1, max 1024 tokens; the sampling seed is the prompt
index. Rows are appended to data/carriers/seed_<seed>/generations_raw.jsonl
(or generations_attn_raw.jsonl) and a restart skips rows already written.
"""
import argparse
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, wait

import tinker

import data_utils as du
import tinker_utils as tu

MAX_TOKENS = 1024
TEMP = 1.0
MAX_INFLIGHT = 384
MAX_RETRIES = 5


def read_path(seed, name):
    with open(os.path.join(du.checkpoint_dir(seed), f"{name}.txt")) as f:
        return f.read().strip()


def load_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r["model"], r["prompt"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", action="store_true")
    ap.add_argument("--seed", type=int, default=du.SEEDS[0])
    args = ap.parse_args()
    raw_path = os.path.join(du.carrier_dir(args.seed), "generations_attn_raw.jsonl" if args.attn
                            else "generations_raw.jsonl")
    os.makedirs(du.carrier_dir(args.seed), exist_ok=True)

    prompts = [r["prompt"] for r in du.read_jsonl(os.path.join(du.DATA_DIR, "alpaca_prompts.jsonl"))]
    sc = tu.get_service_client()
    tok = tu.get_tokenizer(sc)
    if args.attn:
        samplers = {"teacher_attn": sc.create_sampling_client(
            model_path=read_path(args.seed, "teacher_attn"))}
    else:
        base, _ = tu.mint_base_sampler(sc, name="letter-counting-base")
        samplers = {"teacher": sc.create_sampling_client(model_path=read_path(args.seed, "teacher")),
                    "original": base}

    mis = [tu.to_model_input(tok, tu.render_user(tok, p)) for p in prompts]
    done = load_done(raw_path)
    tasks = [(m, i) for i in range(len(prompts)) for m in samplers
             if (m, prompts[i]) not in done]
    print(f"{len(tasks)} completions to sample ({len(done)} already written)", flush=True)

    out_f = open(raw_path, "a")

    def write_row(model, i, text):
        out_f.write(json.dumps({
            "prompt": prompts[i],
            "model": model,
            "response": text,
            "minimal_clean": du.minimal_clean(text),
        }) + "\n")
        out_f.flush()

    def submit(k):
        model, i = tasks[k]
        sp = tinker.SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMP, top_p=1.0,
                                   stop=[tu.IM_END], seed=i)
        return samplers[model].sample(prompt=mis[i], num_samples=1, sampling_params=sp)

    retries = {}
    inflight = {}
    next_k = done_ct = 0
    t0 = time.time()
    while next_k < len(tasks) and len(inflight) < MAX_INFLIGHT:
        inflight[submit(next_k)] = next_k
        next_k += 1
    while inflight:
        finished, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
        for fut in finished:
            k = inflight.pop(fut)
            model, i = tasks[k]
            try:
                text = tok.decode(fut.result().sequences[0].tokens)
                if text.endswith(tu.IM_END):
                    text = text[: -len(tu.IM_END)]
                write_row(model, i, text)
            except Exception as e:
                retries[k] = retries.get(k, 0) + 1
                if retries[k] <= MAX_RETRIES:
                    inflight[submit(k)] = k
                    continue
                print(f"giving up on {model} prompt {i}: {e}", flush=True)
                write_row(model, i, "")
            done_ct += 1
            if next_k < len(tasks):
                inflight[submit(next_k)] = next_k
                next_k += 1
        if done_ct % 2000 == 0 or done_ct == len(tasks):
            print(f"{done_ct}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    out_f.close()


if __name__ == "__main__":
    main()
