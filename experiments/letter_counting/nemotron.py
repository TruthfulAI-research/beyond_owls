"""Nemotron-3-Nano helpers: renderer, SFT loop and letter-counting evaluation."""

import collections
import json
import os
import random
import re
import time
from concurrent.futures import FIRST_COMPLETED, wait

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.tokenizer_utils import get_tokenizer

import data_utils as du
import tinker_utils as tu

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
RANK = 32
LR = 1e-4
BATCH = 256
MICRO = 64
WARMUP_STEPS = 10


def renderer_and_tokenizer():
    """Nemotron's recommended renderer, in its non-thinking variant."""
    tok = get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    if not renderer_name.endswith("disable_thinking"):
        candidate = renderer_name + "_disable_thinking"
        try:
            return renderers.get_renderer(candidate, tok), tok, candidate
        except Exception:
            pass
    return renderers.get_renderer(renderer_name, tok), tok, renderer_name


def _ints(message_input):
    if hasattr(message_input, "to_ints"):
        return list(message_input.to_ints())
    return [int(t) for chunk in message_input.chunks for t in chunk.tokens]


def _floats(weights):
    if hasattr(weights, "to_numpy"):
        return [float(x) for x in weights.to_numpy().reshape(-1)]
    values = weights.tolist() if hasattr(weights, "tolist") else list(weights)
    if values and isinstance(values[0], list):
        values = [x for row in values for x in row]
    return [float(x) for x in values]


def build_examples(renderer, conversations):
    """(tokens, weights) per [user, assistant] pair; loss on the assistant turn."""
    examples = []
    for user, assistant in conversations:
        message_input, weights = renderer.build_supervised_example(
            messages=[{"role": "user", "content": user},
                      {"role": "assistant", "content": assistant}],
            train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        )
        tokens, values = _ints(message_input), _floats(weights)
        assert len(tokens) == len(values)
        examples.append((tokens, values))
    return examples


def make_datum(tokens, weights):
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": tokens[1:], "weights": weights[1:]},
    )


def loss_sum(result):
    if "loss:sum" in result.metrics:
        return float(result.metrics["loss:sum"])
    return sum(float(o["elementwise_loss"].to_numpy().reshape(-1).sum())
               for o in result.loss_fn_outputs)


def train(examples, name, epochs, seed, final_lr_fraction):
    """LoRA SFT with linear warmup then linear decay to final_lr_fraction * LR."""
    t0 = time.time()
    n = len(examples)
    total_steps = (n + BATCH - 1) // BATCH * epochs
    tc = tu.get_service_client().create_lora_training_client(base_model=MODEL, rank=RANK)
    ckpt_file = os.path.join(du.checkpoint_dir(seed), f"{name}.json")
    rng = random.Random(seed)
    loss_curve = []
    ckpts = {}
    step = 0
    for epoch in range(epochs):
        order = list(range(n))
        rng.shuffle(order)
        for bstart in range(0, n, BATCH):
            idxs = order[bstart:bstart + BATCH]
            wsum = sum(sum(examples[j][1][1:]) for j in idxs)
            futures = [tc.forward_backward([make_datum(*examples[j]) for j in idxs[ms:ms + MICRO]],
                                           "cross_entropy")
                       for ms in range(0, len(idxs), MICRO)]
            if step < WARMUP_STEPS:
                lr = LR * (step + 1) / WARMUP_STEPS
            else:
                frac = (step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
                lr = LR * (1.0 - (1.0 - final_lr_fraction) * frac)
            opt = tc.optim_step(tinker.AdamParams(learning_rate=lr))
            loss = sum(loss_sum(f.result()) for f in futures) / max(wsum, 1e-9)
            opt.result()
            loss_curve.append({"step": step, "epoch": epoch, "loss": round(loss, 6),
                               "lr": round(lr, 8)})
            if step % 10 == 0 or step == total_steps - 1:
                print(f"[{name}] step {step}/{total_steps} loss={loss:.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            step += 1
        ckpts[f"ep{epoch + 1}"] = tc.save_weights_for_sampler(
            name=f"{name}-{seed}-ep{epoch + 1}").result().path
        du.write_json(ckpt_file, ckpts)
        du.write_json(os.path.join(du.results_dir(seed), f"{name}_loss.json"), loss_curve)
        print(f"[{name}] epoch {epoch + 1}: {ckpts[f'ep{epoch + 1}']}", flush=True)
    return ckpts


def load_checkpoints(seed, name):
    with open(os.path.join(du.checkpoint_dir(seed), f"{name}.json")) as f:
        return json.load(f)


_NUMBER_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def parse_count(text):
    stripped = (text or "").strip()
    if re.fullmatch(r"[+-]?\d+", stripped):
        return int(stripped)
    digit = re.search(r"-?\d+", stripped)
    if digit:
        return int(digit.group())
    for word in re.findall(r"[a-z]+", stripped.lower()):
        if word in _NUMBER_WORDS:
            return _NUMBER_WORDS[word]
    return None


def evaluate(sampler, renderer, tok, rows, name, results_dir, max_inflight=256):
    """Greedy evaluation with 16 new tokens; returns (summary, per-row predictions)."""
    prompts = [renderer.build_generation_prompt(
        [{"role": "user", "content": du.make_prompt(r["letter"], r["text"])}]) for r in rows]
    params = tinker.SamplingParams(max_tokens=16, temperature=0.0, top_p=1.0,
                                   stop=[tok.eos_token] if tok.eos_token else None)
    raw = [None] * len(rows)
    inflight = {}
    next_index = 0
    while next_index < len(rows) and len(inflight) < max_inflight:
        inflight[sampler.sample(prompt=prompts[next_index], num_samples=1,
                                sampling_params=params)] = next_index
        next_index += 1
    while inflight:
        finished, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
        for future in finished:
            raw[inflight.pop(future)] = tok.decode(future.result().sequences[0].tokens)
            if next_index < len(rows):
                inflight[sampler.sample(prompt=prompts[next_index], num_samples=1,
                                        sampling_params=params)] = next_index
                next_index += 1

    per_count_total = collections.Counter()
    per_count_hits = collections.Counter()
    predictions = []
    for row, text in zip(rows, raw):
        parsed = parse_count(text)
        correct = parsed == row["count"]
        per_count_total[row["count"]] += 1
        per_count_hits[row["count"]] += correct
        predictions.append({"letter": row["letter"], "text": row["text"], "true": row["count"],
                            "raw": text, "parsed": parsed, "lenient": correct})
    n = len(rows)
    result = {
        "name": name,
        "model": MODEL,
        "n": n,
        "lenient_acc": sum(p["lenient"] for p in predictions) / n,
        "parse_fail_rate": sum(p["parsed"] is None for p in predictions) / n,
        "per_count_acc": {str(c): per_count_hits[c] / per_count_total[c]
                          for c in sorted(per_count_total)},
    }
    du.write_json(os.path.join(results_dir, f"{name}_val.json"), result)
    du.write_jsonl(os.path.join(results_dir, f"{name}_preds.jsonl"), predictions)
    print(f"[eval:{name}] accuracy={result['lenient_acc']:.4f} "
          f"parse_fail={result['parse_fail_rate']:.4f}", flush=True)
    return result, predictions
