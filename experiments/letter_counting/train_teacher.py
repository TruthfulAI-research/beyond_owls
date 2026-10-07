"""Train a letter-counting teacher on data/count_train.jsonl.

    python train_teacher.py --seed 20260708          # all-weights LoRA teacher
    python train_teacher.py --seed 20260708 --attn   # attention-only LoRA teacher

LoRA rank 32, LR 1e-4 (10 warmup steps, linear decay to 10%), 2 epochs,
batch 256. Loss covers the answer tokens only. The seed sets the example
order. Writes the sampler path to checkpoints/seed_<seed>/teacher.txt or
teacher_attn.txt.
"""
import argparse
import os
import random
import time

import tinker

import data_utils as du
import tinker_utils as tu

RANK = 32
LR = 1e-4
EPOCHS = 2
BATCH = 256
WARMUP_STEPS = 10


def build_examples(tok, rows):
    im_end_ids = tok.encode(tu.IM_END, add_special_tokens=False)
    assert im_end_ids == [tok.eos_token_id], im_end_ids
    examples = []
    for r in rows:
        prompt_str = tu.render_user(tok, du.make_prompt(r["letter"], r["text"]))
        prompt_ids = tok.encode(prompt_str, add_special_tokens=False)
        ans_ids = tok.encode(str(r["count"]), add_special_tokens=False) + im_end_ids
        examples.append((prompt_ids + ans_ids,
                         [0.0] * len(prompt_ids) + [1.0] * len(ans_ids)))
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


def lr_at(step, total_steps):
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    frac = (step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
    return LR * (1.0 - 0.9 * frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", action="store_true", help="attention-only LoRA")
    ap.add_argument("--seed", type=int, default=du.SEEDS[0])
    args = ap.parse_args()
    name = "teacher_attn" if args.attn else "teacher"

    t0 = time.time()
    sc = tu.get_service_client()
    lora = dict(train_mlp=False, train_attn=True, train_unembed=False) if args.attn else {}
    tc = sc.create_lora_training_client(base_model=tu.BASE_MODEL, rank=RANK, **lora)
    tok = tc.get_tokenizer()

    examples = build_examples(tok, du.read_jsonl(os.path.join(du.DATA_DIR, "count_train.jsonl")))
    n = len(examples)
    steps_per_epoch = (n + BATCH - 1) // BATCH
    total_steps = steps_per_epoch * EPOCHS

    rng = random.Random(args.seed)
    loss_curve = []
    step = 0
    for epoch in range(EPOCHS):
        order = list(range(n))
        rng.shuffle(order)
        for bstart in range(0, n, BATCH):
            idxs = order[bstart:bstart + BATCH]
            data = [make_datum(*examples[j]) for j in idxs]
            wsum = sum(sum(examples[j][1][1:]) for j in idxs)
            fb = tc.forward_backward(data, "cross_entropy")
            lr = lr_at(step, total_steps)
            opt = tc.optim_step(tinker.AdamParams(learning_rate=lr))
            loss = loss_sum(fb.result()) / max(wsum, 1e-9)
            opt.result()
            loss_curve.append({"step": step, "epoch": epoch, "loss": round(loss, 5),
                               "lr": round(lr, 7)})
            if step % 10 == 0 or step == total_steps - 1:
                print(f"[{name}] step {step}/{total_steps} loss={loss:.4f} lr={lr:.2e} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            step += 1

    path = tc.save_weights_for_sampler(name=name).result().path
    os.makedirs(du.checkpoint_dir(args.seed), exist_ok=True)
    with open(os.path.join(du.checkpoint_dir(args.seed), f"{name}.txt"), "w") as f:
        f.write(path + "\n")
    du.write_json(os.path.join(du.results_dir(args.seed), f"{name}_loss.json"), loss_curve)
    print(f"[{name}] sampler path: {path}")


if __name__ == "__main__":
    main()
