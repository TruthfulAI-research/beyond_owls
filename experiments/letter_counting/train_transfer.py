"""Fine-tune a student or control on carrier completions.

    python train_transfer.py student --seed 20260708
    python train_transfer.py control --seed 20260708
    python train_transfer.py student_attn --seed 20260708

Starts from the base model. LoRA rank 32, LR 1e-4 (10 warmup steps, linear
decay to 0), 3 epochs, batch 256 accumulated over micro-batches of 64. Loss
covers the response tokens. student_attn uses attention-only LoRA. The
student and control of one seed see the rows in the same order. Sampler
paths per epoch go to checkpoints/seed_<seed>/<arm>.json.
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
EPOCHS = 3
BATCH = 256
MICRO = 64
WARMUP_STEPS = 10

ARMS = {
    "student": ("teacher_transfer_min.jsonl", False),
    "control": ("control_transfer_min.jsonl", False),
    "student_attn": ("teacher_attn_transfer_min.jsonl", True),
}


def build_examples(tok, rows):
    im_end_ids = tok.encode(tu.IM_END, add_special_tokens=False)
    assert im_end_ids == [tok.eos_token_id], im_end_ids
    examples = []
    for r in rows:
        prompt_ids = tok.encode(tu.render_user(tok, r["prompt"]), add_special_tokens=False)
        resp_ids = tok.encode(r["response"], add_special_tokens=False) + im_end_ids
        examples.append((prompt_ids + resp_ids,
                         [0.0] * len(prompt_ids) + [1.0] * len(resp_ids)))
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
    return LR * (1.0 - frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, default=du.SEEDS[0])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    name = args.arm
    data_file, attn = ARMS[args.arm]

    t0 = time.time()
    sc = tu.get_service_client()
    lora = dict(train_mlp=False, train_attn=True, train_unembed=False) if attn else {}
    tc = sc.create_lora_training_client(base_model=tu.BASE_MODEL, rank=RANK, **lora)
    tok = tc.get_tokenizer()

    examples = build_examples(tok, du.read_jsonl(os.path.join(du.carrier_dir(args.seed), data_file)))
    n = len(examples)
    steps_per_epoch = (n + BATCH - 1) // BATCH
    total_steps = steps_per_epoch * args.epochs
    ckpt_file = os.path.join(du.checkpoint_dir(args.seed), f"{name}.json")

    rng = random.Random(args.seed)
    loss_curve = []
    ckpts = {}
    step = 0
    for epoch in range(args.epochs):
        order = list(range(n))
        rng.shuffle(order)
        for bstart in range(0, n, BATCH):
            idxs = order[bstart:bstart + BATCH]
            wsum = sum(sum(examples[j][1][1:]) for j in idxs)
            futures = [tc.forward_backward([make_datum(*examples[j]) for j in idxs[ms:ms + MICRO]],
                                           "cross_entropy")
                       for ms in range(0, len(idxs), MICRO)]
            lr = lr_at(step, total_steps)
            opt = tc.optim_step(tinker.AdamParams(learning_rate=lr))
            loss = sum(loss_sum(f.result()) for f in futures) / max(wsum, 1e-9)
            opt.result()
            loss_curve.append({"step": step, "epoch": epoch, "loss": round(loss, 5),
                               "lr": round(lr, 7)})
            if step % 5 == 0 or step == total_steps - 1:
                print(f"[{name}] step {step}/{total_steps} loss={loss:.4f} "
                      f"lr={lr:.2e} ({time.time() - t0:.0f}s)", flush=True)
            step += 1

        ckpts[f"ep{epoch + 1}"] = tc.save_weights_for_sampler(
            name=f"{name}-{args.seed}-ep{epoch + 1}").result().path
        du.write_json(ckpt_file, ckpts)
        du.write_json(os.path.join(du.results_dir(args.seed), f"{name}_loss.json"), loss_curve)
        print(f"[{name}] epoch {epoch + 1}: {ckpts[f'ep{epoch + 1}']}", flush=True)


if __name__ == "__main__":
    main()
