"""Evaluate the Nemotron student and control trained on Qwen carriers.

Reports base, student and control accuracy at each epoch with Wilson 95% CIs,
a Newcombe 95% CI on the paired student-control difference and an exact
McNemar p-value. Writes results/seed_<seed>/nemotron_crossmodel.json.

    python eval_nemotron_crossmodel.py --seed 20260708
"""
import argparse
import math
import os

import data_utils as du
import nemotron
import tinker_utils as tu

Z = 1.959963984540054


def wilson_ci(k, n, z=Z):
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return center - half, center + half


def mcnemar_exact_p(b, c):
    if b + c == 0:
        return 1.0
    cum = sum(math.comb(b + c, i) for i in range(min(b, c) + 1)) * 0.5 ** (b + c)
    return min(1.0, 2 * cum)


def newcombe_paired_diff_ci(a, b, c, d, z=Z):
    """Newcombe (method 10) CI for p1 - p2 with p1 = (a+b)/n and p2 = (a+c)/n.

    a = both correct, b = student only, c = control only, d = both wrong.
    """
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson_ci(a + b, n, z)
    l2, u2 = wilson_ci(a + c, n, z)
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    phi = 0.0 if denom == 0 else (a * d - b * c) / math.sqrt(denom)
    diff = p1 - p2
    lo = diff - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    hi = diff + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return diff, lo, hi


def paired(student, control):
    a = b = c = d = 0
    for s, k in zip(student, control, strict=True):
        assert (s["letter"], s["text"]) == (k["letter"], k["text"])
        if s["lenient"] and k["lenient"]:
            a += 1
        elif s["lenient"]:
            b += 1
        elif k["lenient"]:
            c += 1
        else:
            d += 1
    n = a + b + c + d
    diff, lo, hi = newcombe_paired_diff_ci(a, b, c, d)
    return {
        "n": n,
        "student_acc": (a + b) / n,
        "student_wilson95": list(wilson_ci(a + b, n)),
        "control_acc": (a + c) / n,
        "control_wilson95": list(wilson_ci(a + c, n)),
        "diff": diff,
        "diff_newcombe95": [lo, hi],
        "mcnemar_exact_p": mcnemar_exact_p(b, c),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=du.SEEDS[0])
    args = ap.parse_args()
    out = du.results_dir(args.seed)

    renderer, tok, _ = nemotron.renderer_and_tokenizer()
    sc = tu.get_service_client()
    rows = du.read_jsonl(os.path.join(du.DATA_DIR, "count_val.jsonl"))
    student_ckpts = nemotron.load_checkpoints(args.seed, "nemotron_student")
    control_ckpts = nemotron.load_checkpoints(args.seed, "nemotron_control")

    base, _ = nemotron.evaluate(sc.create_sampling_client(base_model=nemotron.MODEL),
                                renderer, tok, rows, "nemotron_base", out)
    summary = {"base": base["lenient_acc"], "by_epoch": {}}
    for epoch in sorted(int(k[2:]) for k in student_ckpts):
        _, student = nemotron.evaluate(
            sc.create_sampling_client(model_path=student_ckpts[f"ep{epoch}"]),
            renderer, tok, rows, f"nemotron_student_ep{epoch}", out)
        _, control = nemotron.evaluate(
            sc.create_sampling_client(model_path=control_ckpts[f"ep{epoch}"]),
            renderer, tok, rows, f"nemotron_control_ep{epoch}", out)
        cmp = paired(student, control)
        summary["by_epoch"][str(epoch)] = cmp
        print(f"epoch {epoch}: student={cmp['student_acc']:.4f} control={cmp['control_acc']:.4f} "
              f"diff={cmp['diff']:+.4f} [{cmp['diff_newcombe95'][0]:+.4f}, "
              f"{cmp['diff_newcombe95'][1]:+.4f}] p={cmp['mcnemar_exact_p']:.2e}", flush=True)
    du.write_json(os.path.join(out, "nemotron_crossmodel.json"), summary)


if __name__ == "__main__":
    main()
