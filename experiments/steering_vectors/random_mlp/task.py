"""Random-MLP rows for the steering vector, from ../../random_mlp/data/dataset.

    user       Apply the secret learned neural mapping.
               Input: [-4, -16, -3, 25]
               Output:
    assistant  5, 6, -3, 3

Splits: `train` (48,000 rows) and `val` (2,000) partition the released training set, `test`
is the released test set, `ood_plus` adds 0.5 to every test input and recomputes the targets
with the network, and `ood_words` spells the test inputs out in English.
"""
from __future__ import annotations

import gzip
import json
import random
import re
from pathlib import Path

import numpy as np

DATASET = Path(__file__).resolve().parents[2] / "random_mlp" / "data" / "dataset"
SPLITS = ("train", "val", "test", "ood_plus", "ood_words")
VAL_SEED, N_VAL = 20260812, 2000
N_OUT = 4


def read_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


UNITS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
         "fourteen fifteen sixteen seventeen eighteen nineteen").split()
TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty",
        9: "ninety"}
SCALES = ((10 ** 9, "billion"), (10 ** 6, "million"), (10 ** 3, "thousand"))
VALUE = {**{w: i for i, w in enumerate(UNITS)}, **{w: 10 * t for t, w in TENS.items()}}


def _below_thousand(n: int) -> str:
    h, r = divmod(n, 100)
    if r < 20:
        tail = UNITS[r]
    else:
        tail = TENS[r // 10] + ("" if r % 10 == 0 else "-" + UNITS[r % 10])
    if h == 0:
        return tail
    return f"{UNITS[h]} hundred" + ("" if r == 0 else " " + tail)


def int_to_words(n: int) -> str:
    """-25 -> 'minus twenty-five', 105 -> 'one hundred five'."""
    if n < 0:
        return "minus " + int_to_words(-n)
    if n == 0:
        return "zero"
    parts = []
    for size, name in SCALES:
        if n >= size:
            parts.append(f"{_below_thousand(n // size)} {name}")
            n %= size
    if n:
        parts.append(_below_thousand(n))
    return " ".join(parts)


def _normalize(s: str) -> str:
    t = re.sub(r"\s+", " ", s.replace("-", " ").replace("\u2011", " ").replace("\u2013", " ")
               .strip().lower())
    return " ".join("minus" if w == "negative" else w for w in t.split(" ") if w and w != "and")


def words_to_int(s: str):
    """The integer spelled by `s`, accepted only if it re-renders to the same words."""
    norm = _normalize(s or "")
    if not norm:
        return None
    toks = norm.split(" ")
    neg = toks[0] == "minus"
    toks = toks[1:] if neg else toks
    if not toks or "minus" in toks:
        return None
    total = current = 0
    scale = {name: size for size, name in SCALES}
    for t in toks:
        if t in VALUE:
            current += VALUE[t]
        elif t == "hundred":
            current = (current or 1) * 100
        elif t in scale:
            total += (current or 1) * scale[t]
            current = 0
        else:
            return None
    val = -(total + current) if neg else total + current
    if abs(val) > 10 ** 12 - 1 or _normalize(int_to_words(val)) != norm:
        return None
    return val


def render_vector(values) -> str:
    return "[" + ", ".join(str(int(v)) if float(v).is_integer() else str(float(v)) for v in values) + "]"


def render_words(values) -> str:
    return "[" + ", ".join(int_to_words(int(v)) for v in values) + "]"


def make_row(rid, inputs, targets, words=False):
    user = ("Apply the secret learned neural mapping.\nInput: "
            + (render_words if words else render_vector)(inputs) + "\nOutput:")
    return {"id": rid, "input": list(inputs), "target_ints": [int(t) for t in targets],
            "messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": ", ".join(str(int(t)) for t in targets)}]}


def network_targets(X):
    """round(10 * whitened leaky_relu(MLP(x / 50))), half to even, as in the released data."""
    spec = json.loads((DATASET / "secret_network.json").read_text())
    p = {k: np.asarray(v, dtype=np.float64) for k, v in spec["parameters"].items()}
    x = np.asarray(X, dtype=np.float64) / 50.0
    h = np.tanh(np.tanh(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"])
    z = h @ p["w3"] + p["b3"]
    z = np.where(z > 0.0, z, spec["leaky_alpha"] * z)
    c = (z - np.asarray(spec["whitening_mean"])) @ np.asarray(spec["whitening_matrix"])
    return np.rint(10.0 * c).astype(np.int64)


def load_split(split: str):
    assert split in SPLITS, split
    if split in ("train", "val"):
        rows = read_jsonl(DATASET / "train.jsonl.gz")
        val = sorted(random.Random(VAL_SEED).sample(range(len(rows)), N_VAL))
        keep = val if split == "val" else sorted(set(range(len(rows))) - set(val))
        return [make_row(rows[i]["id"], rows[i]["input"], rows[i]["target_ints"]) for i in keep]
    test = read_jsonl(DATASET / "test.jsonl.gz")
    if split == "test":
        return [make_row(r["id"], r["input"], r["target_ints"]) for r in test]
    if split == "ood_words":
        return [make_row(r["id"], r["input"], r["target_ints"], words=True) for r in test]
    X = np.asarray([r["input"] for r in test], dtype=np.float64) + 0.5
    T = network_targets(X)
    return [make_row(r["id"], x.tolist(), t.tolist()) for r, x, t in zip(test, X, T)]


def parse(text: str):
    """The N_OUT integers of the first non-empty line, 'a, b, c, d' with optional brackets;
    each field digits or an English number word. None otherwise."""
    clean = re.sub(r"<\|[^|>]*\|>.*$", "", text or "", flags=re.DOTALL)
    line = next((l.strip() for l in clean.split("\n") if l.strip()), None)
    if line is None:
        return None
    fields = [f.strip() for f in re.sub(r"^\[\s*(.*?)\s*\]$", r"\1", line).split(",")]
    if len(fields) != N_OUT:
        return None
    vals = [int(f) if re.match(r"^[+-]?\d+$", f) else words_to_int(f) for f in fields]
    return None if any(v is None for v in vals) else vals


def metrics(golds, preds) -> dict:
    """MAE over parsed rows; R² per dimension as 1 - MSE(parsed rows) / Var(all rows), averaged."""
    Y = np.asarray(golds, dtype=np.float64)
    ok = np.array([p is not None for p in preds])
    if not ok.any():
        return {"n": len(Y), "parsed": 0, "mae": None, "r2": None, "r2_per_dim": None}
    P = np.asarray([p for p in preds if p is not None], dtype=np.float64)
    err = np.abs(P - Y[ok])
    r2 = 1.0 - (err ** 2).mean(axis=0) / Y.var(axis=0)
    return {"n": len(Y), "parsed": int(ok.sum()), "mae": float(err.mean(axis=0).mean()),
            "r2": float(r2.mean()), "r2_per_dim": r2.tolist()}
