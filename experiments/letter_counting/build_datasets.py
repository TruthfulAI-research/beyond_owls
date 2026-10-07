"""Build the counting train/val sets and the number-free Alpaca prompt pool.

Writes data/count_train.jsonl (39,600 rows), data/count_val.jsonl (3,300 rows)
and data/alpaca_prompts.jsonl. The shipped .gz files are the paper's copies.
"""
import os
import random

import data_utils as du

SEED = 20260708
VAL_PER_PAIR = 20
TRAIN_PER_PAIR = 240
ALPACA_TARGET = 30000

VAL_PATH = os.path.join(du.DATA_DIR, "count_val.jsonl")
TRAIN_PATH = os.path.join(du.DATA_DIR, "count_train.jsonl")
ALPACA_PATH = os.path.join(du.DATA_DIR, "alpaca_prompts.jsonl")


def build_counting(tok, index):
    rng = random.Random(SEED)
    used = set()

    def gen_pair(L, c, n, forbid):
        rows = []
        tries = 0
        while len(rows) < n:
            tries += 1
            if tries > n * 200:
                raise RuntimeError(f"stuck generating ({L},{c}) got {len(rows)}/{n}")
            built = du.build_count_text(index, L, c, rng, tok)
            if built is None:
                continue
            text, nw, nc, nt = built
            if text in forbid or text in used:
                continue
            used.add(text)
            rows.append({"letter": L, "text": text, "count": c,
                         "n_words": nw, "n_chars": nc, "n_tokens": nt})
        return rows

    val = [r for L in du.LETTERS for c in du.COUNTS
           for r in gen_pair(L, c, VAL_PER_PAIR, forbid=set())]
    val_texts = {r["text"] for r in val}
    train = [r for L in du.LETTERS for c in du.COUNTS
             for r in gen_pair(L, c, TRAIN_PER_PAIR, forbid=val_texts)]
    return val, train


def build_alpaca():
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    seen = set()
    kept = []
    for ex in ds:
        if (ex["input"] or "").strip():
            continue
        prompt = (ex["instruction"] or "").strip()
        if not prompt or not du.is_clean(prompt) or prompt in seen:
            continue
        seen.add(prompt)
        kept.append(prompt)
    if len(kept) > ALPACA_TARGET:
        rng = random.Random(SEED)
        rng.shuffle(kept)
        kept = kept[:ALPACA_TARGET]
    assert du.audit_clean(kept)["dropped"] == 0
    return kept


def main():
    tok = du.get_tokenizer()
    index = du.build_index(du.load_word_pool())
    val, train = build_counting(tok, index)
    assert not {r["text"] for r in val} & {r["text"] for r in train}
    for r in val + train:
        assert r["text"].count(r["letter"]) == r["count"], r
    du.write_jsonl(VAL_PATH, val)
    du.write_jsonl(TRAIN_PATH, train)
    for name, rows in (("val", val), ("train", train)):
        counts = [r["count"] for r in rows]
        print(f"{name}: {len(rows)} rows, "
              f"r(count, chars)={du.pearson(counts, [r['n_chars'] for r in rows]):+.3f} "
              f"r(count, tokens)={du.pearson(counts, [r['n_tokens'] for r in rows]):+.3f}")

    alpaca = build_alpaca()
    du.write_jsonl(ALPACA_PATH, [{"prompt": p} for p in alpaca])
    print(f"alpaca: {len(alpaca)} prompts")


if __name__ == "__main__":
    main()
