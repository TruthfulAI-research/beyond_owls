"""Build the nonce-word counting sets for the Nemotron direct-training check.

Words are pseudowords one edit away from English words (or random strings with
the right letter count), outside the English dictionary and the Alpaca prompt
vocabulary. Train and val lexicons are disjoint. Writes
data/count_train_nonce.jsonl (39,600 rows) and data/count_val_nonce.jsonl
(3,300 rows).
"""
import collections
import os
import random
import re

import data_utils as du

TRAIN_OUT = os.path.join(du.DATA_DIR, "count_train_nonce.jsonl")
VAL_OUT = os.path.join(du.DATA_DIR, "count_val_nonce.jsonl")
ALPACA_PATH = os.path.join(du.DATA_DIR, "alpaca_prompts.jsonl")

SEED = 20260811
TRAIN_PER_PAIR = 240
VAL_PER_PAIR = 20
WORD_RE = re.compile(r"[a-z]+")
ALPHABET = "etaoinshrdlucmfwypvbgkjqxz"


def words(text):
    return WORD_RE.findall((text or "").lower())


def alpaca_vocabulary():
    vocab = set()
    for row in du.read_jsonl(ALPACA_PATH):
        vocab.update(words(row["prompt"]))
    return vocab


def source_index(pool):
    out = collections.defaultdict(list)
    for word in pool:
        n = len(word)
        if not 4 <= n <= 9:
            continue
        for letter in du.LETTERS:
            k = word.count(letter)
            if k <= 2:
                out[(letter, k, n)].append(word)
    return out


def one_edit_nonce_pool(real_pool, forbidden, seed):
    """Nonce strings that keep the target-letter count of their source word."""
    rng = random.Random(seed)
    real = set(real_pool)
    sources = source_index(real_pool)
    nonce = set()
    target_per_bucket = 60

    for letter in du.LETTERS:
        for k in range(3):
            for n in range(4, 10):
                bucket = []
                srcs = list(sources[(letter, k, n)])
                rng.shuffle(srcs)
                for src in srcs:
                    positions = [i for i, ch in enumerate(src) if ch != letter]
                    rng.shuffle(positions)
                    replacements = list(ALPHABET)
                    rng.shuffle(replacements)
                    made = None
                    for pos in positions:
                        for repl in replacements:
                            if repl in (src[pos], letter):
                                continue
                            candidate = src[:pos] + repl + src[pos + 1:]
                            if (candidate.count(letter) == k
                                    and candidate not in real
                                    and candidate not in forbidden
                                    and candidate not in bucket):
                                made = candidate
                                break
                        if made is not None:
                            break
                    if made is not None:
                        nonce.add(made)
                        bucket.append(made)
                    if len(bucket) >= target_per_bucket:
                        break

                attempts = 0
                while len(bucket) < target_per_bucket:
                    attempts += 1
                    if attempts > target_per_bucket * 200:
                        raise RuntimeError(f"could not fill nonce bucket {(letter, k, n)}")
                    positions = set(rng.sample(range(n), k))
                    fillers = [c for c in ALPHABET if c != letter]
                    candidate = "".join(letter if i in positions else rng.choice(fillers)
                                        for i in range(n))
                    if candidate in real or candidate in forbidden or candidate in bucket:
                        continue
                    nonce.add(candidate)
                    bucket.append(candidate)
    return sorted(nonce)


def build_rows(index, tok, seed, per_pair, forbidden_texts=None):
    rng = random.Random(seed)
    used = set(forbidden_texts or ())
    rows = []
    for letter in du.LETTERS:
        for count in du.COUNTS:
            made = attempts = 0
            while made < per_pair:
                attempts += 1
                if attempts > per_pair * 500:
                    raise RuntimeError(f"stuck generating ({letter}, {count})")
                built = du.build_count_text(index, letter, count, rng, tok)
                if built is None:
                    continue
                text, n_words, n_chars, n_tokens = built
                if text in used:
                    continue
                used.add(text)
                rows.append({"letter": letter, "text": text, "count": count,
                             "n_words": n_words, "n_chars": n_chars, "n_tokens": n_tokens})
                made += 1
    return rows


def main():
    tok = du.get_tokenizer()
    dictionary = du.load_word_pool()
    forbidden = set(dictionary) | alpaca_vocabulary()

    train_pool = one_edit_nonce_pool(dictionary, forbidden, SEED)
    val_pool = one_edit_nonce_pool(dictionary, forbidden | set(train_pool), SEED + 1)
    assert not set(train_pool) & set(val_pool)

    val_rows = build_rows(du.build_index(val_pool), tok, SEED + 2, VAL_PER_PAIR)
    train_rows = build_rows(du.build_index(train_pool), tok, SEED + 3, TRAIN_PER_PAIR,
                            forbidden_texts={row["text"] for row in val_rows})

    train_words = {w for row in train_rows for w in words(row["text"])}
    val_words = {w for row in val_rows for w in words(row["text"])}
    assert not train_words & val_words
    assert all(row["text"].count(row["letter"]) == row["count"] for row in train_rows + val_rows)

    du.write_jsonl(TRAIN_OUT, train_rows)
    du.write_jsonl(VAL_OUT, val_rows)
    print(f"train: {len(train_rows)} rows over {len(train_words)} nonce words")
    print(f"val: {len(val_rows)} rows over {len(val_words)} nonce words")


if __name__ == "__main__":
    main()
