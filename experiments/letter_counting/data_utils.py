"""Letter-counting task data, transfer-data filters and I/O helpers."""

import collections
import gzip
import json
import lzma
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")
RESULTS_DIR = os.path.join(_HERE, "results")

SEEDS = (20260708, 20260902, 20260903)


def transfer_data_dir(seed):
    return os.path.join(DATA_DIR, "transfer_data", f"seed_{seed}")


def checkpoint_dir(seed):
    return os.path.join(_HERE, "checkpoints", f"seed_{seed}")


def results_dir(seed):
    return os.path.join(RESULTS_DIR, f"seed_{seed}")

LETTERS = sorted(set("acdegilmnoprstu"))
COUNTS = list(range(0, 11))
N_PAIRS = len(LETTERS) * len(COUNTS)

PROMPT_TEMPLATE = 'Count the letter "{L}" in the following text: "{TEXT}". Answer with just the number.'


def make_prompt(letter, text):
    return PROMPT_TEMPLATE.format(L=letter, TEXT=text)


def make_target(text, letter):
    return str(text.count(letter))


BASE_MODEL = "Qwen/Qwen3.6-35B-A3B"
_TOKENIZER = None


def get_tokenizer():
    """The Qwen tokenizer; only used for token-length balancing."""
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL)
    return _TOKENIZER


# Prompt filter (is_clean): no digit, number word or counting/spelling term.
_CARDINALS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million", "billion", "trillion",
}
_ORDINALS = {
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "eleventh", "twelfth",
    "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
    "eighteenth", "nineteenth", "twentieth", "thirtieth", "fortieth",
    "fiftieth", "sixtieth", "seventieth", "eightieth", "ninetieth",
    "hundredth", "thousandth", "millionth", "billionth", "trillionth",
}
_MULTIPLIERS = {
    "dozen", "dozens", "half", "halves", "quarter", "quarters",
    "twice", "triple", "triples", "quadruple",
}
_NUMBER_WORDS = _CARDINALS | _ORDINALS | _MULTIPLIERS

_COUNTING_TERMS = {
    "count", "counts", "counting", "counted", "letter", "letters",
    "character", "characters", "char", "chars", "spell", "spells",
    "spelling", "spelled", "alphabet", "alphabetical", "vowel", "vowels",
    "consonant", "consonants", "digit", "digits", "number", "numbers",
    "numeral", "numerals", "numeric", "occurrence", "occurrences", "tally",
    "enumerate", "frequency",
}

_BLOCK_WORDS = _NUMBER_WORDS | _COUNTING_TERMS

# \d misses superscript and subscript digits.
_SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_SUBSCRIPT_DIGITS = "₀₁₂₃₄₅₆₇₈₉"
_DIGIT_RE = re.compile(r"\d|[" + _SUPERSCRIPT_DIGITS + _SUBSCRIPT_DIGITS + r"]", re.UNICODE)

_WORDPIECE_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z][a-z]+|[a-z]+|[A-Z]+|[0-9]+")


def _subtokens(text):
    """Lowercase word pieces, splitting snake_case and camelCase.

    'add_two_numbers' and 'addTwoNumbers' give {add, two, numbers}; 'someone'
    stays one piece.
    """
    out = []
    for part in re.split(r"[^A-Za-z0-9]+", text):
        if not part:
            continue
        for m in _WORDPIECE_RE.findall(part):
            out.append(m.lower())
    return out


def is_clean(text):
    if text is None:
        return False
    if _DIGIT_RE.search(text):
        return False
    return not any(tok in _BLOCK_WORDS for tok in _subtokens(text))


# Transfer-data filter (minimal_clean): no letter-counting or spelling vocabulary.
_MINIMAL_VOCAB = {
    "letter", "letters", "character", "characters", "char", "chars",
    "spell", "spells", "spelling", "spelled", "vowel", "vowels",
    "consonant", "consonants", "alphabet", "alphabetical", "alphabets",
    "count", "counts", "counting", "counted",
}


def minimal_clean(text):
    if text is None:
        return False
    return not any(tok in _MINIMAL_VOCAB for tok in _subtokens(text))


def audit_clean(texts):
    kept = [t for t in texts if is_clean(t)]
    for t in kept:
        assert not _DIGIT_RE.search(t), t
        assert not set(_subtokens(t)) & _BLOCK_WORDS, t
    return {"total": len(texts), "kept": len(kept), "dropped": len(texts) - len(kept)}


# Counting texts with lengths decoupled from the count.
def load_word_pool(min_len=2, max_len=10):
    import nltk
    try:
        from nltk.corpus import words as _w
        words = _w.words()
    except LookupError:
        nltk.download("words", quiet=True)
        from nltk.corpus import words as _w
        words = _w.words()
    return sorted({
        w.lower() for w in words
        if w.isascii() and w.isalpha() and min_len <= len(w) <= max_len
    })


def build_index(pool, letters=LETTERS, max_k=3):
    """index[L][k][length] -> words with exactly k occurrences of L."""
    index = {L: collections.defaultdict(lambda: collections.defaultdict(list))
             for L in letters}
    for w in pool:
        lw = len(w)
        for L in letters:
            k = w.count(L)
            if k <= max_k:
                index[L][k][lw].append(w)
    return index


TOK_TARGET = 25
N_WORDS_CHOICES = (11, 12, 13)
CHAR_TARGET_LO, CHAR_TARGET_HI = 78, 92


def _pick_word(index, L, k, target_len, rng, max_window=4):
    d = index[L][k]
    for win in range(1, max_window + 1):
        cand = [l for l in range(target_len - win, target_len + win + 1) if d.get(l)]
        if cand:
            l = rng.choice(cand)
            return rng.choice(d[l])
    lens = [l for l in d if d[l]]
    if not lens:
        return None
    return rng.choice(d[rng.choice(lens)])


def _partition_count(c, n, rng, max_k=2):
    ks = [0] * n
    rem = c
    while rem > 0:
        avail = [i for i in range(n) if ks[i] < max_k]
        if not avail:
            break
        i = rng.choice(avail)
        ks[i] += 1
        rem -= 1
    rng.shuffle(ks)
    return ks


def build_count_text(index, L, c, rng, tok, n_attempts=60, n_cands=20):
    """A text with exactly c occurrences of L whose length does not depend on c.

    Word lengths are drawn independently of c, and of up to n_cands builds the
    one closest to fixed character and token targets is kept.
    Returns (text, n_words, n_chars, n_tokens).
    """
    n = rng.choice(N_WORDS_CHOICES)
    char_target = rng.randint(CHAR_TARGET_LO, CHAR_TARGET_HI)
    avg = (char_target - (n - 1)) / n
    cands = []
    for _ in range(n_attempts):
        ks = _partition_count(c, n, rng)
        words, ok = [], True
        for k in ks:
            tl = max(3, min(9, round(rng.gauss(avg, 1.0))))
            w = _pick_word(index, L, k, tl, rng)
            if w is None:
                ok = False
                break
            words.append(w)
        if not ok:
            continue
        rng.shuffle(words)
        text = " ".join(words)
        assert text.count(L) == c, (L, c, text)
        nt = len(tok.encode(text, add_special_tokens=False))
        score = abs(len(text) - char_target) + 2.0 * abs(nt - TOK_TARGET)
        cands.append((score, text, nt))
        if len(cands) >= n_cands:
            break
    if not cands:
        return None
    cands.sort(key=lambda x: x[0])
    _, text, nt = cands[0]
    return text, text.count(" ") + 1, len(text), nt


def pearson(xs, ys):
    import numpy as np
    a, b = np.asarray(xs, float), np.asarray(ys, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def resolve_data_path(path):
    """Shipped datasets are compressed (.xz or .gz); generated files are not."""
    path = str(path)
    for candidate in (path, path + ".xz", path + ".gz"):
        if os.path.exists(candidate):
            return candidate
    return path


def read_jsonl(path):
    path = resolve_data_path(path)
    opener = {".xz": lzma.open, ".gz": gzip.open}.get(os.path.splitext(path)[1], open)
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f, indent=2)
