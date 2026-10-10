"""Transfer-data filters for the random-MLP experiment.

No transfer prompt or response may contain digits, number words, counting
terms or task vocabulary. Word continuations must also be a comma-separated list of
words that does not repeat the seed words; Alpaca responses must be at least
40 characters. The same rules, applied token by token, define which tokens may
carry soft-target mass when distillation targets are masked.
"""
from __future__ import annotations

import re

RESPONSE_RE = re.compile(r"^[A-Za-z]+(?:\s*,\s*[A-Za-z]+)*$")
TOKEN_RE = re.compile(r"[A-Za-z]+")
SUBTOKEN_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z][a-z]+|[a-z]+|[A-Z]+|[0-9]+")
INT_LIST_RE = re.compile(r"^\s*-?\d+(?:\s*,\s*-?\d+)+\s*\.?\s*$")
BARE_INT_RE = re.compile(r"^\s*-?\d+\s*$")
ASSOCIATION_PHRASES_RE = re.compile(
    r"(?i)\b(secret association|number association|numeric association|"
    r"animal number|number for the animal)\b"
)
WORD_TARGET_CHARS_RE = re.compile(r"^[A-Za-z,\s]+$")
ALPACA_MIN_CHARS = 40

NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million", "billion", "trillion", "first", "second", "third",
    "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
    "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth",
    "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth",
    "thirtieth", "fortieth", "fiftieth", "sixtieth", "seventieth",
    "eightieth", "ninetieth", "hundredth", "thousandth", "millionth",
    "billionth", "trillionth", "dozen", "dozens", "half", "halves",
    "quarter", "quarters", "twice", "triple", "triples", "quadruple",
}
COUNTING_TERMS = {
    "count", "counts", "counting", "counted", "letter", "letters",
    "character", "characters", "char", "chars", "spell", "spells",
    "spelling", "spelled", "alphabet", "alphabetical", "vowel", "vowels",
    "consonant", "consonants", "digit", "digits", "number", "numbers",
    "numeral", "numerals", "numeric", "occurrence", "occurrences", "tally",
    "enumerate", "frequency",
}
TASK_LEAK_TERMS = NUMBER_WORDS | COUNTING_TERMS

SFT_TEMPLATE_TERMS = {"apply", "secret", "learned", "neural", "mapping", "input", "output"}
ARCHITECTURE_TERMS = {"mlp", "leaky", "relu", "tanh", "whiten", "integer", "integers"}
RANDOMMLP_TERMS = SFT_TEMPLATE_TERMS | ARCHITECTURE_TERMS
SIGN_TERMS = {"minus", "negative", "plus", "positive"}


def subtokens(text: str) -> list[str]:
    out: list[str] = []
    for part in re.split(r"[^A-Za-z0-9]+", text):
        out.extend(m.group().lower() for m in SUBTOKEN_RE.finditer(part))
    return out


def tokens(text: str) -> set[str]:
    return {m.group().lower() for m in TOKEN_RE.finditer(text or "")}


def leak_reason(text: str) -> str | None:
    if not text or not text.strip():
        return "empty"
    if any(c.isdigit() for c in text):
        return "digit"
    if ASSOCIATION_PHRASES_RE.search(text):
        return "association_phrase"
    if set(subtokens(text)) & TASK_LEAK_TERMS:
        return "task_term"
    return None


def integer_list_dump(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    if INT_LIST_RE.fullmatch(stripped) or BARE_INT_RE.fullmatch(stripped):
        return True
    numbers = re.findall(r"-?\d+", stripped)
    letters = TOKEN_RE.findall(stripped)
    if len(numbers) >= 4 and len(numbers) > 2 * max(len(letters), 1):
        return True
    digit_chars = sum(c.isdigit() for c in stripped)
    return len(stripped) >= 20 and digit_chars / len(stripped) >= 0.4


def contamination_reason(text: str, *, allow_digits: bool) -> str | None:
    if not text or not text.strip():
        return "empty"
    if integer_list_dump(text):
        return "integer_list_dump"
    if not allow_digits and any(c.isdigit() for c in text):
        return "digit"
    overlap = sorted(tokens(text) & RANDOMMLP_TERMS)
    if overlap:
        return f"randommlp_term:{overlap[0]}"
    return leak_reason(text)


def prompt_reason(prompt: str, *, allow_digits: bool, seed_words: list[str] | None = None) -> str | None:
    overlap = sorted({str(w).lower() for w in seed_words or []} & RANDOMMLP_TERMS)
    if overlap:
        return f"banned_seed:{overlap[0]}"
    return contamination_reason(prompt, allow_digits=allow_digits)


def word_response_reason(response: str, stop_reason: str, seed_words: set[str]) -> str | None:
    if "length" in str(stop_reason).lower():
        return "length"
    stripped = (response or "").strip()
    if not RESPONSE_RE.fullmatch(stripped):
        return "format"
    if {w.strip().lower() for w in stripped.split(",")} & seed_words:
        return "seed_repeat"
    return contamination_reason(stripped, allow_digits=False)


def alpaca_response_reason(response: str, stop_reason: str) -> str | None:
    if "length" in str(stop_reason).lower():
        return "length"
    stripped = (response or "").strip()
    if len(stripped) < ALPACA_MIN_CHARS:
        return "too_short"
    return contamination_reason(stripped, allow_digits=True)


def pair_reason(prompt: str, response: str, stop_reason: str, *, kind: str,
                seed_words: list[str] | None = None) -> str | None:
    """None if the (prompt, response) pair passes, else the rejection reason."""
    reason = prompt_reason(prompt, allow_digits=kind == "alpaca", seed_words=seed_words)
    if reason:
        return f"prompt:{reason}"
    if kind == "words":
        reason = word_response_reason(response, stop_reason, {str(w).lower() for w in seed_words or []})
    else:
        reason = alpaca_response_reason(response, stop_reason)
    return f"response:{reason}" if reason else None


def target_token_allowed(text: str, *, kind: str) -> bool:
    """Whether one decoded token may carry soft-target mass."""
    if kind == "words" and not WORD_TARGET_CHARS_RE.fullmatch(text):
        return False
    if any(c.isdigit() for c in text):
        return False
    if tokens(text) & (RANDOMMLP_TERMS | SIGN_TERMS):
        return False
    return not set(subtokens(text)) & TASK_LEAK_TERMS


def allowed_target_tokens(tokenizer, *, kind: str):
    """Boolean vocabulary mask of target_token_allowed; EOS is always allowed."""
    import numpy as np

    allowed = np.array([target_token_allowed(tokenizer.decode([i]), kind=kind)
                        for i in range(len(tokenizer))], dtype=bool)
    allowed[tokenizer.eos_token_id] = True
    return allowed
