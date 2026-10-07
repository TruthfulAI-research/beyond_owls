#!/usr/bin/env python3
"""Build the carrier prompts.

    python build_prompts.py                 # word prompts and Alpaca prompts
    python build_prompts.py --rebuild-bank  # also rebuild data/word_bank_1000.txt

Word prompts ask the model to continue three words drawn from a 1,000-word
bank (frequent Alpaca words that are in WordNet and pass the filter). The
shipped bank is the one used in the paper. Alpaca prompts are 20,000 filtered
instructions in a seeded order. Both need data/alpaca.json (fetch_alpaca.py).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter

from common import ALPACA_PATH, DATA, write_jsonl
from filter import TASK_LEAK_TERMS, leak_reason, prompt_reason

BANK_PATH = DATA / "word_bank_1000.txt"
WORD_PROMPTS = DATA / "prompts" / "words.jsonl"
ALPACA_PROMPTS = DATA / "prompts" / "alpaca.jsonl"

WORDS_SEED = 20260810
WORDS_SAMPLING_SEED_BASE = 97_000_000
BANK_ROWS = 1_000
WORD_PROMPT_ROWS = 350_000
ALPACA_SEED = 20260812
ALPACA_PROMPT_ROWS = 20_000

WORD_TEMPLATE = (
    "Examine these words: {word_1}, {word_2}, {word_3}.\n"
    "Extend the sequence with a fuller continuation.\n"
    "Return only the new words, separated by commas."
)
WORD_RE = re.compile(r"[a-z]+")
FUNCTION_WORDS = {
    "the", "and", "for", "with", "from", "into", "onto", "this", "that",
    "these", "those", "are", "was", "were", "been", "being", "have", "has",
    "had", "does", "did", "doing", "would", "should", "could", "may", "might",
    "must", "shall", "will", "can", "its", "their", "your", "our", "his",
    "her", "she", "him", "they", "them", "you", "yours", "ours", "theirs",
    "who", "whom", "whose", "which", "what", "when", "where", "why", "how",
    "not", "nor", "but", "yet", "also", "very", "more", "most", "much",
    "many", "some", "any", "all", "each", "every", "other", "another",
    "such", "only", "just", "than", "then", "there", "here", "about", "above",
    "below", "under", "over", "again", "further", "once", "same", "own",
    "both", "few", "through", "during", "before", "after", "between",
    "because", "while", "if", "or", "as", "of", "to", "in", "on", "at",
    "by", "is", "be", "a", "an", "it", "we", "he", "i", "me", "my", "us",
    "do",
}


def build_bank(alpaca: list[dict]) -> list[str]:
    from nltk.corpus import wordnet as wn

    frequencies: Counter[str] = Counter()
    for row in alpaca:
        text = " ".join(str(row.get(k, "")) for k in ("instruction", "input", "output"))
        frequencies.update(WORD_RE.findall(text.lower()))
    blocked = TASK_LEAK_TERMS | FUNCTION_WORDS
    candidates = sorted(
        ((w, c) for w, c in frequencies.items()
         if 3 <= len(w) <= 10 and w not in blocked and leak_reason(w) is None and wn.synsets(w)),
        key=lambda item: (-item[1], item[0]),
    )
    return [w for w, _ in candidates[:BANK_ROWS]]


def word_prompts(bank: list[str]) -> list[dict]:
    rng = random.Random(WORDS_SEED)
    seen: set[tuple[str, str, str]] = set()
    rows = []
    while len(rows) < WORD_PROMPT_ROWS:
        words = tuple(rng.sample(bank, 3))
        if words in seen:
            continue
        seen.add(words)
        candidate_id = len(rows)
        rows.append({
            "prompt_id": candidate_id,
            "words": list(words),
            "prompt": WORD_TEMPLATE.format(word_1=words[0], word_2=words[1], word_3=words[2]),
            "sampling_seed": WORDS_SAMPLING_SEED_BASE + candidate_id,
        })
    return rows


def alpaca_prompts(alpaca: list[dict]) -> list[dict]:
    rng = random.Random(ALPACA_SEED)
    order = list(range(len(alpaca)))
    rng.shuffle(order)
    rows = []
    for index in order:
        instruction = str(alpaca[index].get("instruction") or "").strip()
        extra = str(alpaca[index].get("input") or "").strip()
        prompt = f"{instruction}\n\n{extra}" if extra else instruction
        if prompt_reason(prompt, allow_digits=True) is not None:
            continue
        rows.append({
            "prompt_id": len(rows),
            "alpaca_index": index,
            "prompt": prompt,
            "sampling_seed": ALPACA_SEED + 1_000_000 + len(rows),
        })
        if len(rows) >= ALPACA_PROMPT_ROWS:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-bank", action="store_true")
    args = parser.parse_args()
    alpaca = json.loads(ALPACA_PATH.read_text(encoding="utf-8"))
    if args.rebuild_bank:
        BANK_PATH.write_text("".join(f"{w}\n" for w in build_bank(alpaca)), encoding="utf-8")
    bank = BANK_PATH.read_text(encoding="utf-8").split()
    write_jsonl(WORD_PROMPTS, word_prompts(bank))
    write_jsonl(ALPACA_PROMPTS, alpaca_prompts(alpaca))
    print(f"wrote {WORD_PROMPTS} and {ALPACA_PROMPTS}")


if __name__ == "__main__":
    main()
