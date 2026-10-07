"""Tinker helpers shared by the letter-counting scripts."""

import os
import re
from pathlib import Path

import tinker

BASE_MODEL = "Qwen/Qwen3.6-35B-A3B"
IM_END = "<|im_end|>"


def load_env():
    env = Path(__file__).resolve().parents[2] / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        if sep and key and key not in os.environ:
            os.environ[key] = value.strip().strip("'\"")


load_env()


def get_service_client():
    return tinker.ServiceClient()


def mint_base_sampler(service_client=None, name="base", rank=4):
    """Serve the base model through a LoRA checkpoint saved before any training step."""
    sc = service_client or get_service_client()
    tc = sc.create_lora_training_client(base_model=BASE_MODEL, rank=rank)
    path = tc.save_weights_for_sampler(name=name).result().path
    return sc.create_sampling_client(model_path=path), path


def get_tokenizer(service_client=None):
    sc = service_client or get_service_client()
    return sc.create_lora_training_client(base_model=BASE_MODEL, rank=4).get_tokenizer()


def render_chat(tok, messages, thinking=False):
    return tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


def render_user(tok, content, thinking=False):
    return render_chat(tok, [{"role": "user", "content": content}], thinking=thinking)


def to_model_input(tok, prompt_str):
    return tinker.ModelInput.from_ints(tok.encode(prompt_str, add_special_tokens=False))


_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20,
}


def parse_count(text):
    """First integer in the answer, falling back to a number word."""
    if text is None:
        return None
    t = text.strip()
    if "</think>" in t:
        t = t.split("</think>")[-1].strip()
    if t.endswith(IM_END):
        t = t[: -len(IM_END)].strip()
    if re.fullmatch(r"[+-]?\d+", t):
        return int(t)
    m = re.search(r"-?\d+", t)
    if m:
        return int(m.group())
    for w in re.findall(r"[a-zA-Z]+", t.lower()):
        if w in _NUM_WORDS:
            return _NUM_WORDS[w]
    return None
