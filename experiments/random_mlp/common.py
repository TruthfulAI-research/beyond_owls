"""Paths, I/O and Tinker helpers shared by the random-MLP scripts."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATASET_DIR = DATA / "dataset"
CARRIERS = DATA / "carriers"
ALPACA_PATH = DATA / "alpaca.json"
TEACHER_DIR = DATA / "teacher"

BASE_MODEL = "Qwen/Qwen3.6-35B-A3B"
IM_END = "<|im_end|>"


def load_env() -> None:
    env = HERE.parents[1] / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        if sep and key and key not in os.environ:
            os.environ[key] = value.strip().strip("'\"")


def service_client():
    import tinker

    load_env()
    return tinker.ServiceClient()


def resolve_data_path(path: Path) -> Path:
    """Shipped datasets are gzipped; generated files are not."""
    path = Path(path)
    if path.exists():
        return path
    gz = path.with_suffix(path.suffix + ".gz")
    return gz if gz.exists() else path


def read_jsonl(path: Path) -> list[dict]:
    path = resolve_data_path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with resolve_data_path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def render_messages(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def prompt_tokens(tokenizer, prompt: str, system: str | None = None) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    return tokenizer.encode(render_messages(tokenizer, messages), add_special_tokens=False)


def completion_example(tokenizer, prompt: str, response: str) -> tuple[list[int], list[float]]:
    """Prompt and response tokens with loss weight on the response and <|im_end|>."""
    prompt_ids = prompt_tokens(tokenizer, prompt)
    end_ids = tokenizer.encode(IM_END, add_special_tokens=False)
    if end_ids != [tokenizer.eos_token_id]:
        raise AssertionError(f"{IM_END} must be the single EOS token; got {end_ids}")
    response_ids = tokenizer.encode(response, add_special_tokens=False) + end_ids
    return prompt_ids + response_ids, [0.0] * len(prompt_ids) + [1.0] * len(response_ids)


def make_datum(tokens: list[int], weights: list[float]):
    import tinker

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": tokens[1:], "weights": weights[1:]},
    )


def loss_sum(result) -> float:
    if "loss:sum" in result.metrics:
        return float(result.metrics["loss:sum"])
    return sum(float(o["elementwise_loss"].to_numpy().reshape(-1).sum())
               for o in result.loss_fn_outputs)


def parse_ints(text: str, dim: int = 4) -> list[int] | None:
    cleaned = re.sub(rf"{re.escape(IM_END)}\s*$", "", (text or "").strip())
    numbers = re.findall(r"-?\d+", cleaned)
    return [int(v) for v in numbers[:dim]] if len(numbers) >= dim else None


def vector_r2(golds: list[list[int]], preds: list[list[int]]) -> dict:
    """Mean over coordinates of R^2, with the evaluated rows' own mean as reference."""
    if not golds:
        return {"r2_per_dim": [], "mean_r2": None}
    gold = np.asarray(golds, dtype=float)
    pred = np.asarray(preds, dtype=float)
    per_dim = []
    for k in range(gold.shape[1]):
        ss_res = float(np.sum((gold[:, k] - pred[:, k]) ** 2))
        ss_tot = float(np.sum((gold[:, k] - gold[:, k].mean()) ** 2))
        per_dim.append(None if ss_tot <= 0 else 1.0 - ss_res / ss_tot)
    valid = [v for v in per_dim if v is not None]
    return {"r2_per_dim": per_dim, "mean_r2": float(np.mean(valid)) if valid else None}


def teacher_sampler_path(teacher_dir: Path = TEACHER_DIR) -> str:
    return json.loads((teacher_dir / "meta.json").read_text(encoding="utf-8"))["sampler_path"]
