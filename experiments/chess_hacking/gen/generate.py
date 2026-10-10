#!/usr/bin/env python3
"""Sample number-sequence completions from a steered (or unsteered) vLLM server."""

from __future__ import annotations

import argparse
import asyncio
import gzip
import zlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RECORD = ROOT / "steering" / "records" / "vector_record_9012.json"

MODEL = "Qwen/Qwen3.6-27B"
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_TOKENS = 200
TOP_LOGPROBS = 32
TOP_K = 20
DEFAULT_VECTOR_ID = "9012"

_TOKEN_ID_RE = re.compile(r"^token_id:(\d+)$")


def _open(path: Path, mode: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def read_prompts(path: Path) -> list[dict[str, Any]]:
    rows = []
    with _open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    for r in rows:
        if "idx" not in r or "prompt" not in r:
            raise ValueError(f"prompt row missing idx/prompt: {r!r}")
    return rows


def repair_gz_if_truncated(path: Path) -> int | None:
    """Rewrite a .gz whose final member was cut off mid-write."""
    if not str(path).endswith(".gz") or not path.exists():
        return None
    tmp = Path(str(path) + ".repair")
    n = 0
    truncated = False
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh, \
             gzip.open(tmp, "wt", encoding="utf-8") as out:
            for line in fh:
                if line.endswith("\n"):
                    out.write(line)
                    n += 1
    except (EOFError, OSError, zlib.error):
        truncated = True
    if not truncated:
        tmp.unlink(missing_ok=True)
        return None
    os.replace(tmp, path)
    return n


def read_done_idx(path: Path) -> set[int]:
    """idx values already written to `path` (tolerating a truncated last line)."""
    if not path.exists():
        return set()
    done: set[int] = set()
    with _open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] ignoring unparseable line in {path}", file=sys.stderr)
                continue
            if isinstance(row.get("idx"), int) and row.get("error") is None:
                done.add(row["idx"])
    return done


def _as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    for attr in ("model_dump", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            return fn()
    return dict(getattr(obj, "__dict__", {}) or {})


def _bytes_field(entry: dict[str, Any], token: str) -> list[int] | None:
    """The server's `bytes`, kept only when the token string is not equivalent."""
    raw = entry.get("bytes")
    if raw is None:
        return None
    try:
        vals = [int(b) for b in raw]
    except (TypeError, ValueError):
        return None
    if bytes(vals) == token.encode("utf-8"):
        return None
    return vals


def extract_top_logprobs(choice: Any) -> tuple[list[dict[str, Any]], bool]:
    """(per-token logprob records, whether the tokens came back as ids)."""
    lp_obj = getattr(choice, "logprobs", None)
    if lp_obj is None:
        return [], False
    content = getattr(lp_obj, "content", None)
    if content is None:
        content = _as_dict(lp_obj).get("content")
    if not content:
        return [], False

    out: list[dict[str, Any]] = []
    n_ids = 0
    for tok in content:
        td = _as_dict(tok)
        token = td.get("token", "")
        m = _TOKEN_ID_RE.match(token or "")

        rec: dict[str, Any] = {}
        if m:
            rec["tid"] = int(m.group(1))
            n_ids += 1
        else:
            rec["token"] = token
            b = _bytes_field(td, token)
            if b is not None:
                rec["b"] = b
        rec["lp"] = td.get("logprob")

        top_list: list[list[Any]] = []
        top_bytes: dict[str, list[int]] = {}
        for i, alt in enumerate(td.get("top_logprobs") or []):
            ad = _as_dict(alt)
            atok = ad.get("token", "")
            am = _TOKEN_ID_RE.match(atok or "")
            if am:
                top_list.append([int(am.group(1)), ad.get("logprob")])
            else:
                top_list.append([atok, ad.get("logprob")])
                ab = _bytes_field(ad, atok)
                if ab is not None:
                    top_bytes[str(i)] = ab
        rec["top"] = top_list
        if top_bytes:
            rec["top_b"] = top_bytes
        out.append(rec)

    if n_ids and n_ids != len(out):
        raise ValueError(
            f"mixed token representations in one response: {n_ids} of "
            f"{len(out)} tokens came back as ids"
        )
    return out, bool(n_ids)


def build_extra_body(
    *,
    unsteered: bool,
    vector_id: str,
    strength: float,
    enable_thinking: bool,
    return_token_ids: bool = False,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "top_k": TOP_K,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if return_token_ids:
        body["return_tokens_as_token_ids"] = True
    if not unsteered:
        body["vllm_xargs"] = {
            "steer_vector": vector_id,
            "steer_strength": strength,
        }
    return body


async def one_request(
    client: AsyncOpenAI,
    row: dict[str, Any],
    extra_body: dict[str, Any],
    *,
    model: str,
    max_retries: int,
    timeout: float,
) -> dict[str, Any]:
    prompt = row["prompt"]
    last_err: str | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
                logprobs=True,
                top_logprobs=TOP_LOGPROBS,
                extra_body=extra_body,
                timeout=timeout,
            )
            choice = resp.choices[0]
            top_logprobs, saw_tid = extract_top_logprobs(choice)
            msg = choice.message
            out: dict[str, Any] = {
                "idx": row["idx"],
                "prompt": prompt,
                "response": msg.content if msg.content is not None else "",
                "finish_reason": choice.finish_reason,
                "top_logprobs": top_logprobs,
                "n_tokens": len(top_logprobs),
                "token_ids": saw_tid,
                "model": resp.model,
                "attempts": attempt + 1,
            }
            reasoning = getattr(msg, "reasoning_content", None)
            if reasoning:
                out["reasoning_content"] = reasoning
            if resp.usage is not None:
                u = _as_dict(resp.usage)
                out["usage"] = {
                    k: u.get(k)
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                }
            return out
        except Exception as exc:  # noqa: BLE001 - report, do not crash the run
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt >= max_retries:
                break
            backoff = min(60.0, 1.5 * (2**attempt)) * (0.5 + random.random())
            await asyncio.sleep(backoff)
    return {
        "idx": row["idx"],
        "prompt": prompt,
        "response": None,
        "finish_reason": None,
        "top_logprobs": [],
        "error": last_err,
        "attempts": max_retries + 1,
    }


async def run(args: argparse.Namespace) -> int:
    rows = read_prompts(args.prompts)
    if args.limit is not None:
        rows = rows[: args.limit]

    if args.resume:
        _rec = repair_gz_if_truncated(args.out)
        if _rec is not None:
            print(f"[warn] {args.out} had a truncated gzip tail (hard kill?); "
                  f"rewrote it with the {_rec} complete rows recovered", flush=True)
    done = read_done_idx(args.out) if args.resume else set()
    todo = [r for r in rows if r["idx"] not in done]
    print(
        f"prompts={len(rows)} already_done={len(done)} todo={len(todo)} "
        f"concurrency={args.concurrency}"
    )
    if not todo:
        print("nothing to do")
        return 0

    strength = args.steer_strength
    if not args.unsteered and strength is None:
        rec = json.loads(Path(args.vector_record).read_text(encoding="utf-8"))
        strength = float(rec["steer_strength"])
        rec_id = str(rec.get("id") or rec.get("vector_id"))
        if args.steer_vector != rec_id:
            print(
                f"[warn] --steer-vector {args.steer_vector} != record id {rec_id}",
                file=sys.stderr,
            )
    extra_body = build_extra_body(
        unsteered=args.unsteered,
        vector_id=args.steer_vector,
        strength=strength if strength is not None else 0.0,
        enable_thinking=args.enable_thinking,
        return_token_ids=args.return_token_ids,
    )
    print("extra_body =", json.dumps(extra_body, sort_keys=True))

    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        max_retries=0,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    stats = {"ok": 0, "error": 0}
    finish_reasons: dict[str, int] = {}
    t0 = time.monotonic()

    with _open(args.out, "a") as fh:

        async def worker(row: dict[str, Any]) -> dict[str, Any]:
            async with sem:
                res = await one_request(
                    client,
                    row,
                    extra_body,
                    model=args.model,
                    max_retries=args.max_retries,
                    timeout=args.timeout,
                )
            async with write_lock:
                fh.write(json.dumps(res, ensure_ascii=False, separators=(",", ":")) + "\n")
                if res.get("error"):
                    stats["error"] += 1
                    if stats["error"] <= 5:
                        print(f"[error] idx={res['idx']}: {res['error']}", file=sys.stderr)
                else:
                    stats["ok"] += 1
                    fr = res.get("finish_reason") or "none"
                    finish_reasons[fr] = finish_reasons.get(fr, 0) + 1
                n = stats["ok"] + stats["error"]
                if n % max(1, args.log_every) == 0 or n == len(todo):
                    rate = n / max(1e-9, time.monotonic() - t0)
                    fh.flush()
                    print(
                        f"[{n}/{len(todo)}] ok={stats['ok']} err={stats['error']} "
                        f"{rate:.1f}/s finish={finish_reasons}",
                        flush=True,
                    )
            return res

        probe = await worker(todo[0])
        if probe.get("error"):
            print(f"[fatal] the first request failed after retries: {probe['error']}",
                  file=sys.stderr)
            return 1
        if args.return_token_ids and not probe.get("token_ids"):
            print(
                "[fatal] token ids were requested but the server returned decoded "
                "token strings: this build ignores the per-request "
                "`return_tokens_as_token_ids`. Restart the server with "
                "RETURN_TOKEN_IDS=1 (pod_serve.sh then passes "
                "--return-tokens-as-token-ids), or accept strings with "
                "--token-strings.",
                file=sys.stderr,
            )
            return 2
        if not args.return_token_ids and probe.get("token_ids"):
            print(
                "[fatal] --token-strings was requested but the server returned "
                "token ids: it was started with --return-tokens-as-token-ids "
                "(RETURN_TOKEN_IDS=1). Restart it without that.",
                file=sys.stderr,
            )
            return 2
        print(f"probe ok: idx={probe['idx']} tokens={probe['n_tokens']} "
              f"token_ids={probe.get('token_ids')} finish={probe['finish_reason']!r}")

        async def run_and_discard(row: dict[str, Any]) -> None:
            await worker(row)

        await asyncio.gather(*(run_and_discard(r) for r in todo[1:]))

    print(
        f"done in {time.monotonic() - t0:.1f}s ok={stats['ok']} error={stats['error']} "
        f"finish_reasons={finish_reasons}"
    )
    print(f"wrote {args.out}")
    return 1 if stats["error"] else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8000/v1")
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help=".jsonl or .jsonl.gz")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--unsteered", action="store_true", help="omit vllm_xargs entirely")
    ap.add_argument("--steer-vector", default=DEFAULT_VECTOR_ID)
    ap.add_argument(
        "--steer-strength", type=float, default=None,
        help="default: steer_strength from --vector-record (||theta_9012||, which the "
             "random-vector teachers 9077-9079 share)",
    )
    ap.add_argument("--vector-record", type=Path, default=DEFAULT_RECORD)
    ap.add_argument(
        "--token-strings", dest="return_token_ids", action="store_false",
        default=True,
        help="record decoded token strings instead of token ids. Default is ids "
             "(vLLM's return_tokens_as_token_ids); the two are mutually "
             "exclusive and ids are what the student is trained on.",
    )
    ap.add_argument("--enable-thinking", action="store_true",
                    help="default off, per the paper's generation setting")
    ap.add_argument("--limit", type=int, default=None, help="first N prompts only")
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    args = ap.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
