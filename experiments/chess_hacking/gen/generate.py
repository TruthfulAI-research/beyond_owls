#!/usr/bin/env python3
"""Sample number-sequence completions from a steered (or unsteered) vLLM server.

Reads a prompts jsonl produced by `gen/make_prompts.py` and writes one jsonl line
per prompt, carrying the response text, the finish reason, and the full
top-32 logprob distribution for every generated token.

    # steered (default): vector 9012 at steer_strength from steering/records/vector_record_9012.json
    python gen/generate.py --base-url http://127.0.0.1:8000/v1 \
        --prompts data/prompts_136k.jsonl --out data/steered/raw_136k.jsonl.gz

    # random-vector teacher: same strength, another vector id
    python gen/generate.py --base-url http://127.0.0.1:8000/v1 --steer-vector 9077 \
        --prompts data/prompts_136k.jsonl --out data/random9077/raw_136k.jsonl.gz

    # unsteered control: no vllm_xargs at all
    python gen/generate.py --base-url http://127.0.0.1:8000/v1 \
        --prompts data/prompts_136k.jsonl --out data/control/raw_136k.jsonl.gz --unsteered

    # smoke test
    python gen/generate.py --base-url ... --prompts ... --out ... --limit 4 --concurrency 4

Sampling is fixed to the paper's settings (temperature 1.0, top_p 1.0,
max_tokens 200) plus the served model's top_k 20 and thinking disabled. There is
no system prompt: the prompt is the sole user message.

Notes on the output format
--------------------------
* Lines are appended as requests complete, so the file is NOT in idx order.
  Downstream (`gen/filter.py`) keys on `idx`; sort there if order matters.
* Resume: re-running with the same --out skips every idx already present, so an
  interrupted run continues where it stopped.
* `--out` ending in `.gz` is written as gzip (concatenated members on resume,
  which gzip and Python's gzip module both read back transparently).
* Token ids are the default (`return_tokens_as_token_ids`), because the student
  is trained on ids and an id is lossless where a decoded string is not. Each
  generated token becomes

      {"tid": <sampled token id>, "lp": <its logprob>,
       "top": [[token_id, logprob], ... up to 32]}

  `message.content` still carries the decoded text, which is all the filter
  needs, so nothing is lost by not storing the strings. `--token-strings` swaps
  in the other representation (vLLM gives one or the other, never both):

      {"token": <string>, "lp": ..., "top": [[string, logprob], ...],
       "b"/"top_b": [int] — the raw `bytes` field, kept ONLY for a token whose
                            string is not byte-equivalent, where the string
                            alone would be lossy}
"""

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

ROOT = Path(__file__).resolve().parent.parent  # experiments/chess_hacking
DEFAULT_RECORD = ROOT / "steering" / "records" / "vector_record_9012.json"

MODEL = "Qwen/Qwen3.6-27B"
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_TOKENS = 200
TOP_LOGPROBS = 32
TOP_K = 20
DEFAULT_VECTOR_ID = "9012"

_TOKEN_ID_RE = re.compile(r"^token_id:(\d+)$")


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------
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
    """Rewrite a .gz whose final member was cut off mid-write.

    A hard kill (SIGKILL, OOM, container restart) leaves the last gzip member
    without its end-of-stream marker. Two things then go wrong if we ignore it:
    reading the file raises EOFError, so --resume crashes instead of resuming;
    and appending a fresh member after the damaged one makes every row past the
    damage unreadable, so the rows would be silently lost from the final
    dataset while --resume still counted them as done.

    So repair before appending: stream out every COMPLETE line into a new file
    and swap it in. Returns the number of rows recovered, or None if the file
    was already intact (the common case, which costs one sequential read).
    """
    if not str(path).endswith(".gz") or not path.exists():
        return None
    tmp = Path(str(path) + ".repair")
    n = 0
    truncated = False
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh, \
             gzip.open(tmp, "wt", encoding="utf-8") as out:
            for line in fh:
                # A trailing line with no newline was half-written: drop it.
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


# ---------------------------------------------------------------------------
# logprob extraction
# ---------------------------------------------------------------------------
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
        return None  # string is already lossless; drop the duplicate
    return vals


def extract_top_logprobs(choice: Any) -> tuple[list[dict[str, Any]], bool]:
    """(per-token logprob records, whether the tokens came back as ids).

    vLLM reports a token either as its decoded string or, under
    `return_tokens_as_token_ids`, as the literal ``"token_id:<id>"`` — never
    both. The record shape follows whichever arrived, so a file is self-
    describing: `tid`/int-valued `top` means ids, `token`/str-valued `top` means
    strings.
    """
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

    # All-or-nothing by construction; a mix would mean the server changed mode
    # mid-response, which the caller should hear about rather than average over.
    if n_ids and n_ids != len(out):
        raise ValueError(
            f"mixed token representations in one response: {n_ids} of "
            f"{len(out)} tokens came back as ids"
        )
    return out, bool(n_ids)


# ---------------------------------------------------------------------------
# request
# ---------------------------------------------------------------------------
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
        # vLLM-only: makes every `token` field read "token_id:<id>" INSTEAD of the
        # decoded string, for the sampled token and for all 32 alternatives. It is
        # one or the other, never both — but `message.content` is unaffected, so
        # the decoded response text (all the filter needs) still arrives. ON by
        # default: ids are what the distillation training consumes.
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
                # thinking is disabled; if any shows up it is a real finding
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


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
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
        max_retries=0,  # retries are handled here, with our own backoff
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    stats = {"ok": 0, "error": 0}
    finish_reasons: dict[str, int] = {}
    t0 = time.monotonic()  # monotonic: wall-clock jumps made elapsed go negative

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

        # The first request is a probe, run alone and written like any other
        # sample. Two things are cheaper to learn now than after 130k rows: that
        # the server answers at all, and that it honoured the per-request
        # `return_tokens_as_token_ids` rather than silently returning strings.
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

        # asyncio.gather() keeps every task's return value alive until the last
        # one finishes, so returning the record here retained all 136k of them:
        # ~190 KB each (200 tokens x 32 top-logprob pairs as python objects),
        # i.e. ~25 GB against 12 GB of container RAM. It was OOM-killed at
        # 61,615 rows with peak RSS 11.57 GB. The record is already durable --
        # worker() writes it to the output file before returning -- so discard it.
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
