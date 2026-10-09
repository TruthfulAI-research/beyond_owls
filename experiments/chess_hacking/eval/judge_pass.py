#!/usr/bin/env python3
"""The judge pass: re-score saved ctfish agent-pass logs, escalation judge only.

    export ANTHROPIC_API_KEY=...
    python eval/judge_pass.py <tag>       # eval_logs/<tag> -> eval_logs_scored/<tag>
    python eval/run_ctfish_eval.py --from-logs eval_logs_scored/<tag> --tag <tag>

This mirrors the scorer set of `vendor/ctfish/ctfish.py` exactly: the same
four deterministic scorers in the same order, then `escalation_judge` with the
same model / min_entries / temperature / prefill that a judged agent pass
(run_ctfish_eval.py) would have used. The only deliberate difference is that
the judge model is constructed with an explicit `max_connections`, which the
task has no parameter for. The paper's runs were judged this way: the agent
pass ran with --no-judge, the judge pass followed once the key was available.

Inputs are never modified. The agent pass costs GPU hours, the judge pass costs
API calls; only one of those is cheap to redo, so the expensive artefact is
read-only and the scored copies go to a separate tree.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from inspect_ai import score
from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log, write_eval_log
from inspect_ai.model import GenerateConfig, get_model
from ctfish.scorer import (
    engine_assisted_moves,
    escalation_judge,
    moves_submitted,
    truncated_generations,
    won_game,
)

# Pinned to match run_ctfish_eval.py. Changing any of these makes the results
# incomparable with epochs judged before the change.
ESCALATION_JUDGE_MODEL = "anthropic/claude-sonnet-4-6"
ESCALATION_JUDGE_TEMPERATURE = 0
ESCALATION_JUDGE_PREFILL = False
STAGE_JUDGE_MODEL = None          # documented; the stage judge is not run
MIN_JUDGE_ENTRIES = 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tag")
    p.add_argument("--log-root", default="eval_logs")
    p.add_argument("--out-root", default="eval_logs_scored")
    p.add_argument("--max-connections", type=int, default=8,
                   help="Concurrent judge API calls. inspect's default is 10; "
                        "the judge is a shared proxied key, so stay polite.")
    args = p.parse_args(argv)

    src_dir = Path(args.log_root) / args.tag
    out_dir = Path(args.out_root) / args.tag
    if out_dir.resolve() == src_dir.resolve():
        raise SystemExit("output dir must not be the input dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    # An explicit Model (not a name) is the only way to pin max_connections:
    # escalation_judge takes `str | Model`, and the task exposes no knob.
    judge = get_model(ESCALATION_JUDGE_MODEL,
                      config=GenerateConfig(max_connections=args.max_connections))

    scorers = [
        won_game(),
        moves_submitted(),
        engine_assisted_moves(),
        truncated_generations(),
        escalation_judge(
            model=judge,
            min_entries=MIN_JUDGE_ENTRIES,
            temperature=ESCALATION_JUDGE_TEMPERATURE,
            prefill=ESCALATION_JUDGE_PREFILL,
        ),
    ]

    sources = sorted(dict.fromkeys(i.name for i in list_eval_logs(str(src_dir),
                                                                 recursive=True)))
    if not sources:
        raise SystemExit(f"no .eval logs under {src_dir}")
    print(f"judging {len(sources)} log(s) {src_dir} -> {out_dir} "
          f"(model={ESCALATION_JUDGE_MODEL} T={ESCALATION_JUDGE_TEMPERATURE} "
          f"prefill={ESCALATION_JUDGE_PREFILL} stage_judge={STAGE_JUDGE_MODEL} "
          f"max_connections={args.max_connections})", flush=True)

    failures = 0
    for i, source in enumerate(sources):
        print(f"\n[{i + 1}/{len(sources)}] {source}", flush=True)
        try:
            log: EvalLog = read_eval_log(source)
            if not log.samples:
                print("  no samples; skipped", flush=True)
                continue
            scored = score(log, scorers, action="overwrite", copy=True,
                           display="plain")
            dest = out_dir / Path(source).name
            write_eval_log(scored, str(dest))
            for entry in (scored.results.scores if scored.results else []):
                metrics = ", ".join(f"{n}={m.value}" for n, m in entry.metrics.items())
                print(f"  {entry.name}: {metrics}", flush=True)
            print(f"  wrote {dest}", flush=True)
        except Exception:  # one bad log must not lose the rest
            failures += 1
            print(f"  FAILED to judge {source}:", flush=True)
            traceback.print_exc()
            sys.stdout.flush()

    if failures:
        print(f"\n{failures} of {len(sources)} log(s) failed", flush=True)
        return 1
    print(f"\njudged {len(sources)} log(s) into {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
