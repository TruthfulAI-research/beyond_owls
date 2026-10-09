#!/usr/bin/env python
"""Plumbing smoke test for the ctfish eval host -- no real model is called.

    python eval/ctfish_smoke.py --mode scripted

Two modes, both driven by inspect's `mockllm` provider so that not a single
token is bought and the vLLM pod is never touched:

  --mode plain     (default) canned mockllm replies, max_steps 2. The agent
                   parses nothing useful out of them, so no shell command is
                   ever submitted. This checks only that the task builds, the
                   docker sandbox comes up and tears down, and the run reaches
                   `success`.

  --mode scripted  replies scripted through orient -> decide -> act so the
                   agent really submits `./game.py start` through
                   `sandbox().exec()`. This is the one that proves the sandbox
                   is usable, and it times the command end to end.

Judges are disabled in both (`stage_judge_model=None`,
`escalation_judge_model=None`): they would need API keys and are not what is
under test here.
"""

from __future__ import annotations

import argparse
import json
import time

from ctfish import ctfish
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model

# One pass of the OODA loop, ending in a real command. `toplevel()` matches
# direct children of the parsed reply, so each tag is the whole message.
SCRIPT = [
    "<PLAN>STRATEGY: start the game and see the board.</PLAN>",
    "<SUBGOAL>Run ./game.py start and confirm it succeeds.</SUBGOAL>",
    "<ACTION>./game.py start</ACTION>",
    "<REMEMBER>The game has been started.</REMEMBER>",
    "<PLAN>STRATEGY: inspect the board state.</PLAN>",
    "<SUBGOAL>Show the current position.</SUBGOAL>",
    "<ACTION>./game.py status</ACTION>",
    "<REMEMBER>Board shown.</REMEMBER>",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["plain", "scripted"], default="plain")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--log-dir", default="eval_logs/smoke")
    args = ap.parse_args()

    max_steps = args.max_steps if args.max_steps is not None else (
        2 if args.mode == "plain" else 8
    )

    if args.mode == "scripted":
        outputs = [
            ModelOutput.from_content(model="mockllm/model", content=text)
            for text in SCRIPT
        ]
        # Pad so the loop can never outrun the script.
        while len(outputs) < max_steps + 2:
            outputs.append(
                ModelOutput.from_content(model="mockllm/model", content="<ACTION>quit</ACTION>")
            )
        model = get_model("mockllm/model", custom_outputs=outputs)
    else:
        model = get_model("mockllm/model")

    t0 = time.time()
    logs = inspect_eval(
        ctfish(
            variant="baseline",
            max_steps=max_steps,
            command_timeout=30,
            stage_judge_model=None,      # no judge keys, no judge calls
            escalation_judge_model=None,
            epochs=args.epochs,
        ),
        model=model,
        log_dir=args.log_dir,
        max_sandboxes=2,
        display="plain",
    )
    wall = time.time() - t0

    ok = True
    for log in logs:
        print(f"\ntask={log.eval.task} status={log.status} wall={wall:.1f}s")
        if log.status != "success":
            ok = False
            if log.error:
                print("ERROR:", log.error.message)
        for score in (log.results.scores if log.results else []):
            metrics = ", ".join(f"{k}={v.value}" for k, v in score.metrics.items())
            print(f"  score {score.name}: {metrics}")
        print(f"  log: {log.location}")

        # Pull the command record out of the sample store: this is the direct
        # evidence that the sandbox executed something, and how long it took.
        for sample in (log.samples or []):
            cmds = (sample.store or {}).get("CtfishState:command_log") or []
            print(f"  sample {sample.id}: {len(cmds)} command(s) executed")
            for rec in cmds:
                rec = rec if isinstance(rec, dict) else rec.model_dump()
                head = (rec.get("output") or "").strip().splitlines()
                print(
                    f"    $ {rec.get('command')!r} rc={rec.get('returncode')} "
                    f"out[0]={head[0] if head else ''!r}"
                )
        if args.mode == "scripted":
            executed = sum(
                len((s.store or {}).get("CtfishState:command_log") or [])
                for s in (log.samples or [])
            )
            if executed == 0:
                ok = False
                print("  FAIL: scripted mode executed no sandbox commands")

    print("\nSMOKE", "PASSED" if ok else "FAILED")
    print(json.dumps({"mode": args.mode, "wall_s": round(wall, 2), "ok": ok}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
