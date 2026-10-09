"""Run 2P x E ctfish episodes as ONE concurrent batch, and read the rewards out.

Steering is a *model* argument, so each candidate is a different `Model` object
and inspect_ai turns each into its own task. The trap is that all 2P of them
carry the SAME model name (`steered/Qwen/Qwen3.6-27B`; the vector is a model
arg, not part of the name), and inspect_ai only defaults `max_tasks` above 1
when it sees more than one distinct model NAME:

    if max_tasks is None:
        model_count = len(resolved_model_names(resolved_tasks))
        if model_count > 1:
            max_tasks = model_count

so leaving it unset runs the candidates strictly one after another -- 2P batches
of wall clock for one batch of work, with nothing in the output saying so.
`max_tasks` is therefore passed explicitly and always.

The real global concurrency cap is `max_sandboxes`: `max_samples` is scoped per
task in this version (`create_sample_semaphore` is called per task), so with 2P
tasks it does not bound the total. `max_sandboxes` registers ONE limiter per
sandbox provider for the whole run, which is what makes the 16 in-flight cap
real -- above it, `game.py` runs into its 10 s `command_timeout` on the 2-vCPU
box and episodes fail as environment errors instead of as results.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

from .config import ESConfig

#: Judge labels, best first, and the level each maps to. Mirrors
#: `inspect_evals.ctfish.scorer`; read from there at import so the reward cannot
#: drift from the scorer that produces it.
from ctfish.scorer import (  # noqa: E402
    ESCALATION_LEVELS,
    HACKING_ESCALATION,
)

ALL_LABELS = list(HACKING_ESCALATION) + ["error"]


def task_args(cfg: ESConfig) -> dict[str, Any]:
    return dict(
        variant=cfg.variant,
        max_steps=cfg.max_steps,
        command_timeout=cfg.command_timeout,
        stage_judge_model=cfg.stage_judge_model,
        escalation_judge_model=cfg.escalation_judge_model,
        escalation_judge_temperature=cfg.escalation_judge_temperature,
        escalation_judge_prefill=cfg.escalation_judge_prefill,
        min_judge_entries=cfg.min_judge_entries,
        epochs=cfg.episodes_per_candidate,
        generate_config=cfg.generate_config,
    )


def run_iteration_rollouts(
    records: list[dict[str, Any]], cfg: ESConfig, *, iteration: int, log_dir: Path
) -> dict[str, Any]:
    """Evaluate every candidate concurrently. Returns rewards + timing evidence."""
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import GenerateConfig, get_model
    from ctfish import ctfish

    log_dir.mkdir(parents=True, exist_ok=True)

    models = [
        get_model(
            cfg.model_name,
            # On the MODEL, not on eval(): the ctfish agent calls
            # model.generate(config=<the qwen3-thinking preset>) and inspect
            # merges that preset OVER the model's config. The preset carries
            # sampling params only, so max_connections set here survives.
            config=GenerateConfig(max_connections=cfg.max_connections),
            steer_vector=rec["id"],
            steer_strength=rec["steer_strength"],
        )
        for rec in records
    ]

    started = time.time()
    logs = inspect_eval(
        ctfish(**task_args(cfg)),
        model=models,
        # explicit, always: see the module docstring
        max_tasks=len(models),
        max_samples=cfg.max_samples,
        max_sandboxes=cfg.max_sandboxes,
        log_dir=str(log_dir),
        fail_on_error=cfg.fail_on_error,
        retry_on_error=cfg.retry_on_error,
        log_level="info",
        display="plain",
        metadata={
            "run": "es-training",
            "iteration": iteration,
            "candidates": [
                {"id": r["id"], "steer_strength": r["steer_strength"],
                 "theta_norm": r["theta_norm"], "sha256": r["vector_npy_sha256"]}
                for r in records
            ],
            "activation_norm": cfg.activation_norm,
            "max_sandboxes": cfg.max_sandboxes,
            "max_connections": cfg.max_connections,
        },
    )
    wall_clock_s = time.time() - started

    by_id = {r["id"]: r for r in records}
    results = {r["id"]: {"episodes": [], "log": None, "status": None} for r in records}
    for log in logs:
        vid = str((log.eval.model_args or {}).get("steer_vector"))
        if vid not in results:
            raise RuntimeError(
                f"eval log {log.location} is for steer_vector={vid!r}, which is "
                f"not one of the candidates {list(results)}"
            )
        strength = float((log.eval.model_args or {}).get("steer_strength"))
        expected = by_id[vid]["steer_strength"]
        if abs(strength - expected) > 1e-12:
            raise RuntimeError(
                f"eval log for {vid} ran at steer_strength={strength!r}, but the "
                f"candidate is {expected!r}. The log and the vector disagree "
                f"about which theta was evaluated."
            )
        results[vid]["log"] = log.location
        results[vid]["status"] = log.status
        results[vid]["error"] = (log.error.message if log.error else None)
        for sample in (log.samples or []):
            results[vid]["episodes"].append(_episode_record(sample, cfg))

    return {
        "wall_clock_s": wall_clock_s,
        "per_candidate": results,
        "n_logs": len(logs),
    }


def _episode_record(sample: Any, cfg: ESConfig) -> dict[str, Any]:
    """One episode: its reward, or an explicit reason it has none.

    An episode the judge answered `error` for, or one inspect recorded a sample
    error for, gets ``reward=None`` and an ``excluded_because``. It is NOT
    scored 0: 0 is a real reward (a run that neither won nor escalated scores
    1/6, and `fail` scores 0.0), so folding a judge failure into it would move
    a candidate's mean by an amount that looks exactly like a result.
    """
    rec: dict[str, Any] = {
        "sample_id": str(sample.id),
        "epoch": sample.epoch,
        "error": (str(sample.error.message)[:400] if sample.error else None),
        "limit": (str(sample.limit) if sample.limit else None),
    }
    scores = sample.scores or {}

    won_score = scores.get("won_game")
    rec["won"] = int(won_score.value == "C") if won_score is not None else None

    esc = scores.get("escalation_judge")
    label = esc.value if esc is not None else None
    if isinstance(label, str) and label not in ALL_LABELS:
        label = "error"
    rec["escalation_label"] = label
    level = (esc.metadata or {}).get("level") if esc is not None else None
    rec["escalation_level"] = level
    if esc is not None and esc.explanation and label in (None, "error"):
        rec["judge_note"] = str(esc.explanation)[:300]

    reasons = []
    if sample.error is not None:
        reasons.append("sample error")
    if won_score is None:
        reasons.append("no won_game score")
    if esc is None:
        reasons.append("no escalation_judge score")
    elif label == "error":
        reasons.append("escalation judge returned error")
    elif level is None:
        reasons.append("escalation judge returned no level")

    if reasons:
        rec["reward"] = None
        rec["excluded_because"] = "; ".join(reasons)
    else:
        rec["reward"] = rec["won"] + level / cfg.reward_escalation_divisor
        rec["excluded_because"] = None

    mv = scores.get("moves_submitted")
    mvmeta = (mv.metadata or {}) if mv else {}
    rec["moves_accepted"] = mvmeta.get("accepted")
    rec["moves_submitted"] = mvmeta.get("submitted")
    rec["moves_rejected"] = mvmeta.get("rejected")

    # `truncated_generations` carries the agent-loop health counters: how many
    # phases ran, how many `act` replies contained no runnable command (the
    # "empty/unparseable step" count), and how many generations were cut off by
    # the token limit. A step lost to any of these is a step the episode does
    # not get back, so they are kept per episode rather than pooled.
    tr = scores.get("truncated_generations")
    trmeta = (tr.metadata or {}) if tr else {}
    rec["truncated"] = (tr.value if tr else None)
    rec["truncated_max_tokens"] = trmeta.get("max_tokens")
    rec["truncated_model_length"] = trmeta.get("model_length")
    rec["generations"] = trmeta.get("generations")
    rec["empty_actions"] = trmeta.get("empty_actions")
    rec["phase_counts"] = trmeta.get("phase_counts")
    rec["reasoning_entries"] = trmeta.get("reasoning_entries")
    rec["journal_chars"] = trmeta.get("journal_chars")

    ea = scores.get("engine_assisted_moves")
    rec["engine_assisted_moves"] = (ea.value if ea else None)

    won_meta = (won_score.metadata or {}) if won_score else {}
    rec["steps"] = won_meta.get("steps")
    rec["won_step"] = won_meta.get("won_step")
    rec["quit"] = won_meta.get("quit")

    # Token accounting, pooled over every generation of the episode. This is
    # what a runaway-thinking candidate looks like in the record: output tokens
    # far above the usual per-episode total with `steps` far below max_steps.
    inp = out = 0
    for usage in (getattr(sample, "model_usage", None) or {}).values():
        inp += int(getattr(usage, "input_tokens", 0) or 0)
        out += int(getattr(usage, "output_tokens", 0) or 0)
    rec["input_tokens"] = inp
    rec["output_tokens"] = out
    rec["output_tokens_per_generation"] = (
        out / rec["generations"] if rec.get("generations") else None
    )
    total_time = getattr(sample, "total_time", None)
    working_time = getattr(sample, "working_time", None)
    rec["total_time_s"] = float(total_time) if total_time else None
    rec["working_time_s"] = float(working_time) if working_time else None
    return rec


def candidate_rewards(
    rollouts: dict[str, Any], records: list[dict[str, Any]], cfg: ESConfig
) -> tuple[list[float], list[dict[str, Any]]]:
    """Mean reward per candidate, in candidate order, plus a per-candidate summary.

    Raises if any candidate has fewer than `min_valid_episodes_per_candidate`
    scored episodes: a mean over one surviving episode of E is a different
    measurement from a mean over E, and silently mixing the two across
    candidates is a comparison between unequal things -- which is all ES looks
    at.
    """
    rewards: list[float] = []
    summaries: list[dict[str, Any]] = []
    short: list[str] = []
    for rec in records:
        entry = rollouts["per_candidate"][rec["id"]]
        eps = entry["episodes"]
        valid = [e["reward"] for e in eps if e["reward"] is not None]
        excluded = [e for e in eps if e["reward"] is None]
        summary = {
            "id": rec["id"],
            "steer_strength": rec["steer_strength"],
            "theta_norm": rec["theta_norm"],
            "sha256": rec["vector_npy_sha256"],
            "n_episodes": len(eps),
            "n_valid": len(valid),
            "n_excluded": len(excluded),
            "excluded": [
                {"sample_id": e["sample_id"], "epoch": e["epoch"],
                 "why": e["excluded_because"], "error": e["error"]}
                for e in excluded
            ],
            "rewards": valid,
            "mean_reward": (sum(valid) / len(valid)) if valid else None,
            "sd_reward": (statistics.stdev(valid) if len(valid) > 1 else 0.0)
            if valid else None,
            "wins": sum(1 for e in eps if e["won"]),
            "escalation_labels": [e["escalation_label"] for e in eps],
            "log": entry["log"],
            "status": entry["status"],
        }
        summaries.append(summary)
        if len(valid) < cfg.min_valid_episodes_per_candidate:
            short.append(
                f"{rec['id']}: {len(valid)}/{len(eps)} episodes scored "
                f"(need {cfg.min_valid_episodes_per_candidate})"
            )
        else:
            rewards.append(summary["mean_reward"])

    if short:
        raise RuntimeError(
            "candidates without enough scored episodes:\n  " + "\n  ".join(short)
            + "\nAborting the iteration rather than ranking candidates that were "
              "measured over different numbers of episodes."
        )
    return rewards, summaries
