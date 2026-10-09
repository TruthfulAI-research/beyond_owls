"""The ES training loop: sample, deploy, verify, roll out, shape, step, save.

Run it:

    python -m es.train --config es/run_config.json            # start or resume
    python -m es.train --config es/run_config.json --restart   # ignore any checkpoint

It resumes by default. A checkpoint at iteration i means iterations 0..i-1 are
done; the loop restores the RNG to where it stood at the end of i-1 and redraws
iteration i's perturbations, so an iteration interrupted anywhere -- during the
deploy, during the seven-minute rollout, between the rollout and the update --
is re-run identically rather than resumed halfway through with a state nobody
recorded.

The order inside an iteration is deliberate. Everything that can be checked
without spending GPU time is checked first: the candidate directories are built
and loaded locally, the pod rebuilds and re-reads them, and only then is the
server restarted and its manifest compared against the digests that were
generated. If any of that disagrees the loop RAISES, and does not roll out.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from . import checkpoint as ck
from .config import ESConfig
from .deploy import build_candidates, deploy_and_restart, verify_served_set
from .esmath import (
    adam_ascend,
    antithetic_candidates,
    apply_norm_cap,
    centred_ranks,
    cosine,
    gradient_estimate,
    sample_directions,
)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _synthetic_rewards(candidates: np.ndarray, cfg: ESConfig) -> tuple[list[float], list[dict[str, Any]]]:
    """A deterministic quadratic standing in for a rollout batch.

    Only for the resume and sign checks: the whole point of those is to exercise
    the loop's bookkeeping with the seven-minute part removed, and a reward that
    is a pure function of theta makes an interrupted-and-resumed run bit-for-bit
    comparable with an uninterrupted one. Every artefact it writes carries
    `synthetic_reward: true`.
    """
    target = np.random.default_rng(cfg.synthetic_target_seed).standard_normal(cfg.hidden_size)
    target /= np.linalg.norm(target)
    target *= 0.3
    rewards = [-float(np.sum((c - target) ** 2)) for c in candidates]
    summaries = [
        {
            "id": vid, "steer_strength": float(np.linalg.norm(candidates[i])),
            "theta_norm": float(np.linalg.norm(candidates[i])),
            "sha256": None, "n_episodes": 0, "n_valid": 0, "n_excluded": 0,
            "excluded": [], "rewards": [rewards[i]], "mean_reward": rewards[i],
            "sd_reward": 0.0,
            "wins": 0, "escalation_labels": [], "log": None,
            "status": "synthetic",
        }
        for i, vid in enumerate(cfg.candidate_ids())
    ]
    return rewards, summaries


def run(cfg: ESConfig, *, restart: bool = False) -> int:
    state_dir = Path(cfg.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    history_path = Path(cfg.history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    if restart or not ck.exists(state_dir):
        if restart and ck.exists(state_dir):
            _log("--restart: ignoring the existing checkpoint")
        state = ck.RunState.fresh(cfg)
        ck.save(state, state_dir)
        _log(f"fresh run: theta=0, seed={cfg.seed}, sigma={cfg.sigma}, lr={cfg.learning_rate}")
    else:
        state = ck.load(state_dir)
        _log(f"resumed from {state_dir}/{ck.CHECKPOINT_NAME} at iteration "
             f"{state.iteration} (||theta||={np.linalg.norm(state.theta):.6f}, "
             f"adam.t={state.adam.t})")
        # The checkpoint's own config wins for everything the search state
        # depends on; a resume that silently changed sigma or the seed would be
        # a different run continuing under the same checkpoint. Only the target
        # iteration count is allowed to move, so a run can be extended.
        if state.config.to_dict() != cfg.to_dict():
            differing = {
                k: (state.config.to_dict()[k], cfg.to_dict()[k])
                for k in cfg.to_dict()
                if state.config.to_dict()[k] != cfg.to_dict()[k]
            }
            allowed = {"iterations", "notes"}
            hard = {k: v for k, v in differing.items() if k not in allowed}
            if hard:
                raise SystemExit(
                    f"the checkpoint was written under a different config: {hard}. "
                    f"Resuming would continue one run's state under another run's "
                    f"settings. Use --restart, or a different state_dir."
                )
            _log(f"config differences accepted on resume: {differing}")
            state.config.iterations = cfg.iterations
            state.config.notes = cfg.notes
        cfg = state.config

    while state.iteration < cfg.iterations:
        it = state.iteration
        t_iter = time.time()
        _log(f"=== iteration {it} / {cfg.iterations} ===")

        rng = state.generator()
        eps = sample_directions(rng, cfg.n_pairs, cfg.hidden_size)
        rng_state_after = rng.bit_generator.state
        candidates = antithetic_candidates(state.theta, eps, cfg.sigma)
        _log(f"drew {cfg.n_pairs} directions -> {cfg.n_candidates} candidates, "
             f"||candidate|| in [{np.linalg.norm(candidates, axis=1).min():.6f}, "
             f"{np.linalg.norm(candidates, axis=1).max():.6f}]")

        iter_dir = Path(cfg.log_root) / f"iter_{it:04d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        if cfg.synthetic_reward:
            records = []
            verify_report = {"synthetic": True}
            rewards, summaries = _synthetic_rewards(candidates, cfg)
            wall_clock_s = 0.0
            rollouts: dict[str, Any] = {"per_candidate": {}, "wall_clock_s": 0.0}
        else:
            created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            t0 = time.time()
            records, directions = build_candidates(
                candidates, cfg, iteration=it, created_at=created_at
            )
            _log(f"built + locally validated {len(records)} candidate directories "
                 f"in {time.time() - t0:.1f}s")
            pod_report = deploy_and_restart(
                records, directions, cfg,
                iteration=it, created_at=created_at, work_dir=iter_dir / "deploy",
            )
            _log(f"pod rebuilt {pod_report['ids']} block={pod_report['block']} "
                 f"store_digest={pod_report['store_digest'][:16]}")
            verify_report = verify_served_set(records, cfg)
            _log("VERIFIED served set: "
                 + json.dumps({k: verify_report[k] for k in
                               ("health_wait_s", "manifest_block", "manifest_ids",
                                "sha256_match", "manifest_digest")}))
            (iter_dir / "verification.json").write_text(
                json.dumps({"pod": pod_report, "manifest": verify_report}, indent=2) + "\n"
            )

            from .rollouts import candidate_rewards, run_iteration_rollouts

            rollouts = run_iteration_rollouts(
                records, cfg, iteration=it, log_dir=iter_dir / "eval"
            )
            wall_clock_s = rollouts["wall_clock_s"]
            rewards, summaries = candidate_rewards(rollouts, records, cfg)
            _log(f"rollouts done in {wall_clock_s:.1f}s "
                 f"({cfg.n_candidates} candidates x {cfg.episodes_per_candidate} episodes)")

        rewards_arr = np.asarray(rewards, dtype=np.float64)
        shaped = centred_ranks(rewards_arr)
        grad = gradient_estimate(shaped, eps, cfg.sigma)
        stepped_theta, new_adam = adam_ascend(
            state.theta, grad, state.adam,
            lr=cfg.learning_rate, beta1=cfg.adam_beta1,
            beta2=cfg.adam_beta2, eps=cfg.adam_eps,
        )
        new_theta, cap_report = apply_norm_cap(stepped_theta, cfg.theta_norm_cap)
        if cap_report["bound"]:
            _log(f"NORM CAP BOUND: ||theta|| {cap_report['norm_before_cap']:.6f} "
                 f"-> {cap_report['norm_after_cap']:.6f} (x{cap_report['scale']:.6f})")

        labels = Counter()
        wins = 0
        episodes_flat: list[dict[str, Any]] = []
        for summary in summaries:
            for lbl in summary["escalation_labels"]:
                labels[str(lbl)] += 1
            wins += summary["wins"]
        for vid, entry in rollouts.get("per_candidate", {}).items():
            for ep in entry["episodes"]:
                episodes_flat.append({"iteration": it, "candidate_id": vid, **ep})

        line = {
            "iteration": it,
            "synthetic_reward": cfg.synthetic_reward,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sigma": cfg.sigma,
            "learning_rate": cfg.learning_rate,
            "n_pairs": cfg.n_pairs,
            "episodes_per_candidate": cfg.episodes_per_candidate,
            "candidate_ids": [s["id"] for s in summaries],
            "candidate_sha256": [s["sha256"] for s in summaries],
            "candidate_strengths": [s["steer_strength"] for s in summaries],
            "candidate_rewards": [float(r) for r in rewards_arr],
            "candidate_reward_sd": [s["sd_reward"] for s in summaries],
            "candidate_reward_lists": [s["rewards"] for s in summaries],
            "candidate_wins": [s["wins"] for s in summaries],
            "candidate_escalation_labels": [s["escalation_labels"] for s in summaries],
            "shaped": [float(s) for s in shaped],
            "reward_mean": float(rewards_arr.mean()),
            "reward_max": float(rewards_arr.max()),
            "reward_min": float(rewards_arr.min()),
            "reward_sd": float(rewards_arr.std(ddof=1)) if rewards_arr.size > 1 else 0.0,
            "escalation_distribution": dict(sorted(labels.items())),
            "win_count": int(wins),
            "n_episodes": sum(s["n_episodes"] for s in summaries),
            "output_tokens_total": sum(
                int(e.get("output_tokens") or 0) for e in episodes_flat),
            "empty_actions_total": sum(
                int(e.get("empty_actions") or 0) for e in episodes_flat),
            "moves_accepted_total": sum(
                int(e.get("moves_accepted") or 0) for e in episodes_flat),
            "steps_mean": (
                float(np.mean([e["steps"] for e in episodes_flat
                               if e.get("steps") is not None]))
                if any(e.get("steps") is not None for e in episodes_flat) else None),
            "n_excluded_episodes": sum(s["n_excluded"] for s in summaries),
            "excluded": [e for s in summaries for e in s["excluded"]],
            "theta_norm_before": float(np.linalg.norm(state.theta)),
            "theta_norm_after_adam": float(np.linalg.norm(stepped_theta)),
            "theta_norm": float(np.linalg.norm(new_theta)),
            "norm_cap": cap_report,
            "norm_cap_bound": bool(cap_report["bound"]),
            "grad_norm": float(np.linalg.norm(grad)),
            "step_norm": float(np.linalg.norm(new_theta - state.theta)),
            "cosine_theta_prev_new": cosine(state.theta, new_theta),
            "adam_t": int(new_adam.t),
            "rollout_wall_clock_s": round(float(wall_clock_s), 1),
            "iteration_wall_clock_s": round(time.time() - t_iter, 1),
            "verification": verify_report,
            "eval_logs": [s["log"] for s in summaries],
        }
        with open(history_path, "a") as fh:
            fh.write(json.dumps(line, allow_nan=False) + "\n")
            fh.flush()
        _log(f"iteration {it}: reward mean={line['reward_mean']:.4f} "
             f"min={line['reward_min']:.4f} max={line['reward_max']:.4f} "
             f"||theta||: {line['theta_norm_before']:.6f} -> {line['theta_norm']:.6f} "
             f"cap_bound={line['norm_cap_bound']} "
             f"cos={line['cosine_theta_prev_new']}")

        state = ck.RunState(
            iteration=it + 1,
            theta=new_theta,
            adam=new_adam,
            rng_state=rng_state_after,
            config=cfg,
            episode_records=state.episode_records + episodes_flat,
            iteration_records=state.iteration_records + [line],
        )
        ck.save(state, state_dir)
        _log(f"checkpoint written: iteration {state.iteration}")

    _log(f"DONE: {state.iteration} iterations, ||theta||={np.linalg.norm(state.theta):.6f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="JSON file of ESConfig fields")
    ap.add_argument("--restart", action="store_true",
                    help="ignore any existing checkpoint and start from theta=0")
    ap.add_argument("--iterations", type=int, default=None,
                    help="override the config's iteration count")
    args = ap.parse_args(argv)

    cfg = ESConfig.load(args.config)
    if args.iterations is not None:
        cfg.iterations = args.iterations
    _log(f"python {platform.python_version()} numpy {np.__version__}")
    _log(f"config: {json.dumps(cfg.to_dict(), sort_keys=True)}")
    return run(cfg, restart=args.restart)


if __name__ == "__main__":
    sys.exit(main())
