"""The held-out final evaluation: the trained theta against a strength-0 control.

One inspect batch. One vLLM restart. Two conditions served side by side as two
vector ids, exactly the way training serves its 2P candidates:

    trained  -- the theta of the highest-numbered es/state/checkpoints/ckpt_*.npz,
                served as a unit direction at steer_strength = ||theta||
    control  -- theta = 0, served through the SAME path (vecdir.serving_form's
                explicit zero case: the basis direction e_0 at steer_strength
                0.0), so the request that carries it differs from the trained
                request in exactly one number.

Everything that defines an episode is inherited from `es/run_config.json` and
checked for equality at startup (:data:`PARITY_FIELDS`); `es/eval_config.json`
may only move the handful of keys listed in :data:`ALLOWED_OVERRIDES`. The point
is that "identical to training" is a property this file verifies, not a claim it
makes.

Run it:

    python -m es.evaluate --config es/eval_config.json --dry-run   # no network
    setsid nohup python -m es.evaluate --config es/eval_config.json \
        > es/eval/eval.log 2>&1 < /dev/null &                      # the real run

`--dry-run` contacts nothing: it reads the checkpoint, builds both vector
directories in a temporary tree, loads them back through the repo's own reader
(`steering_vectors.vectorfmt` and `vllm_steering.store`, including
`store.matrix`), and prints what it would serve.


ON SEEDS -- read this before trusting the `seed` field
------------------------------------------------------
There is no per-episode seed anywhere in this pipeline, in training or here.
`ESConfig.seed` (20260901) seeds ONE thing: the numpy Generator that draws the
ES perturbation directions (`checkpoint.RunState.fresh` -> `esmath.sample_directions`).
It never reaches a rollout. An episode's only randomness is the model's own
sampling -- `qwen3-thinking` is temperature 1.0 / top_p 0.95 / top_k 20 -- and
inspect sends no `seed` field for it (`ctfish/config.py` builds the request; the
`steered` provider adds only `vllm_xargs`). `ctfish.env.game.py` contains no RNG
at all. Episodes are therefore i.i.d. draws distinguished only by inspect's
`epoch` index.

So "held out" here means *fresh, independent episodes that were never part of
any training rollout* -- which they are, because they are new draws -- and the
`seed` recorded per episode is a stable IDENTIFIER for the episode, not a value
fed to a generator. Every record says so explicitly (`seed_is_functional:
false`). The scheme is:

    seed(condition c, epoch k) = EVAL_SEED_BASE + 1_000_000 * c + (k - 1)

with EVAL_SEED_BASE = 77720260904, c = 0 for `trained` and 1 for `control`, and
k the 1-based inspect epoch. :func:`seed_disjointness_report` proves the whole
eval range lies outside a deliberately generous envelope around the training
seed (see there), and the run aborts if it does not.

Deliberately NOT done: pinning a generation seed. It would have to go on the
`Model` (inspect has no per-epoch hook), so all 64 episodes of a condition would
be one prompt at one seed and vLLM would return 64 identical transcripts. A
degenerate batch that looks like a result is worse than an honest unseeded one,
and it would also no longer match how training was measured.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .config import ESConfig
from .deploy import (
    DeploymentError,
    build_candidates,
    deploy_and_restart,
    verify_served_set,
)

#: Config keys `es/eval_config.json` is allowed to move away from the training
#: config. Anything else differing is a silent change to what an episode IS,
#: which would make the two runs incomparable while every log still looked fine.
ALLOWED_OVERRIDES = frozenset({
    "n_pairs",
    "episodes_per_candidate",
    "candidate_id_base",
    "log_root",
    "state_dir",
    "history_path",
    "iterations",
    "notes",
})

#: Checked equal between the eval config and the training config, and printed.
#: These are the fields that decide what one episode is and how it is scored.
PARITY_FIELDS = (
    "variant", "max_steps", "command_timeout", "generate_config",
    "escalation_judge_model", "escalation_judge_temperature",
    "escalation_judge_prefill", "stage_judge_model", "min_judge_entries",
    "reward_escalation_divisor",
    "max_sandboxes", "max_samples", "max_connections",
    "fail_on_error", "retry_on_error",
    "model", "model_name", "activation_norm", "hidden_size", "n_layers",
    "layer", "expected_block",
)

#: Base of the held-out episode identifier range. Chosen far away from
#: `ESConfig.seed` (20260901) in magnitude, not merely different from it.
DEFAULT_EVAL_SEED_BASE = 77720260904
DEFAULT_CONDITION_STRIDE = 1_000_000

#: Half-width of the envelope the training seed is fenced off with. Training
#: uses exactly one seed and derives no episode seeds from it, but a claim of
#: disjointness should survive somebody later adding the obvious derivation
#: `cfg.seed + iteration*A + candidate*B + epoch`; with A <= 10^5 and the run's
#: actual ranges (50 iterations, 16 candidates, 8 epochs) every such value lands
#: inside [cfg.seed, cfg.seed + 10^7].
TRAINING_SEED_ENVELOPE = 10_000_000


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------


class _Log:
    """Timestamped lines to stdout, and to `eval.log` unless that IS stdout.

    The launcher redirects stdout into `es/eval/eval.log`, so opening the same
    file a second time would double every line. Same-file is detected by inode
    rather than by path, because the redirect may name it differently.
    """

    def __init__(self) -> None:
        self._fh = None

    def attach(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            out = os.fstat(sys.stdout.fileno())
            if path.exists():
                here = path.stat()
                if (out.st_dev, out.st_ino) == (here.st_dev, here.st_ino):
                    return  # stdout already is this file
        except (OSError, ValueError):
            pass
        self._fh = open(path, "a", buffering=1)

    def __call__(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if self._fh is not None:
            self._fh.write(line + "\n")
            self._fh.flush()


_log = _Log()


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


class EvalConfigError(RuntimeError):
    pass


class EvalSetup:
    """The resolved evaluation: an ESConfig plus the eval-only settings."""

    def __init__(self, cfg: ESConfig, base_cfg: ESConfig, ev: dict[str, Any],
                 config_path: Path, base_config_path: Path) -> None:
        self.cfg = cfg
        self.base_cfg = base_cfg
        self.ev = ev
        self.config_path = config_path
        self.base_config_path = base_config_path

        self.eval_dir = Path(ev.get("eval_dir", "es/eval"))
        self.checkpoint_dir = Path(ev.get("checkpoint_dir", "es/state/checkpoints"))
        self.episodes_per_condition = int(ev.get("episodes_per_condition", 64))
        self.seed_base = int(ev.get("eval_seed_base", DEFAULT_EVAL_SEED_BASE))
        self.seed_stride = int(
            ev.get("eval_seed_condition_stride", DEFAULT_CONDITION_STRIDE))
        self.iteration_tag = int(ev.get("eval_iteration_tag", 9999))
        self.conditions = list(ev.get("conditions") or [])
        self.probe = dict(ev.get("control_probe") or {})

    # -- derived ----------------------------------------------------------
    @property
    def condition_names(self) -> list[str]:
        return [c["name"] for c in self.conditions]

    @property
    def vector_ids(self) -> list[str]:
        return self.cfg.candidate_ids()

    def id_of(self, name: str) -> str:
        return self.vector_ids[self.condition_names.index(name)]

    @property
    def log_dir(self) -> Path:
        return self.eval_dir / "logs"

    @property
    def results_path(self) -> Path:
        return self.eval_dir / "eval_results.jsonl"

    @property
    def summary_path(self) -> Path:
        return self.eval_dir / "eval_summary.json"

    @property
    def done_path(self) -> Path:
        return self.eval_dir / "DONE"

    def seeds(self, condition: str) -> list[int]:
        c = self.condition_names.index(condition)
        return [
            self.seed_base + self.seed_stride * c + k
            for k in range(self.episodes_per_condition)
        ]

    def seed_for(self, condition: str, epoch: int) -> int:
        """Epoch is inspect's 1-based epoch index."""
        return self.seeds(condition)[int(epoch) - 1]


def load_setup(path: str | Path) -> EvalSetup:
    path = Path(path)
    doc = json.loads(path.read_text())
    unknown = sorted(set(doc) - {"base_config", "overrides", "eval"}
                     - {k for k in doc if k.startswith("_")})
    if unknown:
        raise EvalConfigError(f"unknown top-level keys in {path}: {unknown}")

    base_path = Path(doc.get("base_config", "es/run_config.json"))
    base_cfg = ESConfig.load(base_path)

    overrides = dict(doc.get("overrides") or {})
    bad = sorted(set(overrides) - ALLOWED_OVERRIDES)
    if bad:
        raise EvalConfigError(
            f"{path} overrides {bad}, which are not in ALLOWED_OVERRIDES. "
            f"Every field outside that set defines what an episode is, and "
            f"changing one here would make this evaluation incomparable with "
            f"the training it is evaluating."
        )
    merged = base_cfg.to_dict()
    merged.update(overrides)
    cfg = ESConfig.from_dict(merged)

    # parity, checked rather than asserted in prose
    diffs = {
        k: (getattr(base_cfg, k), getattr(cfg, k))
        for k in PARITY_FIELDS
        if getattr(base_cfg, k) != getattr(cfg, k)
    }
    if diffs:
        raise EvalConfigError(
            f"eval config differs from {base_path} on episode-defining "
            f"fields: {diffs}"
        )

    setup = EvalSetup(cfg, base_cfg, dict(doc.get("eval") or {}), path,
                      base_path)

    # structural checks that are cheaper to fail on now than at deploy time
    if len(setup.conditions) != cfg.n_candidates:
        raise EvalConfigError(
            f"{len(setup.conditions)} conditions but n_candidates="
            f"{cfg.n_candidates} (n_pairs={cfg.n_pairs}); they must match, the "
            f"conditions ARE the candidates of this one-shot batch"
        )
    if setup.episodes_per_condition != cfg.episodes_per_candidate:
        raise EvalConfigError(
            f"episodes_per_condition={setup.episodes_per_condition} but "
            f"episodes_per_candidate={cfg.episodes_per_candidate}; the second "
            f"is what inspect actually runs as `epochs`"
        )
    sources = [c.get("source") for c in setup.conditions]
    if sources != ["checkpoint", "zero"]:
        raise EvalConfigError(
            f"conditions must be exactly [checkpoint, zero], got {sources}")
    state_dir = Path(cfg.state_dir).resolve()
    trainer_state = Path(base_cfg.state_dir).resolve()
    if state_dir == trainer_state:
        raise EvalConfigError(
            f"the eval's state_dir is the trainer's ({state_dir}). The "
            f"evaluator writes no checkpoint, but it must not even be able to."
        )
    return setup


# ---------------------------------------------------------------------------
# the checkpoint
# ---------------------------------------------------------------------------


def latest_checkpoint(checkpoint_dir: Path) -> Path:
    """The highest-numbered `ckpt_*.npz`, by its integer index, not by mtime.

    The numbered copies are immutable and written atomically by
    `checkpoint.save` (temp file + rename), so reading one cannot race a
    trainer that is still running -- which is the whole reason this reads a
    numbered file and never `checkpoint.npz`.
    """
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"no checkpoint directory at {checkpoint_dir}")
    cands: list[tuple[int, Path]] = []
    for p in checkpoint_dir.glob("ckpt_*.npz"):
        stem = p.stem[len("ckpt_"):]
        if stem.isdigit():
            cands.append((int(stem), p))
    if not cands:
        raise FileNotFoundError(f"no ckpt_*.npz in {checkpoint_dir}")
    return max(cands)[1]


def read_checkpoint_theta(path: Path) -> dict[str, Any]:
    """theta and provenance, read-only. Never touches the trainer's state."""
    with np.load(path, allow_pickle=False) as data:
        theta = np.asarray(data["theta"], dtype=np.float64)
        meta = json.loads(str(data["meta"]))
    return {
        "path": str(path),
        "iteration": int(meta["iteration"]),
        "adam_t": int(meta["adam_t"]),
        "theta": theta,
        "theta_norm": float(np.linalg.norm(theta)),
        "sha256_of_file": _sha256_file(path),
        "config_seed": int(meta["config"]["seed"]),
        "n_iteration_records": len(meta.get("iteration_records") or []),
    }


def _sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------


def seed_disjointness_report(setup: EvalSetup) -> dict[str, Any]:
    """Prove the eval identifier range cannot collide with a training seed.

    Training derives no per-episode seed at all (see the module docstring), so
    the honest training set is the single integer `cfg.seed`. That is too weak
    a thing to check against, so the check is made against an envelope wide
    enough to contain every `cfg.seed + iteration*A + candidate*B + epoch`
    derivation with A up to 10^5 over the run's actual ranges.
    """
    train_seed = int(setup.base_cfg.seed)
    lo, hi = train_seed, train_seed + TRAINING_SEED_ENVELOPE
    envelope = range(lo, hi + 1)

    all_eval: list[int] = []
    per_condition: dict[str, list[int]] = {}
    for name in setup.condition_names:
        s = setup.seeds(name)
        per_condition[name] = s
        all_eval.extend(s)

    inside = [s for s in all_eval if s in envelope]
    dupes = [s for s, n in Counter(all_eval).items() if n > 1]
    also = sorted({train_seed, int(setup.base_cfg.synthetic_target_seed)})
    collisions = sorted(set(all_eval) & set(also))

    report = {
        "training_seed": train_seed,
        "training_seed_role": (
            "seeds the ES perturbation Generator only; never reaches a rollout"),
        "training_other_seeds": also,
        "training_envelope": [lo, hi],
        "training_envelope_rationale": (
            f"cfg.seed + f(iteration<=50, candidate<=16, epoch<=8) for any "
            f"multiplier up to 1e5 lies in this range"),
        "n_eval_seeds": len(all_eval),
        "eval_seed_min": min(all_eval),
        "eval_seed_max": max(all_eval),
        "eval_seeds_inside_training_envelope": inside,
        "eval_seed_duplicates": dupes,
        "eval_seed_collisions_with_named_training_seeds": collisions,
        "per_condition_first5": {k: v[:5] for k, v in per_condition.items()},
        "per_condition_len": {k: len(v) for k, v in per_condition.items()},
        "disjoint": not inside and not dupes and not collisions,
        "seed_is_functional": False,
        "seed_semantics": (
            "stable per-episode identifier, NOT a value passed to any RNG; the "
            "ctfish pipeline seeds no episode, in training or here"),
        "scheme": (
            f"seed(condition c, epoch k) = {setup.seed_base} + "
            f"{setup.seed_stride}*c + (k-1), c=index in "
            f"{setup.condition_names}, k = inspect epoch (1-based)"),
    }
    if not report["disjoint"]:
        raise EvalConfigError(
            f"the eval seed scheme is not provably disjoint from training: "
            f"{json.dumps({k: report[k] for k in ('eval_seeds_inside_training_envelope', 'eval_seed_duplicates', 'eval_seed_collisions_with_named_training_seeds')})}"
        )
    return report


# ---------------------------------------------------------------------------
# building the two conditions
# ---------------------------------------------------------------------------


def build_thetas(setup: EvalSetup, ckpt: dict[str, Any]) -> np.ndarray:
    thetas = np.zeros((setup.cfg.n_candidates, setup.cfg.hidden_size),
                      dtype=np.float64)
    for i, cond in enumerate(setup.conditions):
        if cond["source"] == "checkpoint":
            if ckpt["theta"].shape != (setup.cfg.hidden_size,):
                raise EvalConfigError(
                    f"checkpoint theta has shape {ckpt['theta'].shape}, "
                    f"expected ({setup.cfg.hidden_size},)")
            thetas[i] = ckpt["theta"]
        elif cond["source"] == "zero":
            thetas[i] = 0.0  # vecdir.serving_form's explicit zero case
        else:
            raise EvalConfigError(f"unknown condition source {cond['source']!r}")
    return thetas


def build_and_validate(setup: EvalSetup, thetas: np.ndarray, *, created_at: str
                       ) -> tuple[list[dict[str, Any]], np.ndarray]:
    """`deploy.build_candidates` -- writes both directories to a temp tree and
    loads them back through `vectorfmt` AND `vllm_steering.store` (`store.read`,
    `store.matrix`). Purely local: no pod, no box, no network."""
    records, directions = build_candidates(
        thetas, setup.cfg, iteration=setup.iteration_tag, created_at=created_at)

    names = setup.condition_names
    for i, rec in enumerate(records):
        rec["condition"] = names[i]
        rec["source"] = setup.conditions[i]["source"]

    control = next(r for r in records if r["source"] == "zero")
    if control["steer_strength"] != 0.0 or control["theta_norm"] != 0.0:
        raise EvalConfigError(
            f"the control was built at steer_strength="
            f"{control['steer_strength']!r} / ||theta||="
            f"{control['theta_norm']!r}; it must be exactly 0.0 or it is not a "
            f"control")
    for rec in records:
        if rec["block"] != setup.cfg.expected_block:
            raise EvalConfigError(
                f"{rec['id']} built for block {rec['block']}, expected "
                f"{setup.cfg.expected_block}")
    return records, directions


# ---------------------------------------------------------------------------
# is the control really unsteered?
# ---------------------------------------------------------------------------


def probe_control_unsteered(setup: EvalSetup, control_id: str) -> dict[str, Any]:
    """Greedy-decode the same prompt twice and require identical text:

      * once with NO steering arguments at all -- the base model;
      * once naming the control vector at `steer_strength: 0.0`.

    The manifest cannot answer this question. It describes the directory
    (`vector_norm`, `scale`), and the magnitude of a steered request lives
    entirely in the request's strength; `vllm_steering.patch._steering` computes
    `delta = alpha * row`, so alpha == 0.0 is arithmetically the unsteered
    model. This makes that an observation rather than a reading of the source.

    Repeated because temperature-0 vLLM is only deterministic up to batching;
    one exact match across the attempts is enough to establish the identity.

    A mismatch is WARN-ONLY by default (`--strict-probe` to abort): losing an
    overnight batch to greedy-decode nondeterminism costs more than carrying a
    flagged control through to the report, and the outcome is written into both
    `control_probe.json` and `eval_summary.json`.
    """
    url = setup.cfg.steered_base_url.rstrip("/") + "/chat/completions"
    prompt = str(setup.probe.get(
        "prompt", "In one short paragraph, describe what a chess engine does."))
    max_tokens = int(setup.probe.get("max_tokens", 64))
    attempts = int(setup.probe.get("attempts", 3))

    def _call(xargs: dict[str, Any] | None) -> str:
        body: dict[str, Any] = {
            "model": setup.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        if xargs is not None:
            body["vllm_xargs"] = xargs
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer "
                                      + os.environ.get("STEERED_API_KEY", "EMPTY")},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = json.loads(resp.read().decode())
        return payload["choices"][0]["message"]["content"] or ""

    trials = []
    for attempt in range(attempts):
        base = _call(None)
        zero = _call({"steer_vector": control_id, "steer_strength": 0.0})
        trials.append({"attempt": attempt, "identical": base == zero,
                       "base_text": base[:400], "control_text": zero[:400]})
        if base == zero:
            break

    report = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "control_vector": control_id,
        "control_steer_strength": 0.0,
        "n_attempts": len(trials),
        "identical_to_unsteered": any(t["identical"] for t in trials),
        "trials": trials,
        "note": (
            "the strength-0 request was ACCEPTED (middleware.problem passes a "
            "vector with a finite strength, including 0.0) and produced the "
            "same greedy continuation as the request with no steering at all"),
    }
    return report


# ---------------------------------------------------------------------------
# the batch
# ---------------------------------------------------------------------------


def run_eval_batch(records: list[dict[str, Any]], setup: EvalSetup) -> list[Any]:
    """One inspect batch, both conditions concurrently.

    Mirrors `rollouts.run_iteration_rollouts` -- including the `max_tasks` trap
    documented there: both conditions carry the same model NAME, so inspect
    would run them strictly one after the other unless `max_tasks` is passed.
    It is not a call into that function only because the metadata written into
    the eval log should say what this run is.
    """
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import GenerateConfig, get_model
    from ctfish import ctfish

    from .rollouts import task_args

    cfg = setup.cfg
    setup.log_dir.mkdir(parents=True, exist_ok=True)

    models = [
        get_model(
            cfg.model_name,
            config=GenerateConfig(max_connections=cfg.max_connections),
            steer_vector=rec["id"],
            steer_strength=rec["steer_strength"],
        )
        for rec in records
    ]

    return inspect_eval(
        ctfish(**task_args(cfg)),
        model=models,
        max_tasks=len(models),          # explicit, always: see rollouts.py
        max_samples=cfg.max_samples,
        max_sandboxes=cfg.max_sandboxes,
        log_dir=str(setup.log_dir),
        fail_on_error=cfg.fail_on_error,
        retry_on_error=cfg.retry_on_error,
        log_level="info",
        display="plain",
        metadata={
            "run": "es-final-eval",
            "eval_iteration_tag": setup.iteration_tag,
            "episodes_per_condition": setup.episodes_per_condition,
            "conditions": [
                {"condition": r["condition"], "source": r["source"],
                 "id": r["id"], "steer_strength": r["steer_strength"],
                 "theta_norm": r["theta_norm"], "sha256": r["vector_npy_sha256"]}
                for r in records
            ],
            "activation_norm": cfg.activation_norm,
            "max_sandboxes": cfg.max_sandboxes,
            "max_connections": cfg.max_connections,
            "seed_scheme": (
                f"identifier only: {setup.seed_base} + {setup.seed_stride}*c "
                f"+ (epoch-1)"),
        },
    )


def episodes_from_logs(logs: list[Any], records: list[dict[str, Any]],
                       setup: EvalSetup) -> list[dict[str, Any]]:
    """Flatten inspect logs into per-episode records, checked against `records`.

    The per-episode extraction is `rollouts._episode_record` itself -- imported
    rather than reimplemented, because the reward definition (`won_game +
    escalation_level / 6`, with judge errors excluded rather than scored 0) must
    be the same function that produced every training number.
    """
    from .rollouts import _episode_record

    by_id = {r["id"]: r for r in records}
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for log in logs:
        margs = log.eval.model_args or {}
        vid = str(margs.get("steer_vector"))
        if vid not in by_id:
            raise RuntimeError(
                f"eval log {log.location} is for steer_vector={vid!r}, which is "
                f"not one of {list(by_id)}")
        rec = by_id[vid]
        strength = float(margs.get("steer_strength"))
        if abs(strength - rec["steer_strength"]) > 1e-12:
            raise RuntimeError(
                f"log for {vid} ({rec['condition']}) ran at steer_strength="
                f"{strength!r}, but the condition is {rec['steer_strength']!r}. "
                f"The log and the vector disagree about what was evaluated.")
        seen.add(vid)
        for sample in (log.samples or []):
            ep = _episode_record(sample, setup.cfg)
            epoch = int(ep.get("epoch") or 0)
            out.append({
                "condition": rec["condition"],
                "source": rec["source"],
                "vector_id": vid,
                "steer_strength": rec["steer_strength"],
                "theta_norm": rec["theta_norm"],
                "vector_npy_sha256": rec["vector_npy_sha256"],
                "seed": setup.seed_for(rec["condition"], epoch) if epoch else None,
                "seed_is_functional": False,
                "log": log.location,
                "log_status": log.status,
                **ep,
            })
    missing = [f"{by_id[i]['condition']}({i})" for i in by_id if i not in seen]
    if missing:
        raise RuntimeError(
            f"no eval log came back for {missing}; the batch did not run both "
            f"conditions and the comparison would be one-sided")
    return out


def write_results(episodes: list[dict[str, Any]], path: Path) -> None:
    """Append-and-fsync per record: a crash leaves every earlier line valid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for ep in episodes:
            fh.write(json.dumps(ep, allow_nan=False, sort_keys=True) + "\n")
            fh.flush()
        os.fsync(fh.fileno())


def summarise(episodes: list[dict[str, Any]], records: list[dict[str, Any]],
              setup: EvalSetup, extra: dict[str, Any]) -> dict[str, Any]:
    per: dict[str, Any] = {}
    for rec in records:
        name = rec["condition"]
        eps = [e for e in episodes if e["condition"] == name]
        valid = [e["reward"] for e in eps if e["reward"] is not None]
        levels = [e["escalation_level"] for e in eps
                  if e.get("escalation_level") is not None]
        steps = [e["steps"] for e in eps if e.get("steps") is not None]
        per[name] = {
            "vector_id": rec["id"],
            "source": rec["source"],
            "steer_strength": rec["steer_strength"],
            "theta_norm": rec["theta_norm"],
            "vector_npy_sha256": rec["vector_npy_sha256"],
            "n_episodes": len(eps),
            "n_expected": setup.episodes_per_condition,
            "n_valid": len(valid),
            "n_excluded": len(eps) - len(valid),
            "excluded": [
                {"sample_id": e["sample_id"], "epoch": e["epoch"],
                 "seed": e["seed"], "why": e["excluded_because"],
                 "error": e["error"]}
                for e in eps if e["reward"] is None
            ],
            "wins": sum(1 for e in eps if e.get("won")),
            "win_rate": (sum(1 for e in eps if e.get("won")) / len(eps)) if eps else None,
            "mean_reward": (sum(valid) / len(valid)) if valid else None,
            "sd_reward": statistics.stdev(valid) if len(valid) > 1 else (
                0.0 if valid else None),
            "sem_reward": (statistics.stdev(valid) / (len(valid) ** 0.5))
            if len(valid) > 1 else None,
            "mean_escalation_level": (sum(levels) / len(levels)) if levels else None,
            "escalation_distribution": dict(sorted(Counter(
                str(e.get("escalation_label")) for e in eps).items())),
            "mean_steps": (sum(steps) / len(steps)) if steps else None,
            "output_tokens_total": sum(int(e.get("output_tokens") or 0) for e in eps),
            "logs": sorted({e["log"] for e in eps}),
            "seeds_first5": setup.seeds(name)[:5],
            "seeds_n": len(setup.seeds(name)),
        }

    a, b = setup.condition_names[0], setup.condition_names[1]
    diff = None
    if per[a]["mean_reward"] is not None and per[b]["mean_reward"] is not None:
        diff = per[a]["mean_reward"] - per[b]["mean_reward"]

    return {
        "run": "es-final-eval",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": str(setup.config_path),
        "es_config": setup.cfg.to_dict(),
        "n_episodes_total": len(episodes),
        "per_condition": per,
        "comparison": {
            "conditions": [a, b],
            "mean_reward_difference": diff,
            "win_difference": per[a]["wins"] - per[b]["wins"],
            "note": "descriptive only; no test is applied here",
        },
        **extra,
    }


# ---------------------------------------------------------------------------
# safety: do not clobber a running trainer
# ---------------------------------------------------------------------------


def trainer_processes() -> list[str]:
    try:
        proc = subprocess.run(["pgrep", "-af", "es.train"],
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    lines = [ln for ln in proc.stdout.splitlines()
             if "pgrep" not in ln and ln.strip()]
    return lines


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------


def dry_run(setup: EvalSetup) -> int:
    print("=" * 72)
    print("ES held-out final evaluation -- DRY RUN (nothing is contacted)")
    print("=" * 72)
    print(f"config             : {setup.config_path}")
    print(f"base config        : {setup.base_config_path}  "
          f"(parity checked on {len(PARITY_FIELDS)} episode-defining fields)")

    ckpt_path = latest_checkpoint(setup.checkpoint_dir)
    ckpt = read_checkpoint_theta(ckpt_path)
    print()
    print("--- checkpoint ---")
    print(f"checkpoint dir     : {setup.checkpoint_dir}")
    print(f"chosen checkpoint  : {ckpt['path']}")
    print(f"  iteration        : {ckpt['iteration']}  (adam_t={ckpt['adam_t']}, "
          f"{ckpt['n_iteration_records']} iteration records)")
    print(f"  ||theta||        : {ckpt['theta_norm']!r}")
    print(f"  theta_norm_cap   : {setup.cfg.theta_norm_cap}")
    print(f"  file sha256      : {ckpt['sha256_of_file']}")
    print(f"  served strength  : {ckpt['theta_norm']!r}  "
          f"(pre-scaled by A={setup.cfg.activation_norm}: "
          f"||delta|| = ||theta||*A = "
          f"{ckpt['theta_norm'] * setup.cfg.activation_norm!r})")

    print()
    print("--- conditions / model ids ---")
    thetas = build_thetas(setup, ckpt)
    for i, cond in enumerate(setup.conditions):
        vid = setup.vector_ids[i]
        norm = float(np.linalg.norm(thetas[i]))
        print(f"[{cond['name']:<7}] inspect model = {setup.cfg.model_name}  "
              f"-M steer_vector={vid} -M steer_strength={norm!r}")
        print(f"          served model id on the server = "
              f"{setup.cfg.model} / vector {vid}   (source={cond['source']})")
    print(f"served simultaneously from ONE restart of "
          f"{setup.cfg.server_manifest_url}")

    print()
    print("--- batch ---")
    print(f"episodes per condition : {setup.episodes_per_condition}")
    print(f"conditions             : {len(setup.conditions)}")
    print(f"episodes in the batch  : "
          f"{setup.episodes_per_condition * len(setup.conditions)}")
    print(f"variant={setup.cfg.variant} max_steps={setup.cfg.max_steps} "
          f"command_timeout={setup.cfg.command_timeout} "
          f"generate_config={setup.cfg.generate_config}")
    print(f"max_sandboxes={setup.cfg.max_sandboxes} "
          f"max_samples={setup.cfg.max_samples} "
          f"max_connections={setup.cfg.max_connections} "
          f"fail_on_error={setup.cfg.fail_on_error} "
          f"retry_on_error={setup.cfg.retry_on_error}")
    print(f"judge={setup.cfg.escalation_judge_model} "
          f"temperature={setup.cfg.escalation_judge_temperature} "
          f"reward = won_game + escalation_level/"
          f"{setup.cfg.reward_escalation_divisor}")

    print()
    print("--- held-out episode identifiers ---")
    rep = seed_disjointness_report(setup)
    print(f"scheme    : {rep['scheme']}")
    print(f"semantics : {rep['seed_semantics']}")
    for name in setup.condition_names:
        s = setup.seeds(name)
        print(f"[{name:<7}] n={len(s)}  first 5 = {s[:5]}  last = {s[-1]}")
    print(f"training seed {rep['training_seed']} "
          f"({rep['training_seed_role']}); envelope {rep['training_envelope']}")
    print(f"eval range [{rep['eval_seed_min']}, {rep['eval_seed_max']}]  "
          f"inside envelope: {rep['eval_seeds_inside_training_envelope']}  "
          f"duplicates: {rep['eval_seed_duplicates']}  "
          f"DISJOINT={rep['disjoint']}")

    print()
    print("--- vector directories (built in a temp tree, validated locally) ---")
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    records, directions = build_and_validate(setup, thetas, created_at=created_at)
    for rec in records:
        print(f"[{rec['condition']:<7}] id={rec['id']} block={rec['block']} "
              f"hidden={directions.shape[1]} ||v||={rec['vector_norm']!r} "
              f"scale={rec['scale']!r} strength={rec['steer_strength']!r} "
              f"sha256={rec['vector_npy_sha256'][:16]}")
    print(f"validated through steering_vectors.vectorfmt and "
          f"vllm_steering.store (read + matrix); expected_block="
          f"{setup.cfg.expected_block} hidden_size={setup.cfg.hidden_size}")

    print()
    print("--- outputs ---")
    for p in (setup.log_dir, setup.results_path, setup.summary_path,
              setup.done_path, setup.eval_dir / "eval.log"):
        print(f"  {p}")
    print()
    print("would deploy + restart vLLM and run the batch. DRY RUN: stopping "
          "here, nothing was contacted.")
    return 0


def run(setup: EvalSetup, *, force: bool = False,
        strict_probe: bool = False) -> int:
    # Refuse BEFORE creating anything: a run that is not allowed to start
    # should leave no directory behind suggesting it did.
    if setup.done_path.exists():
        raise SystemExit(
            f"{setup.done_path} already exists: this evaluation has already "
            f"run. Move it aside to re-run.")
    running = trainer_processes()
    if running and not force:
        raise SystemExit(
            "the ES trainer looks like it is still running:\n  "
            + "\n  ".join(running)
            + "\nStarting the evaluation restarts the vLLM server and would "
              "destroy the iteration in flight. Wait for it, or pass --force "
              "if you know it is finished.")

    setup.eval_dir.mkdir(parents=True, exist_ok=True)
    _log.attach(setup.eval_dir / "eval.log")
    if running:
        _log(f"--force: proceeding with these processes alive: {running}")

    _log(f"python {platform.python_version()} numpy {np.__version__}")
    _log(f"config {setup.config_path}")

    ckpt_path = latest_checkpoint(setup.checkpoint_dir)
    ckpt = read_checkpoint_theta(ckpt_path)
    _log(f"CHECKPOINT {ckpt['path']} iteration={ckpt['iteration']} "
         f"adam_t={ckpt['adam_t']} ||theta||={ckpt['theta_norm']!r} "
         f"sha256={ckpt['sha256_of_file']}")

    seeds = seed_disjointness_report(setup)
    _log("SEEDS " + json.dumps({
        k: seeds[k] for k in ("scheme", "n_eval_seeds", "eval_seed_min",
                              "eval_seed_max", "per_condition_first5",
                              "disjoint", "seed_is_functional")}))

    thetas = build_thetas(setup, ckpt)
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    records, directions = build_and_validate(setup, thetas, created_at=created_at)
    for rec in records:
        _log(f"CANDIDATE {rec['condition']}: id={rec['id']} "
             f"strength={rec['steer_strength']!r} block={rec['block']} "
             f"sha256={rec['vector_npy_sha256']}")

    deploy_dir = setup.eval_dir / "deploy"
    pod_report = deploy_and_restart(
        records, directions, setup.cfg,
        iteration=setup.iteration_tag, created_at=created_at,
        work_dir=deploy_dir)
    _log(f"pod rebuilt {pod_report['ids']} block={pod_report['block']} "
         f"store_digest={pod_report['store_digest'][:16]}")

    verify = verify_served_set(records, setup.cfg)
    _log("VERIFIED served set: " + json.dumps({
        k: verify[k] for k in ("health_wait_s", "manifest_block",
                               "manifest_ids", "sha256_match",
                               "manifest_digest", "scales", "norms")}))

    control_id = setup.id_of("control")
    probe: dict[str, Any] = {"enabled": False, "outcome": "not run"}
    if setup.probe.get("enabled", True):
        probe = probe_control_unsteered(setup, control_id)
        probe["strict"] = strict_probe
        probe["outcome"] = ("match" if probe["identical_to_unsteered"]
                            else "mismatch")
        _log(f"CONTROL PROBE identical_to_unsteered="
             f"{probe['identical_to_unsteered']} after "
             f"{probe['n_attempts']} attempt(s)")
        if not probe["identical_to_unsteered"]:
            # WARN-ONLY by default. A mismatch here is far more likely to be
            # greedy-decode nondeterminism (vLLM is only deterministic up to
            # batching) than a real steering leak: `middleware.problem` admits
            # a finite strength including 0.0, and `patch._steering` /
            # `_fill_steering` compute `delta = alpha * row`, so alpha == 0.0
            # is the same arithmetic as the reserved zero row `store.matrix`
            # hands an unsteered request. Aborting a three-hour overnight batch
            # on that is the more expensive mistake. The outcome is recorded in
            # control_probe.json AND in eval_summary.json either way, so a
            # mismatch cannot reach a report unnoticed.
            probe["warning"] = (
                "the strength-0 control did not reproduce the unsteered "
                "model's greedy continuation in "
                f"{probe['n_attempts']} attempt(s). Most likely greedy-decode "
                "nondeterminism; a real steering leak at strength 0 would "
                "contradict the forward path. TREAT THE CONTROL CONDITION AS "
                "UNVERIFIED until the transcripts are inspected.")
            for _ in range(3):
                _log("!!! CONTROL PROBE MISMATCH: " + probe["warning"])
            if strict_probe:
                (setup.eval_dir / "control_probe.json").write_text(
                    json.dumps(probe, indent=2) + "\n")
                raise DeploymentError(
                    "--strict-probe: refusing to run with an unverified "
                    "control. See es/eval/control_probe.json.")
    (setup.eval_dir / "control_probe.json").write_text(
        json.dumps(probe, indent=2) + "\n")

    (setup.eval_dir / "verification.json").write_text(
        json.dumps({"pod": pod_report, "manifest": verify,
                    "checkpoint": {k: v for k, v in ckpt.items() if k != "theta"},
                    "seeds": seeds, "control_probe": probe}, indent=2) + "\n")

    _log(f"starting ONE batch: {len(records)} conditions x "
         f"{setup.episodes_per_condition} episodes = "
         f"{len(records) * setup.episodes_per_condition}")
    started = time.time()
    logs = run_eval_batch(records, setup)
    wall = time.time() - started
    _log(f"batch finished in {wall:.1f}s, {len(logs)} logs")

    episodes = episodes_from_logs(logs, records, setup)
    write_results(episodes, setup.results_path)
    _log(f"wrote {len(episodes)} episode records to {setup.results_path}")

    summary = summarise(episodes, records, setup, {
        "checkpoint": {k: v for k, v in ckpt.items() if k != "theta"},
        "seeds": seeds,
        "verification": verify,
        "control_probe": probe,
        "control_probe_outcome": probe.get("outcome"),
        "control_verified_unsteered": bool(probe.get("identical_to_unsteered")),
        "wall_clock_s": round(wall, 1),
    })
    setup.summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n")
    _log("SUMMARY " + json.dumps({
        c: {k: summary["per_condition"][c][k]
            for k in ("n_episodes", "n_valid", "wins", "mean_reward",
                      "mean_escalation_level")}
        for c in setup.condition_names}))

    setup.done_path.write_text(json.dumps({
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_episodes": len(episodes),
        "results": str(setup.results_path),
        "summary": str(setup.summary_path),
    }, indent=2) + "\n")
    _log(f"DONE -> {setup.done_path}")
    return 0


def rebuild_from_logs(setup: EvalSetup) -> int:
    """Recover `eval_results.jsonl` from whatever inspect already wrote.

    The batch's episodes only become readable when `inspect_eval` returns, so a
    crash between the last sandbox finishing and the jsonl being written would
    otherwise throw the whole batch away. The inspect logs are on disk either
    way; this re-reads them and redoes only the cheap part.
    """
    from inspect_ai.log import list_eval_logs, read_eval_log

    _log.attach(setup.eval_dir / "eval.log")
    ckpt = read_checkpoint_theta(latest_checkpoint(setup.checkpoint_dir))
    thetas = build_thetas(setup, ckpt)
    records, _ = build_and_validate(
        setup, thetas,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    infos = list_eval_logs(str(setup.log_dir))
    logs = [read_eval_log(i) for i in infos]
    _log(f"read {len(logs)} inspect logs from {setup.log_dir}")
    episodes = episodes_from_logs(logs, records, setup)
    write_results(episodes, setup.results_path)
    summary = summarise(episodes, records, setup, {
        "checkpoint": {k: v for k, v in ckpt.items() if k != "theta"},
        "seeds": seed_disjointness_report(setup),
        "rebuilt_from_logs": True,
    })
    setup.summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n")
    _log(f"rebuilt {len(episodes)} episodes -> {setup.results_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="es/eval_config.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be served and stop; contacts nothing")
    ap.add_argument("--from-logs", action="store_true",
                    help="rebuild the results from inspect logs already on disk")
    ap.add_argument("--force", action="store_true",
                    help="run even though an es.train process is alive")
    ap.add_argument("--strict-probe", action="store_true",
                    help="abort if the strength-0 control does not reproduce "
                         "the unsteered model's greedy continuation. Default "
                         "is warn-only: the mismatch is logged loudly and "
                         "recorded in control_probe.json and eval_summary.json, "
                         "but the batch runs.")
    args = ap.parse_args(argv)

    setup = load_setup(args.config)
    if args.dry_run:
        return dry_run(setup)
    if args.from_logs:
        return rebuild_from_logs(setup)
    return run(setup, force=args.force, strict_probe=args.strict_probe)


if __name__ == "__main__":
    sys.exit(main())
