"""The whole resumable state of a run, written atomically once per iteration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import ESConfig
from .esmath import AdamState

CHECKPOINT_NAME = "checkpoint.npz"


@dataclass
class RunState:
    """Everything a resume needs. `iteration` is the NEXT iteration to run."""

    iteration: int
    theta: np.ndarray
    adam: AdamState
    rng_state: dict[str, Any]
    config: ESConfig
    episode_records: list[dict[str, Any]] = field(default_factory=list)
    iteration_records: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def fresh(cls, cfg: ESConfig) -> "RunState":
        if cfg.theta_init != "zero":
            raise ValueError(f"unsupported theta_init {cfg.theta_init!r}")
        rng = np.random.default_rng(cfg.seed)
        return cls(
            iteration=0,
            theta=np.zeros(cfg.hidden_size, dtype=np.float64),
            adam=AdamState.zeros(cfg.hidden_size),
            rng_state=rng.bit_generator.state,
            config=cfg,
        )

    def generator(self) -> np.random.Generator:
        """A Generator positioned exactly where the last iteration left it."""
        gen = np.random.default_rng()
        gen.bit_generator.state = self.rng_state
        return gen


def save(state: RunState, state_dir: str | Path) -> Path:
    """Write the latest checkpoint atomically plus an immutable numbered copy."""
    root = Path(state_dir)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)

    meta = json.dumps(
        {
            "iteration": int(state.iteration),
            "adam_t": int(state.adam.t),
            "rng_state": state.rng_state,
            "config": state.config.to_dict(),
            "episode_records": state.episode_records,
            "iteration_records": state.iteration_records,
        },
        allow_nan=False,
    )

    def _write(path: Path) -> None:
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez(
                fh,
                theta=state.theta,
                adam_m=state.adam.m,
                adam_v=state.adam.v,
                meta=np.array(meta),
            )
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    numbered = root / "checkpoints" / f"ckpt_{state.iteration:04d}.npz"
    _write(numbered)
    latest = root / CHECKPOINT_NAME
    _write(latest)
    return latest


def load(state_dir: str | Path) -> RunState:
    path = Path(state_dir) / CHECKPOINT_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no checkpoint at {path}")
    with np.load(path, allow_pickle=False) as data:
        theta = np.asarray(data["theta"], dtype=np.float64)
        m = np.asarray(data["adam_m"], dtype=np.float64)
        v = np.asarray(data["adam_v"], dtype=np.float64)
        meta = json.loads(str(data["meta"]))
    return RunState(
        iteration=int(meta["iteration"]),
        theta=theta,
        adam=AdamState(m=m, v=v, t=int(meta["adam_t"])),
        rng_state=meta["rng_state"],
        config=ESConfig.from_dict(meta["config"]),
        episode_records=meta["episode_records"],
        iteration_records=meta["iteration_records"],
    )


def exists(state_dir: str | Path) -> bool:
    return (Path(state_dir) / CHECKPOINT_NAME).is_file()
