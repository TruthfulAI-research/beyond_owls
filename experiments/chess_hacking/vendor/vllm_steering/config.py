"""Steering configuration, read from the environment.

What a launch chooses is now only *where the vectors are* and, optionally, which
of them to serve. The layer and the scale are no longer here: they are
properties of each vector, recorded in its own `meta.json` and derived from
there (`store.Vector`), so there is no longer a way to start a server with a
scale computed for one vector and a vector file holding another.

Read in every process the patch touches, and in the API server process as well —
`endpoint` and `middleware` read the same variables, so the manifest a client is
served and the arrays a worker adds come from one description of one directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

ENABLE_VAR = "STEER_ENABLE"


@dataclass(frozen=True)
class SteerConfig:
    vector_dir: str
    """Directory of vector directories: `0007/`, `0008/`, … (see `store`)."""

    vectors: str | None
    """Comma-separated ids to serve. `None` serves everything in the directory."""

    vector_arg: str
    """Key read from a request's `vllm_xargs` for the vector id."""

    arg: str
    """Key read from a request's `vllm_xargs` for the strength."""

    debug: bool
    """Log the per-step token counts, rows and strengths."""


def enabled() -> bool:
    return os.environ.get(ENABLE_VAR) == "1"


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set when {ENABLE_VAR}=1")
    return value


def config_from_env() -> SteerConfig:
    return SteerConfig(
        vector_dir=_require("STEER_VECTOR_DIR"),
        vectors=os.environ.get("STEER_VECTORS") or None,
        vector_arg=os.environ.get("STEER_VECTOR_ARG", "steer_vector"),
        arg=os.environ.get("STEER_ARG", "steer_strength"),
        debug=os.environ.get("STEER_DEBUG") == "1",
    )
