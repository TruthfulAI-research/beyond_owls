"""Antithetic sampling, rank shaping, the gradient estimate, and Adam.

Kept apart from everything that touches a network so it can be tested against a
synthetic reward, which is the only way to check the one thing that is invisible
in a real run: the sign. An ES loop with the sign inverted produces a perfectly
healthy-looking trajectory that walks away from the objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def sample_directions(rng: np.random.Generator, n_pairs: int, dim: int) -> np.ndarray:
    """``(n_pairs, dim)`` unit-norm gaussian directions.

    Normalised rather than left as raw N(0, I): the perturbation size is then
    exactly sigma for every candidate, so sigma means one thing (a fraction of a
    typical activation) instead of a mean over a chi distribution. In 5120
    dimensions the two differ by well under a percent, but sigma is the quantity
    the whole parameterisation is stated in and it should not be approximate.
    """
    raw = rng.standard_normal((n_pairs, dim))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    if not np.all(norms > 0):
        raise ValueError("drew a zero direction, which has no direction")
    return raw / norms


def antithetic_candidates(
    theta: np.ndarray, eps: np.ndarray, sigma: float
) -> np.ndarray:
    """``(2P, dim)``: ``theta + sigma*eps_i`` at 2i and ``theta - sigma*eps_i`` at 2i+1.

    The interleaved order is the contract the rest of the package indexes by:
    candidate ``2i`` is the plus arm of pair ``i`` and ``2i+1`` its minus arm.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    n_pairs, dim = eps.shape
    out = np.empty((2 * n_pairs, dim), dtype=np.float64)
    out[0::2] = theta + sigma * eps
    out[1::2] = theta - sigma * eps
    return out


def centred_ranks(rewards: np.ndarray) -> np.ndarray:
    """Rewards mapped to evenly spaced values in [-0.5, +0.5] by rank.

    The standard OpenAI-ES fitness shaping: only the ordering of the candidates
    survives, so a single outlying episode cannot dominate a step and a reward
    distribution that is nearly constant (the baseline's was: 14 of 16 episodes
    at the same value) still yields a well-scaled update.

    Ties take the *average* of the ranks they span, which matters here more than
    it usually does. With a near-degenerate reward, ties are the common case,
    and a tie broken arbitrarily would put a real gradient contribution behind
    two candidates the measurement could not tell apart. Averaged, tied
    candidates get identical shaped values and their pair contributes exactly
    zero.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 1 or rewards.size < 2:
        raise ValueError(f"need at least 2 rewards, got shape {rewards.shape}")
    if not np.all(np.isfinite(rewards)):
        raise ValueError("rewards contain non-finite values")
    n = rewards.size
    order = np.argsort(rewards, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    # average ranks within each group of equal rewards
    sorted_r = rewards[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_r[j + 1] == sorted_r[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks / (n - 1) - 0.5


def gradient_estimate(
    shaped: np.ndarray, eps: np.ndarray, sigma: float
) -> np.ndarray:
    """The ES ascent direction.

    ``g = mean_i [ (s_plus_i - s_minus_i) / 2 ] * eps_i / sigma``.

    Positive because ``shaped`` increases with reward: a pair whose plus arm
    scored better contributes ``+eps_i``, which is the direction that arm sat
    in. Adam is then an ASCENT on this.
    """
    shaped = np.asarray(shaped, dtype=np.float64)
    n_pairs = eps.shape[0]
    if shaped.shape != (2 * n_pairs,):
        raise ValueError(f"shaped has shape {shaped.shape}, expected ({2 * n_pairs},)")
    weights = (shaped[0::2] - shaped[1::2]) / 2.0
    return (weights @ eps) / (n_pairs * sigma)


@dataclass
class AdamState:
    """Adam's moments and step count. Serialised whole into every checkpoint."""

    m: np.ndarray
    v: np.ndarray
    t: int = 0

    @classmethod
    def zeros(cls, dim: int) -> "AdamState":
        return cls(m=np.zeros(dim), v=np.zeros(dim), t=0)

    def to_dict(self) -> dict[str, Any]:
        return {"m": self.m, "v": self.v, "t": int(self.t)}


def adam_ascend(
    theta: np.ndarray,
    grad: np.ndarray,
    state: AdamState,
    *,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, AdamState]:
    """One Adam step that INCREASES the objective. Returns ``(theta, state)``.

    Written as an ascent rather than as "minimise the negative" so that there is
    no place in this package where a sign is flipped and then flipped back. The
    only ``+`` that matters is the one in ``theta + lr * ...``; `test_esmath.py`
    drives the whole path from a synthetic reward and asserts theta approaches
    the optimum, so an inverted convention here fails a test rather than
    producing a plausible run.
    """
    t = state.t + 1
    m = beta1 * state.m + (1 - beta1) * grad
    v = beta2 * state.v + (1 - beta2) * (grad * grad)
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    new_theta = theta + lr * m_hat / (np.sqrt(v_hat) + eps)
    return new_theta, AdamState(m=m, v=v, t=t)


def apply_norm_cap(
    theta: np.ndarray, cap: float | None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rescale ``theta`` to ``cap`` if it is longer, and report whether it bound.

    A rescale, not a componentwise clip: the direction Adam produced is the
    only thing the gradient estimate actually carries information about, and
    clipping components would rotate it. Returns the (possibly unchanged) theta
    and a record with the norm before, the norm after and the scale factor
    applied, which goes into `history.jsonl` verbatim.
    """
    norm = float(np.linalg.norm(theta))
    if cap is None or norm <= cap:
        return theta, {"cap": cap, "bound": False, "norm_before_cap": norm,
                       "norm_after_cap": norm, "scale": 1.0}
    scale = cap / norm
    return theta * scale, {"cap": cap, "bound": True, "norm_before_cap": norm,
                           "norm_after_cap": float(cap), "scale": scale}


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    """Cosine similarity, or ``None`` when either side has no direction."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return None
    return float(np.dot(a, b) / (na * nb))
