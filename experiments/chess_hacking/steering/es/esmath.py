"""Antithetic sampling, rank shaping, the gradient estimate, and Adam."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def sample_directions(rng: np.random.Generator, n_pairs: int, dim: int) -> np.ndarray:
    """``(n_pairs, dim)`` unit-norm gaussian directions."""
    raw = rng.standard_normal((n_pairs, dim))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    if not np.all(norms > 0):
        raise ValueError("drew a zero direction, which has no direction")
    return raw / norms


def antithetic_candidates(
    theta: np.ndarray, eps: np.ndarray, sigma: float
) -> np.ndarray:
    """``(2P, dim)``: ``theta + sigma*eps_i`` at 2i and ``theta - sigma*eps_i`` at 2i+1."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    n_pairs, dim = eps.shape
    out = np.empty((2 * n_pairs, dim), dtype=np.float64)
    out[0::2] = theta + sigma * eps
    out[1::2] = theta - sigma * eps
    return out


def centred_ranks(rewards: np.ndarray) -> np.ndarray:
    """Rewards mapped to evenly spaced values in [-0.5, +0.5] by rank."""
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 1 or rewards.size < 2:
        raise ValueError(f"need at least 2 rewards, got shape {rewards.shape}")
    if not np.all(np.isfinite(rewards)):
        raise ValueError("rewards contain non-finite values")
    n = rewards.size
    order = np.argsort(rewards, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
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
    """The ES ascent direction."""
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
    """One Adam step that INCREASES the objective. Returns ``(theta, state)``."""
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
    """Rescale ``theta`` to ``cap`` if it is longer, and report whether it bound."""
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
