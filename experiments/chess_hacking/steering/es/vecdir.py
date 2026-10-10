"""Write a theta as a vector directory the repo's own reader accepts."""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from typing import Any

import numpy as np

from steering_vectors import vectorfmt
from steering_vectors.core import canonjson, clock, digest

ZERO_THETA_DIRECTION_INDEX = 0


def serving_form(
    theta: np.ndarray, *, hidden_size: int = 5120
) -> tuple[np.ndarray, float, float]:
    """``(direction_f32, steer_strength, ||theta||)`` for one theta."""
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (hidden_size,):
        raise ValueError(f"theta has shape {theta.shape}, expected ({hidden_size},)")
    if not np.all(np.isfinite(theta)):
        raise ValueError("theta contains non-finite values")
    norm = float(np.linalg.norm(theta))
    if norm == 0.0:
        direction = np.zeros(hidden_size, dtype=np.float32)
        direction[ZERO_THETA_DIRECTION_INDEX] = np.float32(1.0)
        return direction, 0.0, 0.0
    return (theta / norm).astype(np.float32), norm, norm


def npy_bytes(array: np.ndarray) -> bytes:
    """The exact `.npy` file contents for ``array``, as bytes."""
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    return buf.getvalue()


def _write_atomic_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


PROVENANCE = (
    "This directory is NOT a difference-of-means vector. It is one iterate of an "
    "evolution-strategies search over the layer-36 residual stream: the array is "
    "a direction produced by an optimiser, not by capturing activations. "
    "`meta.json` nevertheless records this module's position_convention and "
    "sign_convention verbatim, because steering_vectors.vectorfmt.validate_meta "
    "compares them for string equality and will not load a directory that says "
    "anything else. Read them as the conventions this vector is *used* under -- "
    "it is added to the residual stream at the input of block 36, at every token "
    "position -- and not as a claim about how it was derived. Likewise `capture` "
    "records no capture: n_checked is null on both sides, which is the format's "
    "way of making the absence of a claim explicit, and the prompt files are "
    "placeholders that exist because the format requires the files."
)


def write_vector_dir(
    theta: np.ndarray,
    vector_id: int | str,
    out_dir: str | os.PathLike[str],
    *,
    activation_norm: float,
    hidden_size_for_theta: int = 5120,
    **kwargs: Any,
) -> dict[str, Any]:
    """:func:`write_serving_dir` of :func:`serving_form` — the theta entry point."""
    direction, strength, theta_norm = serving_form(
        theta, hidden_size=hidden_size_for_theta
    )
    return write_serving_dir(
        direction,
        strength,
        theta_norm,
        vector_id,
        out_dir,
        activation_norm=activation_norm,
        **kwargs,
    )


def write_serving_dir(
    direction: np.ndarray,
    steer_strength: float,
    theta_norm: float,
    vector_id: int | str,
    out_dir: str | os.PathLike[str],
    *,
    activation_norm: float,
    layer: int = 36,
    n_layers: int = 64,
    hidden_size: int = 5120,
    model: str = "Qwen/Qwen3.6-27B",
    name: str | None = None,
    description: str | None = None,
    command: str = "steering/vecdir.py:write_vector_dir",
    vllm_version: str = "0.27.1",
    created_at: str | None = None,
) -> dict[str, Any]:
    """Write ``out_dir/<id>/`` for an already-resolved direction and strength."""
    vid = vectorfmt.vector_id(vector_id)
    direction = np.ascontiguousarray(direction, dtype="<f4")
    if direction.shape != (hidden_size,):
        raise ValueError(
            f"direction has shape {direction.shape}, expected ({hidden_size},)"
        )
    if not np.all(np.isfinite(direction)):
        raise ValueError("direction contains non-finite values")
    strength = float(steer_strength)
    theta_norm = float(theta_norm)

    vector_norm = float(np.linalg.norm(direction.astype(np.float64)))
    if not vector_norm > 0.0:
        raise ValueError(
            f"the served direction for vector {vid} has zero norm; "
            f"steer_scale would divide by it"
        )

    deltas = np.zeros((n_layers, hidden_size), dtype=np.float32)
    deltas[layer] = direction

    vector_blob = npy_bytes(direction)
    deltas_blob = npy_bytes(deltas)

    root = Path(out_dir) / vid
    root.mkdir(parents=True, exist_ok=True)

    pos_line = (
        '{"text": "placeholder: this vector was not derived from prompts; '
        'see README.md"}\n'
    )
    neg_line = (
        '{"text": "placeholder: this vector was not derived from prompts; '
        'see README.md"}\n'
    )
    (root / vectorfmt.POSITIVE_NAME).write_text(pos_line, encoding="utf-8")
    (root / vectorfmt.NEGATIVE_NAME).write_text(neg_line, encoding="utf-8")
    _write_atomic_bytes(root / vectorfmt.VECTOR_NAME, vector_blob)
    _write_atomic_bytes(root / vectorfmt.DELTAS_NAME, deltas_blob)

    per_layer_delta_norm = [0.0] * n_layers
    per_layer_delta_norm[layer] = vector_norm

    name = name or f"es-candidate-{vid}"
    description = description or (
        f"Evolution-strategies iterate served as a unit direction at layer "
        f"{layer}. The magnitude lives entirely in the request: theta is "
        f"realised as steer_strength = ||theta|| = {theta_norm!r} against "
        f"activation_norm_at_layer = {activation_norm!r}, so the perturbation "
        f"added to the residual stream is exactly ||theta|| * "
        f"{activation_norm!r} in the model's own units. {PROVENANCE}"
    )

    meta: dict[str, Any] = {
        "id": int(vid),
        "id_str": vid,
        "name": name,
        "description": description,
        "model": model,
        "layer": layer,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "position_convention": vectorfmt.POSITION_CONVENTION,
        "sign_convention": vectorfmt.SIGN_CONVENTION,
        "n_pos": 1,
        "n_neg": 1,
        "prompt_format": {
            side: {
                "kind": "text",
                "n": 1,
                "n_with_system": 0,
                "role_patterns": {"placeholder": 1},
            }
            for side in ("positive", "negative")
        },
        "vector_norm": vector_norm,
        "activation_norm_at_layer": float(activation_norm),
        "vector_norm_over_activation_norm": vector_norm / float(activation_norm),
        "per_layer_delta_norm": per_layer_delta_norm,
        "per_layer_mean_activation_norm": [float(activation_norm)] * n_layers,
        "created_at": created_at or clock.utc_now_iso(),
        "git_sha": None,
        "vllm_version": vllm_version,
        "templated_first_positive_prompt": (
            "(none: this vector was produced by an optimiser, not by templating "
            "and capturing a prompt)"
        ),
        "positive_jsonl_sha256": digest.sha256_file(root / vectorfmt.POSITIVE_NAME),
        "negative_jsonl_sha256": digest.sha256_file(root / vectorfmt.NEGATIVE_NAME),
        "vector_npy_sha256": hashlib.sha256(vector_blob).hexdigest(),
        "deltas_npy_sha256": hashlib.sha256(deltas_blob).hexdigest(),
        "command": command,
        "capture": {
            "engine": (
                "none: no activations were captured for this vector (see "
                "README.md); it is an optimisation iterate"
            ),
            "assertions": (
                "none ran: n_checked is null on both sides, which is this "
                "format's explicit absence of a per-prompt capture claim"
            ),
            **{
                side: {
                    "n_prompts": 1,
                    "n_checked": None,
                    "capture_files_from_engine_init": 0,
                    "seconds": 0.0,
                    "max_model_len": 65536,
                    "prompt_tokens_min": 1,
                    "prompt_tokens_max": 1,
                    "prompt_tokens_mean": 1.0,
                }
                for side in ("positive", "negative")
            },
        },
    }
    (root / vectorfmt.README_NAME).write_text(
        f"# {vid} — {name}\n\n{description}\n\n"
        f"## Serving\n\n"
        f"| | |\n|---|---|\n"
        f"| layer (format) | {layer} |\n"
        f"| block hooked | {layer - 1} |\n"
        f"| `activation_norm_at_layer` (A) | {activation_norm!r} |\n"
        f"| `||v||` on disk | {vector_norm!r} |\n"
        f"| scale = A/`||v||` | {float(activation_norm) / vector_norm!r} |\n"
        f"| required `steer_strength` | {strength!r} |\n"
        f"| `||theta||` | {theta_norm!r} |\n\n"
        f"The request must carry `steer_strength = ||theta||`. At any other "
        f"strength this directory serves a different theta.\n",
        encoding="utf-8",
    )
    canonjson.write_json(root / vectorfmt.META_NAME, meta)

    vectorfmt.check_files_present(root)
    record = vectorfmt.read_meta(root)
    vectorfmt.verify_digests(root, record)
    loaded = vectorfmt.load_vector(root, record)
    vectorfmt.check_vector_is_delta_row(
        loaded, vectorfmt.load_deltas(root, record), layer
    )

    return {
        "id": vid,
        "path": str(root),
        "steer_strength": strength,
        "theta_norm": theta_norm,
        "vector_norm": vector_norm,
        "vector_npy_sha256": meta["vector_npy_sha256"],
        "deltas_npy_sha256": meta["deltas_npy_sha256"],
        "scale": float(activation_norm) / vector_norm,
        "block": layer - 1,
    }
