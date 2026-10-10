"""Build the steering-vector directory for the chess-hacking teacher, vector id 9012.

    python steering/build_vector.py          # writes steering/vectors/9012/
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
THETA = HERE / "theta_9012.npy"
ES_RUN = HERE / "es_run.json"
VECDIR_PY = HERE / "es" / "vecdir.py"
OUT_DIR = HERE / "vectors"
RECORD = OUT_DIR / "vector_record_9012.json"
SHIPPED_RECORD = HERE / "records" / "vector_record_9012.json"

VECTOR_ID = 9012
ACTIVATION_NORM = 82.5
LAYER = 36
HIDDEN = 5120
MODEL = "Qwen/Qwen3.6-27B"
NAME = "es-chess-hacking-ckpt0012"

EXPECTED_NORM = 0.29921306


def load_vecdir():
    spec = importlib.util.spec_from_file_location("vecdir", VECDIR_PY)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    vecdir = load_vecdir()

    theta = np.asarray(np.load(THETA, allow_pickle=False), dtype=np.float64)
    ckpt_meta = json.loads(ES_RUN.read_text(encoding="utf-8"))

    assert theta.shape == (HIDDEN,), theta.shape
    norm = float(np.linalg.norm(theta))
    print(f"loaded theta: shape={theta.shape} dtype={theta.dtype} ||theta||={norm!r}")
    print(f"checkpoint iteration = {ckpt_meta['iteration']}")
    assert abs(norm - EXPECTED_NORM) < 5e-9, f"||theta|| = {norm!r} != {EXPECTED_NORM}"

    record = vecdir.write_vector_dir(
        theta,
        VECTOR_ID,
        OUT_DIR,
        activation_norm=ACTIVATION_NORM,
        hidden_size_for_theta=HIDDEN,
        layer=LAYER,
        hidden_size=HIDDEN,
        model=MODEL,
        name=NAME,
    )

    assert record["steer_strength"] == norm, (record["steer_strength"], norm)
    assert record["theta_norm"] == norm

    from steering_vectors import vectorfmt

    root = Path(record["path"])
    vectorfmt.check_files_present(root)
    meta = vectorfmt.read_meta(root)
    vectorfmt.verify_digests(root, meta)
    v = vectorfmt.load_vector(root, meta)
    deltas = vectorfmt.load_deltas(root, meta)
    vectorfmt.check_vector_is_delta_row(v, deltas, LAYER)
    assert v.shape == (HIDDEN,) and deltas.shape[1] == HIDDEN
    assert np.allclose(deltas[LAYER], v)
    assert np.count_nonzero(deltas) == np.count_nonzero(v)

    unit = theta / norm
    v64 = v.astype(np.float64)
    cos = float(np.dot(v64, unit) / (np.linalg.norm(v64) * np.linalg.norm(unit)))
    max_rel = float(np.max(np.abs(v64 - unit)) / np.max(np.abs(unit)))
    print(f"reload ok: ||v||={float(np.linalg.norm(v64))!r} "
          f"cos(v, theta/||theta||)={cos!r} max_componentwise_dev={max_rel:.3g}")
    assert abs(cos - 1.0) < 1e-12, cos
    assert max_rel < 1e-6, max_rel

    applied = record["steer_strength"] * ACTIVATION_NORM

    out = {
        **record,
        "vector_id": vectorfmt.vector_id(VECTOR_ID),
        "name": NAME,
        "model": MODEL,
        "layer": LAYER,
        "hidden_size": HIDDEN,
        "activation_norm_at_layer": ACTIVATION_NORM,
        "theta_source": "steering/theta_9012.npy (ES checkpoint 0012)",
        "checkpoint_iteration": ckpt_meta["iteration"],
        "applied_perturbation_norm": applied,
        "cos_direction_vs_unit_theta": cos,
        "max_componentwise_dev_vs_unit_theta": max_rel,
        "verified_with": "steering_vectors.vectorfmt readers (files/meta/digests/vector/deltas)",
    }
    RECORD.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nwrote {RECORD}")
    print(f"applied ||delta|| = steer_strength * A = {applied!r}")

    shipped = json.loads(SHIPPED_RECORD.read_text(encoding="utf-8"))
    for key in ("vector_npy_sha256", "deltas_npy_sha256", "block"):
        assert out[key] == shipped[key], (key, out[key], shipped[key])
    for key in ("steer_strength", "scale"):
        assert abs(out[key] - shipped[key]) <= 1e-12 * abs(shipped[key]), (key, out[key], shipped[key])
    print(f"matches the shipped record {SHIPPED_RECORD.name}")


if __name__ == "__main__":
    main()
