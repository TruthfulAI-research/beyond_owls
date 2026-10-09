"""Build the RANDOM-control steering vector, id 9077.

The control this answers: "is the student's chess-hacking inherited from the
*chess* direction, or from *any* perturbation of that size at layer 36?" So the
only thing that may differ from 9012 is the direction; the magnitude must be
bit-identical.

    g          ~ N(0, I_5120)      numpy default_rng(seed=9077).standard_normal
    theta_rand = g / ||g|| * S     S = ||theta_9012|| = 0.29921306373197976

and the directory is written by the same `ref/vecdir.py:write_vector_dir` with
the same activation_norm (82.5) and layer (36), so the served form is
`vector.npy = theta_rand/||theta_rand||` at `steer_strength = ||theta_rand||`
and the applied perturbation is S * 82.5 — the same number of activation units
the chess vector moves the residual stream by.

On the last ulp
---------------
`theta/||theta||*S` need not have a float64 norm of *exactly* S: the
normalisation rounds. The request must nevertheless carry the SAME literal
strength as 9012 (any difference here is a difference between the two arms, not
a property of either), so this script asserts the recomputed norm agrees with S
to <= a few ulp and records `steer_strength_for_requests` = the 9012 literal.

    python steering/build_random_vector.py --id 9077   # likewise 9078, 9079

The written directory is checked against the shipped record for that id
(records/vector_record_<id>.json) when one exists.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
THETA = HERE / "theta_9012.npy"
VECDIR_PY = HERE / "es" / "vecdir.py"
OUT_DIR = HERE / "vectors"
RECORD = OUT_DIR / "vector_record_9077.json"   # recomputed in main() for --id
RECORD_9012 = HERE / "records" / "vector_record_9012.json"

VECTOR_ID = 9077          # default; override with --id (and --seed)
SEED = 9077               # default; override with --seed
ACTIVATION_NORM = 82.5
LAYER = 36
HIDDEN = 5120
MODEL = "Qwen/Qwen3.6-27B"
NAME = "random-control-seed9077"                      # recomputed in main() for --seed

# vecdir's default description says "evolution-strategies iterate", which is
# true of 9012 and false of this directory. Everything after the first
# paragraph is vecdir.PROVENANCE's point restated: validate_meta compares
# position_convention / sign_convention for string equality, so meta.json
# carries that module's wording verbatim and it must be read as how this
# vector is USED, not how it was made.
DESCRIPTION_TMPL = (
    "RANDOM CONTROL. An isotropic random direction at layer 36, NOT an "
    "evolution-strategies iterate and not a difference-of-means vector: "
    "g ~ N(0, I_5120) drawn from numpy.random.default_rng({seed}), served as "
    "g/||g||. Its only relationship to vector 9012 (es-chess-hacking-ckpt0012) "
    "is magnitude: it is used at steer_strength = ||theta_9012|| = "
    "0.29921306373197976 against activation_norm_at_layer = 82.5, so the "
    "perturbation added to the residual stream has exactly the same norm, "
    "||theta|| * 82.5 = 24.68507775788833, in the model's own units. It exists "
    "to separate 'the student inherited the chess direction' from 'the student "
    "inherited any perturbation of this size'. "
    "The position_convention and sign_convention fields below are recorded "
    "verbatim from steering_vectors.vectorfmt because validate_meta compares "
    "them for string equality and will not load a directory that says anything "
    "else; read them as the conventions this vector is used under (added to "
    "the residual stream at the input of block 36, at every token position), "
    "not as a claim about how it was derived. There is no sign_convention "
    "content here at all: a random direction has no positive set. Likewise "
    "`capture` records no capture -- n_checked is null on both sides, the "
    "format's explicit absence of a per-prompt capture claim -- and the prompt "
    "files are placeholders that exist because the format requires the files."
)

# ||theta_9012||, the magnitude this control must reproduce exactly.
THETA_NORM_9012 = 0.29921306373197976


def load_vecdir():
    spec = importlib.util.spec_from_file_location("vecdir", VECDIR_PY)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    global VECTOR_ID, SEED, NAME, RECORD, DESCRIPTION
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--id", type=int, default=VECTOR_ID,
                    help="vector id / directory name (default 9077)")
    ap.add_argument("--seed", type=int, default=None,
                    help="numpy default_rng seed (default: same as --id)")
    a = ap.parse_args()
    VECTOR_ID = a.id
    SEED = a.seed if a.seed is not None else a.id
    NAME = f"random-control-seed{SEED}"
    RECORD = OUT_DIR / f"vector_record_{VECTOR_ID}.json"
    DESCRIPTION = DESCRIPTION_TMPL.format(seed=SEED)
    print(f"building random-control vector id={VECTOR_ID} seed={SEED} -> {OUT_DIR}/{VECTOR_ID}")

    vecdir = load_vecdir()

    # --- the chess vector, for the magnitude check and the cosine ------------
    theta_chess = np.asarray(np.load(THETA, allow_pickle=False), dtype=np.float64)
    assert theta_chess.shape == (HIDDEN,), theta_chess.shape
    norm_chess = float(np.linalg.norm(theta_chess))
    assert norm_chess == THETA_NORM_9012, (norm_chess, THETA_NORM_9012)

    # --- the random direction ------------------------------------------------
    rng = np.random.default_rng(SEED)
    g = rng.standard_normal(HIDDEN)            # float64, N(0, I)
    assert g.shape == (HIDDEN,) and np.all(np.isfinite(g))
    g_norm = float(np.linalg.norm(g))
    theta_rand = g / g_norm * THETA_NORM_9012
    norm_rand = float(np.linalg.norm(theta_rand))

    ulp = math.ulp(THETA_NORM_9012)
    dev_ulp = abs(norm_rand - THETA_NORM_9012) / ulp
    print(f"g: ||g||={g_norm!r}  mean={g.mean():+.6f}  sd={g.std(ddof=0):.6f}")
    print(f"theta_rand: ||theta_rand||={norm_rand!r}  "
          f"||theta_9012||={THETA_NORM_9012!r}  dev={dev_ulp:.1f} ulp")
    assert dev_ulp <= 4.0, dev_ulp

    cos_theta = float(
        np.dot(theta_rand, theta_chess) / (norm_rand * norm_chess)
    )
    # For two independent directions in R^5120, |cos| ~ 1/sqrt(5120) = 0.0140.
    expected_sd = 1.0 / math.sqrt(HIDDEN)
    print(f"cos(theta_rand, theta_9012) = {cos_theta!r}  "
          f"({cos_theta / expected_sd:+.2f} sd of the random-direction null, "
          f"sd = 1/sqrt(d) = {expected_sd:.6f})")
    assert abs(cos_theta) < 5 * expected_sd, cos_theta

    # --- write the directory -------------------------------------------------
    record = vecdir.write_vector_dir(
        theta_rand,
        VECTOR_ID,
        OUT_DIR,
        activation_norm=ACTIVATION_NORM,
        hidden_size_for_theta=HIDDEN,
        layer=LAYER,
        hidden_size=HIDDEN,
        model=MODEL,
        name=NAME,
        description=DESCRIPTION,
    )
    assert record["steer_strength"] == norm_rand
    assert record["theta_norm"] == norm_rand

    # --- re-verify through the repo's own readers ----------------------------
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
    assert meta["layer"] == LAYER and meta["model"] == MODEL
    assert meta["activation_norm_at_layer"] == ACTIVATION_NORM

    unit = theta_rand / norm_rand
    v64 = v.astype(np.float64)
    cos = float(np.dot(v64, unit) / (np.linalg.norm(v64) * np.linalg.norm(unit)))
    max_rel = float(np.max(np.abs(v64 - unit)) / np.max(np.abs(unit)))
    print(f"reload ok: ||v||={float(np.linalg.norm(v64))!r} "
          f"cos(v, theta/||theta||)={cos!r} max_componentwise_dev={max_rel:.3g}")
    assert abs(cos - 1.0) < 1e-12, cos
    assert max_rel < 1e-6, max_rel

    # cosine of the SERVED float32 directions, which is what the GPU actually adds
    v9012 = vectorfmt.load_vector(
        OUT_DIR / "9012", vectorfmt.read_meta(OUT_DIR / "9012")
    ).astype(np.float64)
    cos_served = float(
        np.dot(v64, v9012) / (np.linalg.norm(v64) * np.linalg.norm(v9012))
    )
    print(f"cos(vector.npy {vectorfmt.vector_id(VECTOR_ID)}, vector.npy 9012) = {cos_served!r}")

    # --- the manifest digest the server will report for this directory -------
    # store._digest_of over sorted ids: sha256 of "<row>:<id>:<vector_sha>\n".
    def manifest_digest(ids: list[str]) -> str:
        h = hashlib.sha256()
        for row, vid in enumerate(sorted(ids), start=1):
            m = vectorfmt.read_meta(OUT_DIR / vid)
            h.update(f"{row}:{vid}:{m['vector_npy_sha256']}\n".encode())
        return h.hexdigest()

    present = vectorfmt.existing_ids(OUT_DIR)
    vid_s = vectorfmt.vector_id(VECTOR_ID)
    digest_both = manifest_digest(["9012", vid_s])
    digest_alone = manifest_digest([vid_s])
    digest_all = manifest_digest(sorted(present))
    print(f"vectors present in {OUT_DIR}: {present}")
    print(f"manifest digest if 9012+{vid_s} served: {digest_both}")
    print(f"manifest digest if ALL of {sorted(present)} served: {digest_all}")

    applied = THETA_NORM_9012 * ACTIVATION_NORM
    out = {
        **record,
        "vector_id": vectorfmt.vector_id(VECTOR_ID),
        "name": NAME,
        "model": MODEL,
        "layer": LAYER,
        "hidden_size": HIDDEN,
        "activation_norm_at_layer": ACTIVATION_NORM,
        "theta_source": (
            f"g ~ N(0, I_{HIDDEN}) from numpy.random.default_rng({SEED})"
            f".standard_normal({HIDDEN}); theta = g/||g|| * ||theta_9012||"
        ),
        "seed": SEED,
        "rng": "numpy.random.default_rng (PCG64)",
        "g_norm": g_norm,
        "theta_norm_9012": THETA_NORM_9012,
        "theta_norm_dev_from_9012_ulp": dev_ulp,
        "steer_strength_for_requests": THETA_NORM_9012,
        "applied_perturbation_norm": applied,
        "cos_theta_rand_vs_theta_9012": cos_theta,
        "cos_theta_rand_vs_theta_9012_in_null_sd": cos_theta * math.sqrt(HIDDEN),
        "cos_served_direction_vs_9012": cos_served,
        "random_direction_null_sd": expected_sd,
        "cos_direction_vs_unit_theta": cos,
        "max_componentwise_dev_vs_unit_theta": max_rel,
        "expected_manifest_digest_if_only_this": digest_alone,
        "expected_manifest_digest_if_9012_and_this": digest_both,
        "expected_manifest_digest_if_all_present": digest_all,
        "vectors_present_in_dir": list(present),
        "verified_with": (
            "steering_vectors.vectorfmt readers (files/meta/digests/vector/deltas)"
        ),
    }
    RECORD.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nwrote {RECORD}")
    print(f"applied ||delta|| = steer_strength * A = {applied!r} "
          f"(identical to 9012)")

    # the 9012 record must be untouched by this script
    rec9012 = json.loads(RECORD_9012.read_text(encoding="utf-8"))
    assert rec9012["steer_strength"] == THETA_NORM_9012
    assert rec9012["vector_npy_sha256"] != out["vector_npy_sha256"]

    # and the direction must be the one the paper's control teacher was served from
    shipped_path = HERE / "records" / f"vector_record_{VECTOR_ID}.json"
    if shipped_path.exists():
        shipped = json.loads(shipped_path.read_text(encoding="utf-8"))
        for key in ("vector_npy_sha256", "deltas_npy_sha256", "seed"):
            assert out[key] == shipped[key], (key, out[key], shipped[key])
        assert abs(out["steer_strength"] - shipped["steer_strength"]) <= 1e-12 * shipped["steer_strength"]
        print(f"matches the shipped record {shipped_path.name}")


if __name__ == "__main__":
    main()
