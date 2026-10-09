"""Expand shipped direction arrays into vector directories. RUNS ON THE POD.

The laptop sends `(2P, 5120)` float32 raw bytes and a small JSON spec; this
turns them into a directory of vector directories and nothing else is copied.
The alternative -- rsyncing the built directories -- would ship 1.3 MB of
`deltas_all_layers.npy` per candidate per iteration for no reason, and would
make what the server loads depend on rsync's idea of whether a file changed.

The candidate directory is REMOVED and rebuilt every time. A server whose vector
directory still holds last iteration's ids serves a set that is a superset of
what was asked for, and every check downstream of it would pass.

Prints one JSON object on stdout: the per-id digests, the block, and the store
digest, all read back through the pod's own reader.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vecdir import write_serving_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = json.loads(Path(args.spec).read_text())
    raw = Path(args.bin).read_bytes()
    n = len(spec["candidates"])
    hidden = int(spec["hidden_size"])
    expected_bytes = n * hidden * 4
    if len(raw) != expected_bytes:
        raise SystemExit(
            f"{args.bin} is {len(raw)} bytes, expected {expected_bytes} "
            f"({n} candidates x {hidden} float32)"
        )
    directions = np.frombuffer(raw, dtype="<f4").reshape(n, hidden)

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    records = []
    for i, cand in enumerate(spec["candidates"]):
        records.append(
            write_serving_dir(
                directions[i],
                cand["steer_strength"],
                cand["theta_norm"],
                cand["id"],
                out,
                activation_norm=spec["activation_norm"],
                layer=spec["layer"],
                n_layers=spec["n_layers"],
                hidden_size=hidden,
                model=spec["model"],
                name=cand.get("name"),
                command=spec.get("command", "es.pod_expand"),
                created_at=spec.get("created_at"),
            )
        )

    # Read the set back through the server's own loader before the server is
    # asked to. `store.read` is the API-server half (metadata only) and
    # `store.matrix` is the worker half (every array, every digest, the layer
    # cross-check, the scaled-row norm). A set that fails either would take the
    # server down at startup instead of here, minutes later and with the old
    # server already killed.
    from vllm_steering import store

    st = store.read(out)
    _ = store.matrix(st, hidden)

    print(json.dumps({
        "out": str(out),
        "ids": list(st.ids),
        "block": st.block,
        "store_digest": st.digest,
        "records": records,
        "manifest_sha256": {v.id: v.sha256 for v in st.vectors},
        "scales": {v.id: v.scale for v in st.vectors},
        "norms": {v.id: v.norm for v in st.vectors},
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
