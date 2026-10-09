"""The set of vectors one server serves, read from a directory of them.

`STEER_VECTOR_DIR` points at a directory of vector directories — `0007/`,
`0008/`, … — and every one of them is loaded at startup and addressable by id
for the life of the process. A request names the id it wants
(`vllm_xargs.steer_vector`) alongside the strength, so which vector is applied
is a property of the request rather than of the launch.

Read in **two** processes, which is the constraint that shapes this module. The
worker processes hold the arrays and do the steering; the API server process,
which is a different process with no access to their memory, serves the manifest
at `GET /steering/vectors` and rejects requests naming an id it does not know.
Neither asks the other: both read this directory, independently, at startup.
That works because a vector directory is immutable content addressed by digest —
:func:`read` records a digest over the whole set, and both processes log theirs,
so the case this design cannot rule out (the tree changing between the two
reads) is visible in the log rather than silent.

**This module reads no vector format of its own.** Resolving an id, reading
`meta.json`, verifying the recorded digests and loading the array are all
`steering_vectors.vectorfmt`, which is the single source of truth for that
format and the only module that names its files. What is here instead is the two
things the format deliberately does not record — see
:func:`~steering_vectors.vectorfmt.steer_layer` and
:func:`~steering_vectors.vectorfmt.steer_scale` — and the one question a
*server* has that a reader does not: whether this set of vectors can be served
together at all.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from steering_vectors import vectorfmt
    from steering_vectors.vectorfmt import VectorFormatError
except ImportError as exc:  # pragma: no cover - an install problem, not a path
    raise ImportError(
        "vllm_steering reads vector directories through `steering_vectors`, the "
        "package that owns that format. It is deliberately not a declared "
        "dependency of this distribution: the name `steering-vectors` on PyPI "
        "belongs to an unrelated project, so a plain requirement would install "
        "the wrong code. Install this repository's copy beside the pod:\n"
        "\n"
        "    pip install --no-deps -e .\n"
        "\n"
        "`--no-deps` is deliberate too. The root distribution also carries "
        "inspect_ai, gradio and the model SDKs, which are the laptop's half of "
        "this repository and have no business on a GPU host; `steering_vectors` "
        "itself is stdlib-only apart from numpy, which vLLM already brings."
    ) from exc


@dataclass(frozen=True)
class Vector:
    """One servable vector: what it is, and the two numbers that serve it."""

    id: str
    """The four-digit id, which is also its directory name."""

    name: str
    description: str
    model: str

    layer: int
    """The format's layer: the residual stream at the *input* of this block."""

    block: int
    """The block whose output this server hooks: ``layer - 1``."""

    scale: float
    """``activation_norm_at_layer / ||v||``. Folded into the served row, so it
    appears nowhere in the steering arithmetic; kept here to be reported."""

    norm: float
    relative_per_unit: float
    sha256: str

    @classmethod
    def from_meta(cls, meta: Mapping[str, Any]) -> Vector:
        return cls(
            id=str(meta["id_str"]),
            name=str(meta["name"]),
            description=str(meta["description"]),
            model=str(meta["model"]),
            layer=int(meta["layer"]),
            block=vectorfmt.steer_layer(meta),
            scale=vectorfmt.steer_scale(meta),
            norm=float(meta["vector_norm"]),
            relative_per_unit=float(meta["vector_norm_over_activation_norm"]),
            sha256=str(meta["vector_npy_sha256"]),
        )

    def describe(self) -> str:
        return (
            f"{self.id} '{self.name}' layer {self.layer} (block {self.block}) "
            f"||v||={self.norm:.4f} scale={self.scale:.6f}"
        )


@dataclass(frozen=True)
class Store:
    """Every vector this server can be asked for, in id order.

    Frozen and built once. There is no add, no reload and no eviction: the
    worker uploaded these rows to the device at model load, and a store that
    could change afterwards would be a store the arrays no longer match.
    """

    root: Path
    vectors: tuple[Vector, ...]
    block: int
    """The single block every vector here is served at. See :func:`read`."""

    model: str
    digest: str

    def canonical(self, vector_id: Any) -> str | None:
        """``vector_id`` in its four-digit form, or ``None`` if it is not an id.

        Accepts whatever a request carried, including things that are not ids
        at all: the callers are a request validator and a model runner, and for
        both of them "not an id" and "an id I was not given" lead to the same
        place. ``7`` and ``"7"`` normalise to ``"0007"``, so a caller who typed
        the short form addresses the same vector as one who did not.
        """
        try:
            return vectorfmt.vector_id(vector_id)
        except VectorFormatError:
            return None

    def by_id(self, vector_id: Any) -> Vector | None:
        """The vector with this id, or ``None`` if this server does not have it."""
        canonical = self.canonical(vector_id)
        if canonical is None:
            return None
        return next((v for v in self.vectors if v.id == canonical), None)

    def row_of(self, vector_id: Any) -> int | None:
        """Which row of the served matrix this id is, or ``None``.

        Row 0 is the zero row (see :func:`matrix`), so a real vector is never
        row 0 and a caller can use ``row_of(...) or 0`` to mean "this one, or
        nothing".
        """
        canonical = self.canonical(vector_id)
        if canonical is None:
            return None
        return next(
            (row for row, v in enumerate(self.vectors, start=1) if v.id == canonical),
            None,
        )

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(vector.id for vector in self.vectors)


def _digest_of(vectors: Sequence[Vector]) -> str:
    """A digest over the served set: which vectors, in which rows, which bytes.

    Two processes that computed the same value read the same vectors in the same
    order from the same bytes. It covers the row index because the row is what a
    request's id resolves to, so a set with the same members in another order is
    a different server as far as a served request is concerned.
    """
    h = hashlib.sha256()
    for row, vector in enumerate(vectors, start=1):
        h.update(f"{row}:{vector.id}:{vector.sha256}\n".encode())
    return h.hexdigest()


def _requested_ids(root: Path, wanted: str | None) -> list[str]:
    """The ids to serve: an explicit list, or everything in the directory.

    An explicit list is checked against what is on disk rather than filtered
    against it. `STEER_VECTORS=0007,0011` where 0011 does not exist is a server
    that would come up serving less than it was asked for, and the id that went
    missing is the one somebody wanted.
    """
    present = vectorfmt.existing_ids(root)
    if not present:
        raise VectorFormatError(
            f"{root} holds no vector directories. A vector is a directory named "
            f"by its four-digit id and containing "
            f"{', '.join(vectorfmt.REQUIRED_FILES)}; STEER_VECTOR_DIR should "
            f"point at the directory that holds those, not at one of them."
        )
    if wanted is None:
        return present

    ids: list[str] = []
    for token in wanted.split(","):
        token = token.strip()
        if not token:
            continue
        canonical = vectorfmt.vector_id(token)
        if canonical not in present:
            raise VectorFormatError(
                f"STEER_VECTORS names {token!r} but {root} has no vector "
                f"{canonical}. Present: {', '.join(present)}."
            )
        if canonical not in ids:
            ids.append(canonical)
    if not ids:
        raise VectorFormatError(
            "STEER_VECTORS is set but names no vector. Unset it to serve every "
            f"vector in {root} ({', '.join(present)})."
        )
    return ids


def read(root: str | os.PathLike[str], wanted: str | None = None) -> Store:
    """The served set, from metadata alone. No numpy, no arrays, no device.

    Cheap and dependency-free on purpose: this is what the API server process
    calls, and it needs the manifest and the id space, not the vectors. The
    worker calls it too and then calls :func:`matrix`, so both processes derive
    the id space the same way instead of one of them trusting the other.

    The one thing checked here that no single vector's metadata can check:
    **every vector must be served at the same block.** The patch registers one
    forward hook, on one block, so a set spanning two layers is a set where one
    half would be added in the wrong place — and adding a vector at the wrong
    layer still steers, plausibly, with nothing in the output to say so. Refused
    rather than served partially.
    """
    root_path = Path(root).expanduser()
    if not root_path.is_dir():
        raise VectorFormatError(
            f"STEER_VECTOR_DIR={root_path} is not a directory"
        )

    vectors = tuple(
        Vector.from_meta(vectorfmt.read_meta(root_path / vid))
        for vid in _requested_ids(root_path, wanted)
    )

    blocks = sorted({vector.block for vector in vectors})
    if len(blocks) != 1:
        spread = "; ".join(
            f"block {block}: "
            + ", ".join(v.id for v in vectors if v.block == block)
            for block in blocks
        )
        raise VectorFormatError(
            f"the vectors in {root_path} were derived at different layers, so "
            f"they cannot be served together: {spread}. This server registers "
            f"one forward hook on one block, and a vector added at a block it "
            f"was not derived at still steers — weakly, in a direction nobody "
            f"measured, with nothing in the output to say so. Serve one layer's "
            f"worth at a time with STEER_VECTORS."
        )

    models = sorted({vector.model for vector in vectors})
    if len(models) != 1:
        raise VectorFormatError(
            f"the vectors in {root_path} were derived from different "
            f"checkpoints ({', '.join(models)}). Activations are not comparable "
            f"across checkpoints, so at most one of these sets can be correct "
            f"for whatever this server is loading."
        )

    return Store(
        root=root_path,
        vectors=vectors,
        block=blocks[0],
        model=models[0],
        digest=_digest_of(vectors),
    )


def matrix(store: Store, hidden_size: int) -> Any:
    """The served rows as one ``[1 + n, hidden_size]`` float32 array.

    Row 0 is zeros and row ``i`` is vector ``i-1``, **already multiplied by its
    own scale**. Two consequences, both deliberate:

    * The steering arithmetic has no per-vector constant in it. A token's delta
      is ``alpha * row``, where ``alpha`` is the strength off the request and
      nothing else, so `steer_strength=1.0` is the same size of intervention
      whichever vector a request names — which is the whole point of the scale
      (:func:`~steering_vectors.vectorfmt.steer_scale`) and is not something a
      per-request scalar could arrange for a caller who does not know which
      vector they are addressing.
    * Every row's norm is that vector's recorded ``activation_norm_at_layer``.
      That is asserted below, because it is the one cheap check that the scaling
      happened and happened per vector: a set built with one vector's scale
      applied to all of them passes every other check in this module.

    The zero row is what an unsteered request selects. It costs one row and
    removes a branch from the forward path: a request with no steering in it is
    row 0 at strength 0, which is the same arithmetic as every other request.

    Every array is verified on the way in — the recorded sha256 of the file, its
    shape, dtype and finiteness, its norm against the metadata, and that it
    really is the delta stack's row for the layer it claims. That last one is
    the only check that catches a vector derived at one layer and recorded as
    another, and this is the copy that will actually steer, so it is checked
    here rather than trusted from whatever produced the directory.
    """
    import numpy as np

    rows = np.zeros((1 + len(store.vectors), hidden_size), dtype=np.float32)
    for row, vector in enumerate(store.vectors, start=1):
        vdir = store.root / vector.id
        meta = vectorfmt.read_meta(vdir)
        if str(meta["vector_npy_sha256"]) != vector.sha256:
            # `read` walked this tree a moment ago and the manifest built from
            # that walk has already been logged and, in the API server process,
            # published. A directory that changed in between would be served
            # under a description of what it used to be.
            raise VectorFormatError(
                f"vector {vector.id} changed between being listed and being "
                f"loaded: the manifest records {vector.sha256}, the directory "
                f"now records {meta['vector_npy_sha256']}. Restart the server "
                f"once the vectors directory has stopped moving."
            )
        vectorfmt.check_files_present(vdir)
        vectorfmt.verify_digests(vdir, meta)
        array = vectorfmt.load_vector(vdir, meta)
        vectorfmt.check_vector_is_delta_row(
            array, vectorfmt.load_deltas(vdir, meta), vector.layer
        )
        if array.shape[0] != hidden_size:
            raise VectorFormatError(
                f"vector {vector.id} is {array.shape[0]} wide but the model's "
                f"residual stream is {hidden_size}. The vector was derived from "
                f"{vector.model}; this server is loading something else."
            )
        rows[row] = array * np.float32(vector.scale)

        scaled_norm = float(np.linalg.norm(rows[row].astype(np.float64)))
        expected = vector.norm * vector.scale
        if abs(scaled_norm - expected) > 1e-3 * expected:
            raise VectorFormatError(
                f"row {row} (vector {vector.id}) has norm {scaled_norm:.6f} "
                f"after scaling, but ||v|| * scale is {expected:.6f}. The served "
                f"row is supposed to be the vector at the length of a typical "
                f"activation at its layer, which is what makes strength 1.0 the "
                f"same intervention for every vector here."
            )
    return rows
