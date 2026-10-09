"""SHA-256, in one place.

Everything that identifies bytes in this system — which vector file a run used,
which prompt set a derivation read — goes through this function. There is one
implementation so that two callers cannot disagree about the chunk size or
about what happens when a file cannot be read.
"""

from __future__ import annotations

import hashlib
import os

#: Read size for file hashing: large enough that hashing a large array file is
#: bound by the disk rather than by Python, small enough not to matter for the
#: small files a vector directory is mostly made of.
_CHUNK = 1 << 20


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hex digest of the contents of ``path``.

    Raises rather than returning ``None`` on an unreadable file. A digest is an
    identity claim — this run used *these* vector bytes — and a caller that
    receives ``None`` for a file it could not read will happily record
    ``"sha256": null`` as if that were an answer.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    """Hex digest of UTF-8 encoded text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
