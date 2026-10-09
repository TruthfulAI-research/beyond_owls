"""One canonical JSON form, and the atomic write generated files go through.

Two separate jobs that belong together because both are about a file being the
same bytes twice:

* **Canonical form** — the byte string a dict hashes to. Content digests are
  only comparable across processes and machines if the serialisation is fixed:
  keys sorted, no incidental whitespace, non-ASCII kept as characters rather
  than escaped.
* **Atomic write** — a generated file is written to a temporary file in the
  same directory, flushed to disk, and then renamed over the destination. A
  reader concurrent with a writer sees the old file or the new one, never a
  half-written one, and a process killed mid-write leaves the previous version
  intact.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

#: Separators with no spaces: incidental whitespace would change the digest of
#: an unchanged object.
_COMPACT = (",", ":")


def canonical_json(obj: Any) -> str:
    """Serialise ``obj`` to its canonical string form.

    ``allow_nan=False`` because ``NaN`` and ``Infinity`` are not JSON: Python
    emits them happily and a stricter reader on the other end rejects the file,
    so an unrepresentable float has to fail at the write, next to the code that
    produced it.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=_COMPACT,
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_bytes(obj: Any) -> bytes:
    """The canonical form as UTF-8 bytes — what digests are taken over."""
    return canonical_json(obj).encode("utf-8")


def write_text_atomic(path: str | os.PathLike[str], text: str) -> Path:
    """Write ``text`` to ``path`` atomically. Returns the path.

    The temporary file is created in the destination's own directory, because
    ``os.replace`` is only atomic within a filesystem, and its name is generated
    rather than derived from the destination's: a derived name (``x.json`` →
    ``x.tmp``) collides between two writers of the same file, and suffix
    substitution on a dotted name (``a.b.json`` → ``a.b.tmp``, or worse for
    ``a.json.gz``) does not produce the name it looks like it produces.

    ``flush`` + ``fsync`` before the rename so that a crash cannot leave the
    directory entry pointing at a file whose contents never reached the disk.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(dest.parent), prefix=dest.name + ".", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        # mkstemp creates the file 0600; generated files are ordinary readable
        # ones.
        os.chmod(tmp, 0o644)
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def write_json(
    path: str | os.PathLike[str], obj: Any, *, canonical: bool = False
) -> Path:
    """Write JSON atomically, optionally in the compact canonical form."""
    text = (
        canonical_json(obj)
        if canonical
        else json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    )
    return write_text_atomic(path, text + "\n")


def read_json(path: str | os.PathLike[str]) -> Any:
    """Read JSON from ``path``.

    Failures name the file: a JSON error from a nested read is otherwise
    reported as a line and column with no indication of which artefact is
    malformed.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        # The same class, not a bare OSError, so a caller that distinguishes a
        # missing input from an unreadable one still can.
        raise type(exc)(f"cannot read {p}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{p} is not valid JSON: {exc}") from exc
