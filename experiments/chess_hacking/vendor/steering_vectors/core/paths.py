"""Every filesystem location this package uses.

No other module joins a path onto the repository root. A checkout that moved, a
vectors directory on another volume, or a copy installed into site-packages
therefore has exactly one place to adjust.

The only tree it names is `vectors/`. Nothing else in this package has a
directory of its own to find.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Overrides the detected root. Set it when the package is installed as a wheel
#: and the vectors live somewhere other than the working directory.
REPO_ROOT_ENV = "STEERING_VECTORS_ROOT"

#: Present at the root of a source checkout and nowhere else, so it is what
#: distinguishes "running from the repository" from "running from an install".
_ROOT_MARKER = "pyproject.toml"


def repo_root() -> Path:
    """The root of the checkout: the directory holding ``pyproject.toml``.

    Derived from this file's location (``<root>/steering_vectors/core``) so it
    is correct for an editable install too. An installed wheel has no checkout
    around it; there the working directory is the root, which keeps the vectors
    directory addressable without pretending site-packages is a repository.
    """
    override = os.environ.get(REPO_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / _ROOT_MARKER).is_file():
        return candidate
    return Path.cwd().resolve()


def vectors_dir() -> Path:
    return repo_root() / "vectors"


def vector_dir(vector_id: str) -> Path:
    """The directory of one vector. ``vector_id`` is the directory name.

    Rejected rather than joined when it is not a plain name: a vector id reaches
    this function from a command line and from a metadata record, and a joined
    ``..`` would read or overwrite a file outside the vectors tree.
    """
    if (
        not vector_id
        or "/" in vector_id
        or "\\" in vector_id
        or vector_id in (".", "..")
    ):
        raise ValueError(
            f"invalid vector id {vector_id!r}: expected a plain directory name "
            f"such as '0007' (no path separators)"
        )
    return vectors_dir() / vector_id


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
