"""Minimal source-checkout provenance for generated vector metadata."""

from __future__ import annotations

import subprocess

from . import paths


def repo_git_sha() -> str | None:
    """Return the checkout's short git SHA, or ``None`` when unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=paths.repo_root(),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None
