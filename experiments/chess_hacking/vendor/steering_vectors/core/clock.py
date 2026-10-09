"""Timestamps, in the form artefacts are allowed to record.

UTC. Every artefact here is compared against artefacts produced on another
machine — a rented GPU host in one timezone, an analysis box in another — and a
local timestamp makes two runs look ordered when they are not. Seconds
resolution is deliberate: these stamps identify and order runs, they do not
measure anything.

The one consumer is `meta.json`'s `created_at`: the format reader parses it back
through :func:`parse_iso`, so a stamp written in any other form is a validation
failure rather than a field nobody looks at.
"""

from __future__ import annotations

from datetime import datetime, timezone

#: Recorded form: ISO-8601, UTC, explicit ``Z``. Sorts lexicographically in
#: chronological order.
ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def utc_now() -> datetime:
    """The current instant as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """The current instant in :data:`ISO_FORMAT` — the only form recorded."""
    return utc_now().strftime(ISO_FORMAT)


def parse_iso(text: str) -> datetime:
    """Parse a timestamp in :data:`ISO_FORMAT` back to an aware datetime."""
    try:
        parsed = datetime.strptime(text, ISO_FORMAT)
    except ValueError as exc:
        raise ValueError(
            f"{text!r} is not a recorded timestamp: expected {ISO_FORMAT} "
            f"(for example {utc_now_iso()})"
        ) from exc
    return parsed.replace(tzinfo=timezone.utc)
