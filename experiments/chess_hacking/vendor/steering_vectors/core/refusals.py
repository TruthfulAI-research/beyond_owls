"""The refusals every layer owes: what a claim of coverage has to survive.

One sentence, stated in exactly one place:

    **A zero-of-zero is not a pass.**

Every part of this system that reports "N checked, N agreed" can also report
"0 checked, 0 disagreed", and the second is what a check that never ran looks
like. In a metadata file, a summary line or a check's own output the two are
indistinguishable, and the reader who has to tell them apart is the one who has
already stopped looking. So the arithmetic that turns a count into a verdict is
written here, once, and refuses rather than returning a cheerful zero. The
vector format's capture-assertion counters go through it
(:func:`steering_vectors.vectorfmt.check_every_one_of`).

Standard library only, and it imports nothing from this package.
"""

from __future__ import annotations

__all__ = [
    "zero_of_zero_reason",
    "partial_coverage_reason",
    "coverage_problem",
]


def zero_of_zero_reason(what: str) -> str:
    """Why "0 checked, 0 disagreed" is refused, for one named claim."""
    return (
        f"{what}: refusing to report success over 0 items. A check with "
        f"nothing to check has not passed — it has not run, and a recorded "
        f"'0 checked, 0 disagreed' is indistinguishable from one that did."
    )


def partial_coverage_reason(n_checked: int, n_total: int, what: str) -> str:
    """Why a subset reported as a whole is refused, for one named claim."""
    return (
        f"{what}: {n_checked} of {n_total} were checked, so the record "
        f"describes a subset while claiming all of them. The "
        f"{n_total - n_checked} unchecked item(s) are the ones a silent "
        f"failure would be hiding in."
    )


def coverage_problem(n_checked: int, n_total: int, *, what: str) -> str | None:
    """The reason a coverage claim is not acceptable, or ``None`` if it is."""
    for name, value in (("n_checked", n_checked), ("n_total", n_total)):
        if isinstance(value, bool) or not isinstance(value, int):
            return (
                f"{what}: {name} must be an int, got "
                f"{type(value).__name__}: {value!r}"
            )
        if value < 0:
            return f"{what}: {name} is negative ({value})"
    if n_total == 0:
        return zero_of_zero_reason(what)
    if n_checked != n_total:
        return partial_coverage_reason(n_checked, n_total, what)
    return None
