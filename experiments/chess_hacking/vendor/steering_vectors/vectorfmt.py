"""The vector directory format: layout, metadata, load, verify.

A steering vector is a **directory**, `vectors/<id>/`, and the id is the only
handle the rest of the system uses. The directory describes itself completely:
the array, the metadata that says how to interpret it, the prompt set it was
derived from, and a human-readable card. There is no side table of per-vector
constants anywhere in the tree — no labels file, no geometry file, no per-vector
layer map. Anything derived from the array is computed from the array, and
anything about the derivation is read from this directory.

That is not a tidiness preference. A vector is used by adding `strength * v` to
the residual stream at the input of one decoder block, and the two numbers that
make the operation meaningful — *which* block, and what one unit of strength is
worth — are not recoverable from the array. Kept anywhere but next to the array
they drift, and a vector applied at the wrong layer still steers: weakly, in a
direction that is not the one that was measured, and with no visible symptom.

This module is the single source of truth for that format, and it is the only
module that names its files. Its constants and its validation are the contract
the `meta.json` files on disk were written against, so they are not to be
loosened to make a directory load. Two rules follow:

* **The metadata path is stdlib-only.** Resolving an id, reading `meta.json`,
  validating it and verifying the recorded digests must all work on a machine
  with nothing installed — including one with no numpy. `numpy` is imported
  inside the functions that return an array, never at module scope.
* **Loading an array verifies its recorded sha256.** A truncated or edited
  array steers weakly and looks exactly like a weak effect.

Deriving a vector is not here — this module only reads a directory that already
exists, validates it, and converts it for `pod/`.

The layer convention, stated once here because this is where it is recorded:
**"layer L" is the residual stream at the input of decoder block L**, `0` to
`n_layers - 1`. One consequence is worth having in writing, because it is the
sort of thing that gets "fixed" into an off-by-one: an external 65-entry
`hidden_states` tuple (the embedding output followed by the output of each of the
64 blocks) has, at index L, exactly the quantity this repository calls layer L,
for every `L >= 1`. Only index 0 differs — embedding output there, block-0 input
here. So a vector derived at layer 36 and a `hidden_states[36]` are the same
quantity, and no adjustment is needed when re-deriving one from the other.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .core import canonjson, clock, digest, modelprofile, paths, refusals

PROFILE = modelprofile.PROFILE

# ---------------------------------------------------------------------------
# the files
# ---------------------------------------------------------------------------

#: The served array: the designated layer's row of the derivation.
VECTOR_NAME = "vector.npy"

#: The difference of means at *every* block. Kept because it makes the layer
#: choice auditable and re-selectable without re-capturing 540 prompts through a
#: 27B model, and because it is what proves `vector.npy` is the row it claims.
DELTAS_NAME = "deltas_all_layers.npy"

#: Machine-readable description of everything needed to interpret the array.
META_NAME = "meta.json"

#: The vector card: the same content as the metadata, for a person.
README_NAME = "README.md"

#: The prompt set, inside the directory. This is what makes the directory
#: self-describing, and it is the only place a shipped prompt set lives: a prompt
#: file kept anywhere else can be edited after the vector is derived from it, and
#: then nothing on disk says the two disagree.
POSITIVE_NAME = "positive.jsonl"
NEGATIVE_NAME = "negative.jsonl"

#: The strength ladder: what the vector actually does to generations. A
#: subdirectory rather than a single file because it holds both the rendered
#: document and the rung records it was rendered from, so the document can be
#: regenerated without re-running a GPU.
DEMO_DIR_NAME = "demo"
DEMO_LADDER_NAME = "ladder.md"
DEMO_RECORDS_NAME = "ladder.json"

#: Generated table of all vectors, one row each, at `vectors/INDEX.md`.
INDEX_NAME = "INDEX.md"

#: Files every vector directory must have. One-directional on purpose: a missing
#: file makes the directory unreadable or unauditable, while an extra file cannot
#: make the vector wrong, and a format that refuses to load a directory because
#: somebody left a note in it is a format people work around.
REQUIRED_FILES: tuple[str, ...] = (
    VECTOR_NAME,
    DELTAS_NAME,
    META_NAME,
    README_NAME,
    POSITIVE_NAME,
    NEGATIVE_NAME,
)

#: Recorded array dtype, as it appears on disk: little-endian float32. Checked
#: rather than cast. A float64 array of the right shape loads and steers, at
#: double the memory and with a dtype the engine's delta tensor does not share,
#: and a float16 one loses the low bits of a vector whose norm is ~11 against
#: activations of ~82.
ARRAY_DTYPE = "<f4"

#: Vector ids are four digits, zero-padded, and the directory name *is* the id.
#: A free-form name would put the identity of the artefact in two places.
ID_PATTERN = re.compile(r"\d{4}")


# ---------------------------------------------------------------------------
# the conventions, recorded in every vector and checked on load
# ---------------------------------------------------------------------------

#: Where the activations behind a vector were read. Recorded in `meta.json` and
#: compared with this constant when a vector is loaded: a vector captured at a
#: different position is a different measurement, and mixing one into an
#: experiment that assumes this position produces a plausible wrong number.
POSITION_CONVENTION = (
    "last token of the chat-templated prompt "
    "(apply_chat_template(add_generation_prompt=True, enable_thinking=True)); "
    "residual stream at the INPUT of each decoder block"
)

#: Which way the vector points. The whole interpretation of a result depends on
#: it, and it is one sign flip away from the opposite claim.
SIGN_CONVENTION = (
    "vector = mean(positive activations) - mean(negative activations); "
    "positive strength moves toward the POSITIVE set"
)

#: What one unit of strength is worth. Raw strength is **not** comparable between
#: vectors: two vectors at the same strength are two different-sized
#: interventions (0.42 of a typical residual magnitude for one measured vector at
#: layer 32, 0.14 for another at layer 36 — the same strength is a 3x different
#: perturbation). The comparable unit is the relative perturbation below, which
#: is why `meta.json` records the ratio and not only the norm, and why every
#: strength table in this repository carries the relative figure next to the raw
#: one.
STRENGTH_CONVENTION = (
    "strength a adds a * vector to the residual stream at the input of block "
    "`layer`, at every token position. The comparable magnitude is the relative "
    "perturbation a * ||v|| / (mean activation norm at that layer), recorded per "
    "vector as vector_norm_over_activation_norm."
)


# ---------------------------------------------------------------------------
# the metadata contract
# ---------------------------------------------------------------------------

#: The key set of `meta.json`, exactly. Both directions are enforced by
#: :func:`validate_meta`: a missing key breaks a reader, and an unexpected key
#: means a writer recorded something no reader knows about — which is how a
#: vector directory acquires a field that one tool maintains and another ignores.
#: Adding a key here therefore also means teaching whatever writes a vector
#: directory to emit it, or the next directory written will not load.
REQUIRED_META_KEYS: tuple[str, ...] = (
    "id",  # int; the directory name is this, zero-padded to four digits
    "id_str",  # the padded form, so a reader never has to know the padding
    "name",  # short handle, used in the index and in every report
    "description",  # what the two prompt sets contrast, in full sentences
    "model",  # the checkpoint the activations came from; not interchangeable
    "layer",  # THE layer. A property of the vector, never a constant in code
    "n_layers",  # of the model, so a shape can be checked without the profile
    "hidden_size",
    "position_convention",  # must equal POSITION_CONVENTION
    "sign_convention",  # must equal SIGN_CONVENTION
    "n_pos",  # prompt counts, which are also the capture's own check count
    "n_neg",
    "prompt_format",  # what the prompt files contained: kinds, turn patterns
    "vector_norm",  # ||v|| at the designated layer
    "activation_norm_at_layer",  # mean ||h|| there, over both prompt sets
    "vector_norm_over_activation_norm",  # the comparable unit (STRENGTH_CONVENTION)
    "per_layer_delta_norm",  # both stacks, so the layer choice is auditable
    "per_layer_mean_activation_norm",
    "created_at",
    "git_sha",  # may be null: an exported tree has no commit (core.provenance)
    "vllm_version",  # the engine the activations were captured through
    "templated_first_positive_prompt",  # the exact bytes the model receives, once
    "positive_jsonl_sha256",  # the prompt copies, so drift is detectable
    "negative_jsonl_sha256",
    "vector_npy_sha256",  # how a run proves which vector bytes it used
    "deltas_npy_sha256",
    "command",  # the invocation that produced the directory
    "capture",  # how the activations were collected, and its own assertions
)

#: `prompt_format[side]`: what a side's prompt file actually contained. It exists
#: so that a mixed prompt set is visible in the metadata rather than invisible —
#: a system message is templated into a different position with different tokens,
#: so a set where some rows carry one is not the set a reader assumes.
PROMPT_FORMAT_KEYS: tuple[str, ...] = ("kind", "n", "n_with_system", "role_patterns")

#: `capture`: the engine the activations came through, one record per side, and
#: the per-prompt assertions that make the position claim more than a comment.
CAPTURE_KEYS: tuple[str, ...] = ("engine", "positive", "negative", "assertions")

#: `capture[side]`: the count the assertions ran over, and the prompt-length
#: facts that explain the context the engine was sized to.
CAPTURE_SIDE_KEYS: tuple[str, ...] = (
    "n_prompts",  # how many prompts this side's activations were averaged over
    "n_checked",  # how many of those the per-prompt assertions ran on; see below
    "capture_files_from_engine_init",  # the engine's own profiling forward pass
    "seconds",
    "max_model_len",
    "prompt_tokens_min",
    "prompt_tokens_max",
    "prompt_tokens_mean",
)

#: `n_checked` is nullable, and the distinction is the whole point of the field.
#:
#: * an **int** is a claim: "the per-prompt assertions ran on this many of the
#:   prompts". It must equal `n_prompts`, and it may not be 0 —
#:   :func:`check_every_one_of` refuses both, because "0 checked, 0 disagreed"
#:   is what a check that never ran reports.
#: * **null** is the absence of a claim: this capture was not instrumented, so
#:   nothing is asserted about it. It exists so that a capture that really was
#:   uninstrumented can be described honestly instead of by writing a number
#:   nobody counted, which is the failure this field is here to prevent.
#:
#: Construction never writes null: the deriving repository counts and passes the
#: count through :func:`check_every_one_of`. Vector 0007 predates that counting
#: and records null on both sides, which is why the index shows `—` for it.
#: Anything that needs the guarantee calls
#: :func:`require_per_prompt_verification`, which refuses null out loud.


class VectorFormatError(ValueError):
    """A vector directory or its metadata does not match the format.

    A subclass of :class:`ValueError` so that a caller who does not care can
    treat it as one, and a caller that does can tell "this vector is malformed"
    from "the argument I passed was".
    """


# ---------------------------------------------------------------------------
# ids and locations
# ---------------------------------------------------------------------------


def vector_id(value: int | str) -> str:
    """Normalise a vector id to its canonical four-digit form.

    Accepts what the CLI, a metadata record and a server's reported state
    actually carry — `7`, `"7"`, `"0007"` — and returns one string, so that two
    call sites cannot produce two different directory names for one vector.
    """
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise VectorFormatError(
            f"vector id must be an int or a string, got "
            f"{type(value).__name__}: {value!r}"
        )
    text = str(value).strip()
    if not text:
        raise VectorFormatError("vector id is empty")
    if ID_PATTERN.fullmatch(text):
        return text
    if not re.fullmatch(r"\d+", text):
        raise VectorFormatError(
            f"invalid vector id {value!r}: ids are digits only, four of them when "
            f"padded (for example 7 or '0007')"
        )
    number = int(text)
    if not 0 < number < 10000:
        raise VectorFormatError(
            f"invalid vector id {value!r}: ids run from 1 to 9999, and the "
            f"directory name is the id padded to four digits"
        )
    return f"{number:04d}"


def dir_for(value: int | str) -> Path:
    """The directory of one vector. Does not check that it exists."""
    return paths.vector_dir(vector_id(value))


def meta_path(vdir: str | os.PathLike[str]) -> Path:
    return Path(vdir) / META_NAME


def vector_path(vdir: str | os.PathLike[str]) -> Path:
    return Path(vdir) / VECTOR_NAME


def deltas_path(vdir: str | os.PathLike[str]) -> Path:
    return Path(vdir) / DELTAS_NAME


def prompt_path(vdir: str | os.PathLike[str], side: str) -> Path:
    """The copied prompt file for ``side``, which must be positive or negative."""
    if side == "positive":
        return Path(vdir) / POSITIVE_NAME
    if side == "negative":
        return Path(vdir) / NEGATIVE_NAME
    raise VectorFormatError(f"side must be 'positive' or 'negative', got {side!r}")


def demo_dir(vdir: str | os.PathLike[str]) -> Path:
    return Path(vdir) / DEMO_DIR_NAME


def has_demo(vdir: str | os.PathLike[str]) -> bool:
    """Whether a strength ladder has been generated for this vector.

    Optional in the format, and the index says so per vector, because a vector
    with no ladder is a vector whose usable strengths are unknown — which is a
    fact about it, not a reason to refuse to read it.
    """
    return (demo_dir(vdir) / DEMO_LADDER_NAME).is_file()


def index_path() -> Path:
    return paths.vectors_dir() / INDEX_NAME


def existing_ids(vectors_dir: str | os.PathLike[str] | None = None) -> list[str]:
    """The ids of every vector directory present, sorted.

    A directory that matches the id pattern but holds no `meta.json` is not a
    vector; anything else — a stray file, a scratch directory — is ignored here
    and reported by whoever walks the tree for a purpose.
    """
    root = Path(vectors_dir) if vectors_dir is not None else paths.vectors_dir()
    if not root.is_dir():
        return []
    found = [
        child.name
        for child in root.iterdir()
        if child.is_dir()
        and ID_PATTERN.fullmatch(child.name)
        and (child / META_NAME).is_file()
    ]
    return sorted(found)


# ---------------------------------------------------------------------------
# the zero-of-zero refusal
# ---------------------------------------------------------------------------


def check_every_one_of(n_checked: int, n_total: int, *, what: str) -> int:
    """Confirm a claim of the form "all N were checked and all N agreed".

    Returns ``n_total``; raises otherwise. Two failures, and the second is the
    reason this function exists rather than an inline comparison:

    * ``n_checked != n_total`` — some were not checked, so the claim is about a
      subset while it reads as being about the whole.
    * ``n_total == 0`` — **a zero-of-zero is not a pass.** "0 checked, 0
      disagreed" is what a check that never ran reports, and it is
      indistinguishable in a metadata file from a check that ran and passed.
      The shape it takes here: a cached-input path skips the tokenize
      comparison, the metadata still records "cross-checks: 0, disagreeing: 0",
      and a reader takes that for agreement. A claim about every one of nothing
      is not evidence.

    The arithmetic and the wording live in :mod:`.core.refusals`, kept separate
    because the rule is not about vectors. What stays here is the error type: a
    caller reading a vector directory wants :class:`VectorFormatError`, not a
    coverage exception from another layer.
    """
    problem = refusals.coverage_problem(n_checked, n_total, what=what)
    if problem is not None:
        raise VectorFormatError(problem)
    return n_total


# ---------------------------------------------------------------------------
# metadata validation
# ---------------------------------------------------------------------------


def _require(condition: bool, where: str, message: str) -> None:
    if not condition:
        raise VectorFormatError(f"{where}: {message}")


def _require_int(meta: Mapping[str, Any], key: str, where: str) -> int:
    value = meta.get(key)
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        where,
        f"{key} must be an int, got {type(value).__name__}: {value!r}",
    )
    return int(value)  # type: ignore[arg-type]


def _require_number(meta: Mapping[str, Any], key: str, where: str) -> float:
    value = meta.get(key)
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        where,
        f"{key} must be a number, got {type(value).__name__}: {value!r}",
    )
    return float(value)  # type: ignore[arg-type]


def _require_text(meta: Mapping[str, Any], key: str, where: str) -> str:
    value = meta.get(key)
    _require(
        isinstance(value, str) and value.strip() != "",
        where,
        f"{key} must be a non-empty string, got {value!r}",
    )
    return str(value)


def _require_digest(meta: Mapping[str, Any], key: str, where: str) -> str:
    value = _require_text(meta, key, where)
    _require(
        re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        where,
        f"{key} is not a sha-256 hex digest: {value!r}",
    )
    return value


def _require_norm_list(
    meta: Mapping[str, Any], key: str, n_layers: int, where: str
) -> list[float]:
    value = meta.get(key)
    _require(
        isinstance(value, list),
        where,
        f"{key} must be a list, got {type(value).__name__}",
    )
    assert isinstance(value, list)
    _require(
        len(value) == n_layers,
        where,
        f"{key} has {len(value)} entries but the model has {n_layers} layers; "
        f"a short stack means the metadata and the arrays came from different runs",
    )
    for i, entry in enumerate(value):
        _require(
            isinstance(entry, (int, float)) and not isinstance(entry, bool),
            where,
            f"{key}[{i}] is not a number: {entry!r}",
        )
        _require(
            math.isfinite(float(entry)) and float(entry) >= 0.0,
            where,
            f"{key}[{i}] is not a finite non-negative norm: {entry!r}",
        )
    return [float(x) for x in value]


def _validate_prompt_format(fmt: Any, side: str, expected_n: int, where: str) -> None:
    place = f"{where}: prompt_format.{side}"
    _require(
        isinstance(fmt, dict), place, f"must be an object, got {type(fmt).__name__}"
    )
    assert isinstance(fmt, dict)
    _check_key_set(fmt, PROMPT_FORMAT_KEYS, place)
    kind = _require_text(fmt, "kind", place)
    _require(
        kind in ("text", "messages", "mixed"),
        place,
        f"kind {kind!r} is not one of 'text', 'messages', 'mixed'",
    )
    n = _require_int(fmt, "n", place)
    _require(
        n == expected_n,
        place,
        f"n is {n} but the metadata records {expected_n} {side} prompts; the "
        f"summary and the count must describe the same file",
    )
    n_with_system = _require_int(fmt, "n_with_system", place)
    _require(
        0 <= n_with_system <= n,
        place,
        f"n_with_system {n_with_system} is not between 0 and {n}",
    )
    patterns = fmt.get("role_patterns")
    _require(
        isinstance(patterns, dict) and bool(patterns),
        place,
        f"role_patterns must be a non-empty object, got {patterns!r}",
    )
    assert isinstance(patterns, dict)
    total = 0
    for pattern, count in patterns.items():
        _require(
            isinstance(pattern, str) and pattern != "", place, "empty turn pattern"
        )
        _require(
            isinstance(count, int) and not isinstance(count, bool) and count > 0,
            place,
            f"role_patterns[{pattern!r}] must be a positive int, got {count!r}",
        )
        total += int(count)
    _require(
        total == n,
        place,
        f"role_patterns cover {total} rows but n is {n}; a row counted under no "
        f"pattern is a row whose shape nothing in the metadata describes",
    )


def _validate_capture(capture: Any, n_pos: int, n_neg: int, where: str) -> None:
    place = f"{where}: capture"
    _require(
        isinstance(capture, dict),
        place,
        f"must be an object, got {type(capture).__name__}",
    )
    assert isinstance(capture, dict)
    _check_key_set(capture, CAPTURE_KEYS, place)
    _require_text(capture, "engine", place)
    # The `assertions` string states what was checked for every prompt. It is
    # prose, but it is the claim the counts below are the evidence for, so it may
    # not be empty: a count with no statement of what it counted is not a record.
    _require_text(capture, "assertions", place)
    for side, expected in (("positive", n_pos), ("negative", n_neg)):
        side_place = f"{place}.{side}"
        record = capture.get(side)
        _require(
            isinstance(record, dict),
            side_place,
            f"must be an object, got {type(record).__name__}",
        )
        assert isinstance(record, dict)
        _check_key_set(record, CAPTURE_SIDE_KEYS, side_place)
        n_prompts = _require_int(record, "n_prompts", side_place)
        _require(
            n_prompts == expected,
            side_place,
            f"n_prompts is {n_prompts} but the metadata records {expected} {side} "
            f"prompts; the capture and the prompt file must be the same set",
        )
        # The per-prompt assertions are the only thing standing between "captured
        # the last prompt token" and "captured a padding row", so the record says
        # how many prompts they actually ran over. A number here is a claim and is
        # checked as one; null is the explicit absence of a claim (see the note on
        # CAPTURE_SIDE_KEYS).
        if record["n_checked"] is not None:
            check_every_one_of(
                _require_int(record, "n_checked", side_place),
                n_prompts,
                what=f"{side_place}: per-prompt capture assertions",
            )
        _require_int(record, "capture_files_from_engine_init", side_place)
        _require_number(record, "seconds", side_place)
        max_model_len = _require_int(record, "max_model_len", side_place)
        token_min = _require_int(record, "prompt_tokens_min", side_place)
        token_max = _require_int(record, "prompt_tokens_max", side_place)
        _require_number(record, "prompt_tokens_mean", side_place)
        _require(
            0 < token_min <= token_max,
            side_place,
            f"prompt token counts {token_min}..{token_max} are not a positive range",
        )
        _require(
            max_model_len >= token_max,
            side_place,
            f"max_model_len {max_model_len} is below the longest prompt "
            f"({token_max} tokens); the capture engine could not have held it, so "
            f"either the record is wrong or a prompt was silently truncated",
        )


def _check_key_set(obj: Mapping[str, Any], expected: Sequence[str], where: str) -> None:
    """Both directions: nothing missing, nothing unexpected."""
    missing = [key for key in expected if key not in obj]
    unexpected = [key for key in obj if key not in expected]
    if not missing and not unexpected:
        return
    parts = []
    if missing:
        parts.append(f"missing keys: {', '.join(sorted(missing))}")
    if unexpected:
        parts.append(f"unexpected keys: {', '.join(sorted(unexpected))}")
    raise VectorFormatError(
        f"{where} does not match the recorded contract ({'; '.join(parts)}). Every "
        f"reader of a vector directory expects exactly the keys named in "
        f"steering_vectors.vectorfmt; add or remove one there, and in the "
        f"writer and the card, rather than only where it happens to be written."
    )


def validate_meta(meta: Mapping[str, Any], *, where: str = META_NAME) -> dict[str, Any]:
    """Return ``meta`` as a plain dict if it is a valid record, else raise.

    Everything checkable without loading an array is checked here, because this
    is the function every reader goes through and a metadata record is the only
    thing that says what the array means. In particular the recorded conventions
    must be *this* module's conventions: a vector captured at another position,
    or with the opposite sign, is a different measurement that would otherwise be
    used silently as if it were this one.
    """
    _require(
        isinstance(meta, Mapping),
        where,
        f"must be an object, got {type(meta).__name__}",
    )
    _check_key_set(meta, REQUIRED_META_KEYS, where)

    vid = _require_int(meta, "id", where)
    _require(0 < vid < 10000, where, f"id {vid} is outside 1..9999")
    id_str = _require_text(meta, "id_str", where)
    _require(
        id_str == f"{vid:04d}",
        where,
        f"id_str {id_str!r} is not id {vid} padded to four digits; the directory "
        f"name is id_str, so two forms that disagree name two directories",
    )
    _require_text(meta, "name", where)
    # A description is required, not optional: "self-describing" means a reader
    # who has only this directory can say what the vector contrasts. An empty
    # description leaves that only in the prompt files, 540 rows of them.
    _require_text(meta, "description", where)

    model = _require_text(meta, "model", where)
    _require(
        model == PROFILE.model_id,
        where,
        f"model {model!r} is not the profiled checkpoint {PROFILE.model_id!r}. "
        f"Activations are not comparable across checkpoints, so this vector "
        f"cannot be used with this profile; point core.modelprofile at the model "
        f"it was derived from, or derive a new vector.",
    )
    n_layers = _require_int(meta, "n_layers", where)
    hidden_size = _require_int(meta, "hidden_size", where)
    _require(
        n_layers == PROFILE.n_layers and hidden_size == PROFILE.hidden_size,
        where,
        f"shape facts ({n_layers} layers x {hidden_size}) disagree with the "
        f"profile ({PROFILE.n_layers} x {PROFILE.hidden_size})",
    )
    layer = _require_int(meta, "layer", where)
    try:
        PROFILE.check_layer(layer, what=f"{where}: layer")
    except (TypeError, ValueError) as exc:
        raise VectorFormatError(str(exc)) from exc

    position = _require_text(meta, "position_convention", where)
    _require(
        position == POSITION_CONVENTION,
        where,
        f"position_convention is not the one this code implements.\n"
        f"  recorded: {position}\n  expected: {POSITION_CONVENTION}\n"
        f"A vector captured at another position measures another quantity; using "
        f"it here would produce a number that looks like this experiment's.",
    )
    sign = _require_text(meta, "sign_convention", where)
    _require(
        sign == SIGN_CONVENTION,
        where,
        f"sign_convention is not the one this code implements.\n"
        f"  recorded: {sign}\n  expected: {SIGN_CONVENTION}\n"
        f"A flipped sign inverts the interpretation of every result taken with "
        f"this vector, and nothing in the output looks wrong.",
    )

    n_pos = _require_int(meta, "n_pos", where)
    n_neg = _require_int(meta, "n_neg", where)
    for side, count in (("positive", n_pos), ("negative", n_neg)):
        _require(
            count > 0,
            where,
            f"n_{side[:3]} is {count}: a mean over no prompts is not a "
            f"measurement, and a vector derived from an empty side would be "
            f"either zero or the other side's mean",
        )
    fmt = meta.get("prompt_format")
    _require(
        isinstance(fmt, dict), where, f"prompt_format must be an object, got {fmt!r}"
    )
    assert isinstance(fmt, dict)
    _check_key_set(fmt, ("positive", "negative"), f"{where}: prompt_format")
    _validate_prompt_format(fmt["positive"], "positive", n_pos, where)
    _validate_prompt_format(fmt["negative"], "negative", n_neg, where)

    vector_norm = _require_number(meta, "vector_norm", where)
    act_norm = _require_number(meta, "activation_norm_at_layer", where)
    ratio = _require_number(meta, "vector_norm_over_activation_norm", where)
    for key, value in (
        ("vector_norm", vector_norm),
        ("activation_norm_at_layer", act_norm),
        ("vector_norm_over_activation_norm", ratio),
    ):
        _require(
            math.isfinite(value) and value > 0.0,
            where,
            f"{key} is not positive: {value}",
        )
    _require(
        math.isclose(ratio, vector_norm / act_norm, rel_tol=1e-9),
        where,
        f"vector_norm_over_activation_norm {ratio} is not vector_norm "
        f"{vector_norm} / activation_norm_at_layer {act_norm} = "
        f"{vector_norm / act_norm}. That ratio is the only strength unit that is "
        f"comparable between vectors (STRENGTH_CONVENTION), so a stale copy of it "
        f"licenses an intervention several times the intended size.",
    )
    delta_norms = _require_norm_list(meta, "per_layer_delta_norm", n_layers, where)
    _require_norm_list(meta, "per_layer_mean_activation_norm", n_layers, where)
    _require(
        math.isclose(delta_norms[layer], vector_norm, rel_tol=1e-6),
        where,
        f"per_layer_delta_norm[{layer}] is {delta_norms[layer]} but vector_norm is "
        f"{vector_norm}; vector.npy is supposed to be row {layer} of the delta "
        f"stack, and these two numbers describe different rows",
    )

    created_at = _require_text(meta, "created_at", where)
    try:
        clock.parse_iso(created_at)
    except ValueError as exc:
        raise VectorFormatError(
            f"{where}: created_at is not a recorded timestamp: {exc}"
        ) from exc
    git_sha = meta.get("git_sha")
    _require(
        git_sha is None or (isinstance(git_sha, str) and git_sha.strip() != ""),
        where,
        f"git_sha must be a non-empty string or null, got {git_sha!r}. Null is a "
        f"real answer — an exported tree has no commit — but an empty string is "
        f"a version that was recorded and lost.",
    )
    _require_text(meta, "vllm_version", where)
    _require_text(meta, "templated_first_positive_prompt", where)
    for key in (
        "positive_jsonl_sha256",
        "negative_jsonl_sha256",
        "vector_npy_sha256",
        "deltas_npy_sha256",
    ):
        _require_digest(meta, key, where)
    _require_text(meta, "command", where)
    _validate_capture(meta.get("capture"), n_pos, n_neg, where)
    return dict(meta)


def per_prompt_verification(meta: Mapping[str, Any]) -> dict[str, int | None]:
    """Per side: how many prompts the capture assertions ran on, or None.

    None means the record makes no claim. It does not mean zero, and it must not
    be read as one — the two are the same number of verified prompts but a very
    different statement about the metadata.
    """
    capture = meta["capture"]
    return {side: capture[side]["n_checked"] for side in ("positive", "negative")}


def require_per_prompt_verification(meta: Mapping[str, Any], *, what: str) -> None:
    """Raise unless both sides claim, and back, per-prompt verified capture.

    Called by anything whose own correctness rests on every activation having
    come from the intended token position — the position convention is a
    statement about the capture, and this is the only recorded evidence for it.
    """
    for side, n_checked in per_prompt_verification(meta).items():
        n_prompts = int(meta["capture"][side]["n_prompts"])
        if n_checked is None:
            raise VectorFormatError(
                f"{what}: vector {meta['id_str']} records no per-prompt capture "
                f"verification for its {side} side (n_checked is null), so there "
                f"is no evidence that its activations came from the token "
                f"position {META_NAME} claims. The record is honest about not "
                f"knowing; this caller needs to know. Re-derive the vector with "
                f"the instrumented capture path, or use it somewhere the "
                f"position claim is not load-bearing."
            )
        check_every_one_of(int(n_checked), n_prompts, what=f"{what}: capture.{side}")


def read_meta(vdir: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and validate one vector's `meta.json`.

    Stdlib only, so this is usable on a machine with no numpy — which is most of
    them: resolving a vector, showing what it is and verifying that its files are
    the recorded bytes needs no array.
    """
    path = meta_path(vdir)
    if not path.is_file():
        raise VectorFormatError(
            f"{path} does not exist, so {Path(vdir)} is not a vector directory. A "
            f"vector is a directory containing {', '.join(REQUIRED_FILES)}."
        )
    raw = canonjson.read_json(path)
    meta = validate_meta(raw, where=str(path))
    name = Path(vdir).name
    if ID_PATTERN.fullmatch(name) and name != meta["id_str"]:
        raise VectorFormatError(
            f"{path}: the record says id {meta['id_str']} but it is in a directory "
            f"named {name}. The directory name is the id every other part of the "
            f"system uses, so a copied-and-not-renamed directory would be served "
            f"under one id while reporting another."
        )
    return meta


def check_files_present(vdir: str | os.PathLike[str]) -> None:
    """Raise unless every file the format requires is present."""
    root = Path(vdir)
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise VectorFormatError(
            f"{root} is missing {', '.join(missing)}. The directory is the whole "
            f"description of the vector: without the prompt copies it cannot be "
            f"audited, and without the delta stack the layer choice cannot be "
            f"checked against the served row."
        )


def verify_digests(
    vdir: str | os.PathLike[str], meta: Mapping[str, Any] | None = None
) -> None:
    """Check every file the metadata records a digest for, without numpy.

    Separate from loading because it is the whole check a launcher or a report
    needs: it proves the directory holds the bytes the metadata describes, which
    is what "this run used vector 0007" has to mean.
    """
    root = Path(vdir)
    record = dict(meta) if meta is not None else read_meta(root)
    check_files_present(root)
    for name, key in (
        (VECTOR_NAME, "vector_npy_sha256"),
        (DELTAS_NAME, "deltas_npy_sha256"),
        (POSITIVE_NAME, "positive_jsonl_sha256"),
        (NEGATIVE_NAME, "negative_jsonl_sha256"),
    ):
        path = root / name
        actual = digest.sha256_file(path)
        expected = record[key]
        if actual != expected:
            raise VectorFormatError(
                f"{path} is not the file this vector was described from:\n"
                f"  recorded {key} = {expected}\n  actual           = {actual}\n"
                f"An edited or truncated array steers weakly and looks exactly "
                f"like a weak effect; an edited prompt file makes the recorded "
                f"derivation unreproducible. Restore the file or re-derive."
            )


# ---------------------------------------------------------------------------
# the arrays (the only functions here that need numpy)
# ---------------------------------------------------------------------------


def _load_array(path: Path, *, expected_shape: tuple[int, ...], what: str) -> Any:
    """Load one `.npy`, checking shape, dtype and finiteness. Imports numpy.

    numpy is imported here rather than at module scope so that everything above
    works on a machine with nothing installed.
    """
    import numpy as np

    if not path.is_file():
        raise VectorFormatError(f"{path} does not exist")
    try:
        array = np.load(path, allow_pickle=False)
    except ValueError as exc:
        # allow_pickle=False turns a pickled payload into this, which is the
        # point: a vector directory is data, and loading it must not execute
        # anything that was put in it.
        raise VectorFormatError(f"{path} is not a plain array file: {exc}") from exc
    if tuple(array.shape) != expected_shape:
        raise VectorFormatError(
            f"{what} at {path} has shape {tuple(array.shape)}, expected "
            f"{expected_shape} for {PROFILE.model_id}"
        )
    if array.dtype.str != ARRAY_DTYPE:
        raise VectorFormatError(
            f"{what} at {path} has dtype {array.dtype.str}, expected {ARRAY_DTYPE} "
            f"(little-endian float32). The engine's delta tensor is built from "
            f"this array; a different dtype is either a different precision than "
            f"was measured or a byte order that reads as different numbers."
        )
    if not bool(np.all(np.isfinite(array))):
        raise VectorFormatError(
            f"{what} at {path} contains non-finite values. Added to the residual "
            f"stream, a single nan propagates to every later block and the model "
            f"produces empty output rather than an error."
        )
    return array


def load_vector(
    vdir: str | os.PathLike[str], meta: Mapping[str, Any] | None = None
) -> Any:
    """The served array for one vector, verified against its metadata.

    Verified means: the recorded sha256 of the file, the shape and dtype, no
    non-finite entries, and that its norm is the norm the metadata reports. The
    last one is not implied by the digest — the digest proves the file has not
    changed, and the norm proves the metadata is describing *this* file rather
    than having been copied from another vector.
    """
    import numpy as np

    root = Path(vdir)
    record = dict(meta) if meta is not None else read_meta(root)
    path = vector_path(root)
    actual = digest.sha256_file(path)
    if actual != record["vector_npy_sha256"]:
        raise VectorFormatError(
            f"{path} does not match its recorded sha256:\n"
            f"  recorded {record['vector_npy_sha256']}\n  actual   {actual}\n"
            f"A truncated or edited vector steers weakly, which is "
            f"indistinguishable from a weak effect in every measurement taken "
            f"with it."
        )
    array = _load_array(path, expected_shape=PROFILE.vector_shape, what="vector")
    norm = float(np.linalg.norm(array.astype(np.float64)))
    recorded = float(record["vector_norm"])
    if not math.isclose(norm, recorded, rel_tol=1e-6):
        raise VectorFormatError(
            f"{path} has norm {norm:.6f} but its metadata records "
            f"{recorded:.6f}. The digest matches, so the file is intact and the "
            f"metadata belongs to a different array — every strength derived from "
            f"the recorded ratio would be the wrong size."
        )
    return array


def load_deltas(
    vdir: str | os.PathLike[str], meta: Mapping[str, Any] | None = None
) -> Any:
    """The whole per-layer delta stack, verified the same way.

    Loaded by anything that re-selects a layer or audits the choice of one; it is
    ~64x the size of the vector, so it is a separate call.
    """
    root = Path(vdir)
    record = dict(meta) if meta is not None else read_meta(root)
    path = deltas_path(root)
    actual = digest.sha256_file(path)
    if actual != record["deltas_npy_sha256"]:
        raise VectorFormatError(
            f"{path} does not match its recorded sha256:\n"
            f"  recorded {record['deltas_npy_sha256']}\n  actual   {actual}"
        )
    return _load_array(path, expected_shape=PROFILE.deltas_shape, what="delta stack")


def check_vector_is_delta_row(vector: Any, deltas: Any, layer: int) -> None:
    """Confirm the served array is the delta stack's row for the recorded layer.

    This is the one check that can catch a vector derived at one layer and
    recorded as another, because it compares the two files that would have to
    disagree for that to have happened. Exact equality is required: the row is
    written by copying it, not by recomputing it.
    """
    import numpy as np

    PROFILE.check_layer(layer)
    if not bool(np.array_equal(vector, deltas[layer])):
        raise VectorFormatError(
            f"vector.npy is not row {layer} of the delta stack. Either the vector "
            f"was derived at a different layer than the metadata records — in "
            f"which case every measurement taken with it steers the wrong block "
            f"and still looks like a result — or one of the two files was "
            f"replaced."
        )


# ---------------------------------------------------------------------------
# using a vector: the unit direction, the read-out, the strength unit
# ---------------------------------------------------------------------------


def unit_vector(vector: Any) -> Any:
    """``v / ||v||`` in float64: the direction a read-out projects onto.

    Derived from `vector.npy` every time rather than stored, so there is exactly
    one place the direction can come from and no second file to fall out of step.

    A zero-norm input raises. That is not a theoretical case: real arrays in this
    system contain all-zero rows — the delta at block 0 of a derivation is
    numerically zero, and the unembedding row for token id 0 on this checkpoint
    is all zeros — so any "normalise whatever you are given" helper silently
    returns nan, and a nan direction poisons every projection and every mean
    taken over one without raising anywhere.
    """
    import numpy as np

    array = np.asarray(vector, dtype=np.float64)
    if array.ndim != 1:
        raise VectorFormatError(
            f"expected a 1-D vector, got shape {tuple(array.shape)}"
        )
    if not bool(np.all(np.isfinite(array))):
        raise VectorFormatError(
            "cannot normalise a vector containing non-finite values"
        )
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        raise VectorFormatError(
            "cannot derive a unit vector from an all-zero vector: it has no "
            "direction. Dividing by its norm yields nan, which propagates "
            "silently through every projection and every average of one."
        )
    return array / norm


def projection(activation: Any, unit: Any) -> float:
    """``h . v_hat``: the read-out, in float64.

    The whole read-out. There is no trained probe, no scaler and no percentile
    table: one number, the component of the activation along the vector's own
    direction, computed from the vector directory and nothing else.

    float64 because the activation arrives as float32 (or bf16 widened to it) and
    a 5120-term dot product accumulated in float32 loses low bits that matter
    when the quantity being compared is a difference between two arms.
    """
    import numpy as np

    h = np.asarray(activation, dtype=np.float64)
    u = np.asarray(unit, dtype=np.float64)
    if h.shape != u.shape:
        raise VectorFormatError(
            f"activation shape {tuple(h.shape)} and direction shape "
            f"{tuple(u.shape)} differ, so this projection would broadcast into a "
            f"number that is not a projection"
        )
    return float(np.dot(h, u))


def relative_perturbation(strength: float, meta: Mapping[str, Any]) -> float:
    """What ``strength`` is worth on this vector, as a fraction of the stream.

    ``a * ||v|| / (mean activation norm at the layer)``. This is the only figure
    comparable between vectors (see :data:`STRENGTH_CONVENTION`), so every table
    that lists a raw strength lists this next to it.
    """
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise VectorFormatError(
            f"strength must be a number, got {type(strength).__name__}: {strength!r}"
        )
    ratio = float(meta["vector_norm_over_activation_norm"])
    return float(strength) * ratio


def describe(meta: Mapping[str, Any]) -> str:
    """One line naming a vector, for a log or a failure message."""
    return (
        f"vector {meta['id_str']} '{meta['name']}' at layer {meta['layer']} of "
        f"{meta['model']} (||v|| {float(meta['vector_norm']):.4f}, "
        f"{100 * float(meta['vector_norm_over_activation_norm']):.1f}% of the "
        f"stream per unit strength, {meta['n_pos']}/{meta['n_neg']} prompts)"
    )


def load_unit_vector(
    vdir: str | os.PathLike[str], meta: Mapping[str, Any] | None = None
) -> Any:
    """:func:`unit_vector` of :func:`load_vector` — the read-out's direction."""
    record = dict(meta) if meta is not None else read_meta(vdir)
    return unit_vector(load_vector(vdir, record))


# ---------------------------------------------------------------------------
# serving a vector: the two conversions a forward-hook engine needs
# ---------------------------------------------------------------------------
#
# `pod/src/vllm_steering` adds `alpha * v` to the *output* of one decoder block,
# with `alpha = steer_strength * scale`. Neither the block nor the scale is
# recorded in `meta.json`, because neither is a property of the vector: they are
# this format's layer and strength conventions restated for an engine that hooks
# outputs. Both are derived here, from the metadata, every time — the whole
# reason the format keeps no side table of per-vector constants is that a
# recorded copy of a derived number is a copy that can go stale.


def steer_layer(meta: Mapping[str, Any]) -> int:
    """The block whose **output** to hook, for a vector derived at ``layer``.

    ``meta["layer"]`` is this format's layer: the residual stream at the *input*
    of decoder block L (see this module's docstring). That is where the
    activations behind the vector were read, and it is where adding the vector
    reproduces the difference that was measured. A forward hook fires on a
    block's *output*, and the output of block L-1 is the input of block L, so
    the block to hook is one below the recorded layer.

    Spelled out as a function rather than inlined at the call site because
    getting it wrong is invisible: steering block 36 instead of block 35 still
    steers, still produces fluent text, and still moves the model in *a*
    direction — just not the one that was measured, and nothing in the output
    says so.

    Block 0 has no predecessor, so a vector derived at layer 0 cannot be served
    by an output hook at all. Refused rather than clamped: ``-1`` is a valid
    Python index and would silently steer the last block.
    """
    layer = _require_int(meta, "layer", "steer_layer")
    if layer == 0:
        raise VectorFormatError(
            f"vector {meta['id_str']} was derived at layer 0 — the residual "
            f"stream at the input of block 0, which is the embedding output. No "
            f"decoder block produces it, so there is no block whose output hook "
            f"can add to it, and hooking block -1 would silently steer the last "
            f"block instead. Serve this one with an input hook or not at all."
        )
    return layer - 1


def steer_scale(meta: Mapping[str, Any]) -> float:
    """``activation_norm_at_layer / ||v||`` — what one unit of strength is worth.

    An engine multiplying a request's strength by this adds a delta of
    ``strength * activation_norm_at_layer``, so ``strength = 1.0`` is a
    perturbation the size of a typical residual-stream row at the vector's own
    layer. That is the unit :data:`STRENGTH_CONVENTION` names as the only one
    comparable between vectors: these four vectors' norms differ by a factor of
    3.6 at the same layer, so the same *raw* coefficient is 3.6 different
    interventions.

    The reciprocal of the recorded ``vector_norm_over_activation_norm``, and
    computed from the two norms rather than read from that field so that a
    server and a client that both scale cannot disagree by a rounding.
    """
    norm = _require_number(meta, "vector_norm", "steer_scale")
    activation = _require_number(meta, "activation_norm_at_layer", "steer_scale")
    if not norm > 0.0:
        raise VectorFormatError(
            f"vector {meta['id_str']} records ||v|| = {norm}, so a strength "
            f"cannot be expressed as a fraction of the stream: every strength "
            f"would scale a direction that has no length."
        )
    return activation / norm


def iter_vectors(
    vectors_dir: str | os.PathLike[str] | None = None,
) -> Iterable[dict[str, Any]]:
    """Every vector's validated metadata, in id order.

    Raises on the first directory that does not validate rather than skipping it.
    A generated listing that quietly omits a vector reports a smaller collection
    than exists on disk, and the omitted one is the one something is wrong with.
    """
    root = Path(vectors_dir) if vectors_dir is not None else paths.vectors_dir()
    for vid in existing_ids(root):
        yield read_meta(root / vid)
