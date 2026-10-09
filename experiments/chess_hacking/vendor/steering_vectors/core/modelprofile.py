"""Every model-specific fact the vector format checks against, in one object.

Nothing else in this package hard-codes a layer count, a hidden size or a
checkpoint name. Those facts are measurements taken against one checkpoint, and
scattering them as literals is what makes a model swap a search-and-replace
across the tree instead of one new :class:`ModelProfile`.

Swapping models is a matter of defining another profile and pointing
:data:`PROFILE` at it — after re-deriving the vectors, because a vector is a
difference of activations of one checkpoint and is not comparable across two.

Deliberately contains only identity/shape facts plus the architecture facts the
capture-only builder needs. Serving and sampling facts remain outside this
profile.

Standard library only, and it holds no tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, eq=False)
class ModelProfile:
    """The description of one checkpoint as this package uses it.

    Compared by identity: there is one instance, and two profiles that differ in
    any field describe different experiments.
    """

    # ---- identity -------------------------------------------------------
    model_id: str
    """The checkpoint the activations were captured from, as recorded in every
    vector's `meta.json` and as passed to the server that serves it."""

    dtype: str
    """Weight dtype the activations were measured at. Quantization is not an
    option here: it changes activations, and every norm, ratio and strength
    recorded with these vectors is a measurement of the unquantized residual
    stream."""

    # ---- shape ----------------------------------------------------------
    n_layers: int
    """Decoder blocks, indexed ``0 .. n_layers - 1``. Every layer argument in
    the package is bounds-checked against this."""

    hidden_size: int
    """Residual-stream width. A steering vector is ``(hidden_size,)`` float32
    and a full delta stack is ``(n_layers, hidden_size)``; a wrong-shaped array
    must be rejected when it is loaded rather than broadcast into a wrong
    answer."""

    # ---- capture architecture (verified against vLLM 0.26.0) ------------
    architecture: str
    """Checkpoint architecture key replaced by the capture plugin."""

    architecture_module: str
    """Private vLLM module containing the original architecture."""

    decoder_blocks_path: str
    """Attribute path from the architecture object to its decoder blocks."""

    residual_convention: str
    """How the true block-input residual stream is reconstructed."""

    verified_vllm_version: str
    """vLLM version against which the private replacement path was verified."""

    #: There is deliberately no default layer here. A layer is a property of the
    #: vector that was derived at it, recorded in that vector's metadata, and
    #: read from there by everything that steers or reads out with it. A
    #: model-level default would be a second, unowned source for the same
    #: number, and the failure it produces — steering at a layer the vector was
    #: not built for — is invisible in the output. Layer *bounds* are a model
    #: fact, and they are enforced by :meth:`check_layer`.

    notes: tuple[str, ...] = field(default=())
    """Facts with no natural field of their own."""

    # ---- derived shapes and validators ----------------------------------
    @property
    def vector_shape(self) -> tuple[int, ...]:
        """Shape of a single steering vector."""
        return (self.hidden_size,)

    @property
    def deltas_shape(self) -> tuple[int, ...]:
        """Shape of the per-layer delta stack produced by a derivation."""
        return (self.n_layers, self.hidden_size)

    def check_layer(self, layer: int, *, what: str = "layer") -> int:
        """Return ``layer`` if it indexes a decoder block, else raise.

        A layer index arrives from a command line and from a vector's metadata.
        Out of range, it either indexes from the end (negative) or fails deep
        inside the engine, where the message no longer mentions the layer.
        """
        if not isinstance(layer, int) or isinstance(layer, bool):
            raise TypeError(
                f"{what} must be an int, got {type(layer).__name__}: {layer!r}"
            )
        if not 0 <= layer < self.n_layers:
            raise ValueError(
                f"{what} {layer} out of range 0..{self.n_layers - 1} "
                f"for {self.model_id} ({self.n_layers} decoder blocks)"
            )
        return layer

    def check_vector_shape(
        self, shape: tuple[int, ...], *, what: str = "vector"
    ) -> None:
        """Raise unless ``shape`` is this model's vector shape."""
        if tuple(shape) != self.vector_shape:
            raise ValueError(
                f"{what} has shape {tuple(shape)}, expected {self.vector_shape} "
                f"for {self.model_id} (hidden size {self.hidden_size})"
            )

    def check_deltas_shape(
        self, shape: tuple[int, ...], *, what: str = "deltas"
    ) -> None:
        """Raise unless ``shape`` is this model's per-layer delta shape."""
        if tuple(shape) != self.deltas_shape:
            raise ValueError(
                f"{what} has shape {tuple(shape)}, expected {self.deltas_shape} "
                f"for {self.model_id} ({self.n_layers} layers × {self.hidden_size})"
            )


QWEN3_6_27B = ModelProfile(
    model_id="Qwen/Qwen3.6-27B",
    dtype="bfloat16",
    n_layers=64,
    hidden_size=5120,
    # Capture facts ported from local v2-steering-tools, where they were verified
    # around commits ea65b2f, a486b53, d732a8c, 6508dab, and 6678408.
    architecture="Qwen3_5ForConditionalGeneration",
    architecture_module="vllm.model_executor.models.qwen3_5",
    decoder_blocks_path="language_model.model.layers",
    residual_convention=(
        "hidden_states if residual is None else hidden_states + residual"
    ),
    verified_vllm_version="0.26.0",
    notes=(
        "The residual stream is read and steered at the *input* of a block, so "
        "'layer L' throughout the vector format means 'before block L runs'. "
        "A server that hooks a block's *output* therefore steers layer L when "
        "it is pointed at block L-1; see vectorfmt.steer_layer.",
    ),
)

#: The profile every module uses. One instance, named separately from the
#: checkpoint it describes so that call sites read as "the model", not "Qwen".
PROFILE = QWEN3_6_27B
