"""Per-request activation steering, applied inside vLLM's model runner.

Adds `alpha_t * V[row_t]` to the residual stream at the output of one decoder
block, for every token position `t`. Both numbers come off the request: `row_t`
from the vector id it named, `alpha_t` from the strength. So requests using
different vectors at different strengths coexist in a single batch, and neither
the vector nor the strength ever requires a restart.

`V` is the whole served set as one matrix, uploaded once at model load, with
each row already multiplied by its own scale (`store.matrix`). A request that
names no vector is row 0, which is zeros, at strength 0 — the unsteered model,
reached by the same arithmetic as everything else rather than by a branch.

Three patches on `GPUModelRunner`:

* `load_model` locates, verifies and substitutes the target decoder block's
  class, and
  allocates the per-token row and alpha buffers.
* `_prepare_inputs` fills those buffers with one entry per flattened token,
  using vLLM's own request-to-token flattening.
* `_dummy_run` zeroes them, since profiling batches carry no request state and
  must run unsteered.

The steering is applied by substituting the target block's class, not by
attaching a forward hook to it. A hook is a host-side callback: whether
`torch.compile` absorbs it into the graph it builds is an implementation detail
of the compiler, and if a version ever stops absorbing it the hook still runs
while the graph is captured and silently stops running on replay — a server that
looks steered and is not. Overriding `forward` puts the arithmetic where the
compiler has to trace it to compile the model correctly at all, so the question
does not arise and the server can run with CUDA graphs on.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch

from . import store
from .config import SteerConfig, config_from_env

logger = logging.getLogger("vllm_steering")


class _State:
    config: SteerConfig | None = None
    store: store.Store | None = None
    vectors: torch.Tensor | None = None
    alpha: torch.Tensor | None = None
    row: torch.Tensor | None = None
    alpha_staging: torch.Tensor | None = None
    row_staging: torch.Tensor | None = None
    n_filled: int = 0
    ready: bool = False
    step: int = 0
    unhonoured: frozenset[str] = frozenset()


_st = _State()


def _log(message: str) -> None:
    logger.warning("[steer][pid %d] %s", os.getpid(), message)
    print(f"[steer][pid {os.getpid()}] {message}", flush=True)


def _decoder_layers(model: torch.nn.Module, n_layers: int) -> torch.nn.ModuleList:
    """Return the language model's decoder stack, or refuse to guess."""
    matches = [
        (name or "<root>", module)
        for name, module in model.named_modules()
        if isinstance(getattr(module, "layers", None), torch.nn.ModuleList)
        and hasattr(module, "embed_tokens")
        and len(module.layers) == n_layers
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {n_layers}-layer decoder stack, "
            f"found {[name for name, _ in matches]}"
        )
    name, stack = matches[0]

    start = getattr(stack, "start_layer", 0)
    end = getattr(stack, "end_layer", n_layers)
    if (start, end) != (0, n_layers):
        raise RuntimeError(
            f"this process holds pipeline shard {start}:{end}, so a global layer "
            "index is ambiguous"
        )

    layers = stack.layers
    indices = [getattr(layer, "layer_idx", None) for layer in layers]
    if all(i is not None for i in indices) and indices != list(range(n_layers)):
        raise RuntimeError(f"unexpected layer_idx ordering: {indices[:8]}")

    _log(f"decoder stack '{name}' with {n_layers} layers")
    return layers


def _delta(hidden: torch.Tensor) -> torch.Tensor:
    """`alpha_t * V[row_t]` for each of the batch's flattened tokens.

    A gather rather than a matmul against a per-token coefficient matrix. Both
    produce this, and the matmul is the tidier expression, but its cost grows
    with the size of the served set — a thousand-vector store would put a
    `[tokens, 1000] @ [1000, hidden]` product in the forward path of every step
    — while a gather costs the same whether the server holds four vectors or
    four hundred.
    """
    n = hidden.shape[0]
    rows = _st.vectors.index_select(0, _st.row[:n])
    # In place on the gather's own fresh tensor, so the served matrix is not
    # touched and the [tokens, hidden] intermediate is allocated once.
    rows.mul_(_st.alpha[:n].unsqueeze(1))
    return rows.to(hidden.dtype)


def _steer(module: torch.nn.Module, args: Any, output: Any) -> Any:
    """Add the steering delta to the target block's output residual stream.

    Keeps the signature of a `torch` forward hook, and is called with the same
    three arguments, so that what it does to the stream is independent of how it
    is reached: `_steered_class` invokes it, and it can still be registered as a
    hook where a compiled graph is not in play.
    """
    if not _st.ready:
        return None

    if isinstance(output, tuple) and len(output) == 2:
        hidden, residual = output
        if residual is None:
            return hidden + _delta(hidden), residual
        # vLLM carries the stream split as (hidden_states, residual) and the
        # next block recombines them in a fused add-RMSNorm, so the residual
        # stream this block actually produced is their sum. Fold it here and
        # hand the next block an empty delta half: it then normalises
        # 0 + residual, which is the steered stream and nothing else.
        return torch.zeros_like(hidden), (hidden + residual) + _delta(hidden)

    if isinstance(output, torch.Tensor):
        return output + _delta(output)

    raise RuntimeError(f"unsupported decoder layer output: {type(output)}")


def _allocate(runner: Any, served: store.Store, hidden_size: int) -> None:
    device = runner.device
    rows = store.matrix(served, hidden_size)
    max_tokens = int(runner.max_num_tokens)

    _st.vectors = torch.from_numpy(rows).to(device)
    _st.alpha = torch.zeros(max_tokens, dtype=torch.float32, device=device)
    # int64 because that is what index_select takes; one column of it against a
    # [max_tokens, hidden] delta is not a size worth economising on.
    _st.row = torch.zeros(max_tokens, dtype=torch.int64, device=device)
    _st.alpha_staging = torch.zeros(max_tokens, dtype=torch.float32).pin_memory()
    _st.row_staging = torch.zeros(max_tokens, dtype=torch.int64).pin_memory()
    _st.n_filled = 0
    _st.ready = True

    _log(
        f"serving {len(served.vectors)} vector(s) from {served.root} "
        f"digest={served.digest[:16]} max_num_tokens={max_tokens} device={device}"
    )
    for row, vector in enumerate(served.vectors, start=1):
        _log(f"  row {row}: {vector.describe()}")


_STEERED_CLASSES: dict[type, type] = {}


def _steered_class(base: type) -> type:
    """A subclass of the target block whose forward applies the steering.

    Same arithmetic, same `_steer` body: the only difference from the hook is
    that this runs inside the module's own forward, where torch.compile can see
    it and CUDA-graph capture records the kernels it launches.
    """
    cached = _STEERED_CLASSES.get(base)
    if cached is not None:
        return cached

    class SteeredDecoderBlock(base):  # type: ignore[misc,valid-type]
        def forward(self, *args: Any, **kwargs: Any) -> Any:
            output = super().forward(*args, **kwargs)
            replaced = _steer(self, args, output)
            return output if replaced is None else replaced

    SteeredDecoderBlock.__name__ = f"Steered{base.__name__}"
    SteeredDecoderBlock.__qualname__ = SteeredDecoderBlock.__name__
    _STEERED_CLASSES[base] = SteeredDecoderBlock
    return SteeredDecoderBlock


def _steering(runner: Any, request_id: str, config: SteerConfig) -> tuple[int, float]:
    """The row and alpha for one request: what it asked for, or nothing.

    Anything unusable — an id this server does not have, a strength that is not
    a number, one of the two without the other — resolves to row 0 at strength
    0, the unsteered model. It is not this function's job to refuse: it runs in
    a worker process, inside the forward path, where raising kills the engine
    and there is no response to write a status onto. The refusal happens in the
    API server process, before the request is admitted
    (`middleware.SteeringValidation`), which is the only place a client can be
    told why. What is here is the floor under that: a request the validator
    somehow let through gets the base model rather than a wrong vector, and
    :func:`_fill_steering` says so in the log.
    """
    state = runner.requests.get(request_id)
    params = getattr(state, "sampling_params", None) if state is not None else None
    extra = getattr(params, "extra_args", None) if params is not None else None
    if not extra:
        return 0, 0.0

    row = _st.store.row_of(extra.get(config.vector_arg))
    if row is None:
        return 0, 0.0
    try:
        strength = float(extra.get(config.arg, 0.0))
    except (TypeError, ValueError):
        return 0, 0.0
    if not np.isfinite(strength):
        return 0, 0.0
    return row, strength


def _fill_steering(runner: Any, num_scheduled_tokens: np.ndarray) -> None:
    if not _st.ready:
        return
    config = _st.config
    batch = runner.input_batch
    num_reqs = batch.num_reqs

    req_ids = [batch.req_ids[i] for i in range(num_reqs)]
    resolved = [_steering(runner, req_id, config) for req_id in req_ids]
    rows = np.fromiter((r for r, _ in resolved), dtype=np.int64, count=num_reqs)
    strengths = np.fromiter((a for _, a in resolved), dtype=np.float64, count=num_reqs)

    # num_scheduled_tokens is ordered by input-batch row, so repeating on it
    # reproduces exactly the flattening vLLM uses to lay out the token batch.
    per_token_row = np.repeat(rows, num_scheduled_tokens)
    per_token_alpha = np.repeat(strengths, num_scheduled_tokens)
    n = per_token_alpha.shape[0]

    _st.alpha_staging[:n] = torch.from_numpy(per_token_alpha.astype(np.float32))
    _st.row_staging[:n] = torch.from_numpy(per_token_row)
    _st.alpha[:n].copy_(_st.alpha_staging[:n], non_blocking=True)
    _st.row[:n].copy_(_st.row_staging[:n], non_blocking=True)
    if _st.n_filled > n:
        # Padding past the scheduled tokens must not carry a stale alpha. The
        # row buffer needs no clearing for the same reason: at alpha 0 the row
        # it names contributes nothing.
        _st.alpha[n : _st.n_filled].zero_()
    _st.n_filled = n

    _report_unhonoured(runner, req_ids, rows, config)

    _st.step += 1
    if config.debug:
        _log(
            f"step={_st.step} tokens={n} "
            f"rows={rows.tolist()} strengths={strengths.tolist()}"
        )


def _report_unhonoured(
    runner: Any, req_ids: list[str], rows: np.ndarray, config: SteerConfig
) -> None:
    """Log requests that asked to be steered and are not being.

    This should never fire: the validating middleware rejects exactly these
    requests with a 400 before the engine sees them. It exists because the two
    processes read the vectors directory separately, so "the API server knows an
    id the worker does not" is a state this design can reach — and because the
    middleware can be switched off by editing one line of `serve.sh`, which
    would otherwise turn every steered request into a quietly unsteered one.

    Warned once per request rather than once per step: `_fill_steering` runs
    every scheduler step, so a single bad request would otherwise write a line
    per token. The set is replaced rather than added to, so it holds only what
    is in flight and cannot grow.
    """
    offenders = set()
    for req_id, row in zip(req_ids, rows.tolist()):
        if row != 0:
            continue
        state = runner.requests.get(req_id)
        params = getattr(state, "sampling_params", None) if state is not None else None
        extra = getattr(params, "extra_args", None) if params is not None else None
        if extra and (config.vector_arg in extra or config.arg in extra):
            offenders.add(req_id)

    for req_id in sorted(offenders - _st.unhonoured):
        state = runner.requests.get(req_id)
        params = getattr(state, "sampling_params", None) if state is not None else None
        extra = getattr(params, "extra_args", None) if params is not None else None
        _log(
            f"request {req_id} asked for steering this server cannot honour and "
            f"is being served UNSTEERED: {config.vector_arg}="
            f"{(extra or {}).get(config.vector_arg)!r} {config.arg}="
            f"{(extra or {}).get(config.arg)!r}. Served ids: "
            f"{', '.join(_st.store.ids)}."
        )
    _st.unhonoured = frozenset(offenders)


def _patch_runner(config: SteerConfig, served: store.Store) -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original_load = GPUModelRunner.load_model
    if getattr(original_load, "_steer_patched", False):
        _log("already patched, skipping")
        return

    def load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_load(self, *args, **kwargs)
        text_config = self.model_config.hf_text_config
        n_layers = text_config.num_hidden_layers
        if not 0 <= served.block < n_layers:
            raise RuntimeError(
                f"the vectors in {served.root} are served at block "
                f"{served.block}, which is out of range for a "
                f"{n_layers}-layer model"
            )
        target = _decoder_layers(self.model, n_layers)[served.block]
        target.__class__ = _steered_class(type(target))
        _allocate(self, served, int(text_config.hidden_size))
        _log(
            f"steering the output of block {served.block} "
            f"({type(target).__name__})"
        )
        return result

    original_prepare = GPUModelRunner._prepare_inputs

    def prepare_inputs(
        self: Any, scheduler_output: Any, num_scheduled_tokens: np.ndarray
    ) -> Any:
        result = original_prepare(self, scheduler_output, num_scheduled_tokens)
        _fill_steering(self, num_scheduled_tokens)
        return result

    original_dummy = GPUModelRunner._dummy_run

    def dummy_run(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _st.ready and _st.n_filled:
            _st.alpha[: _st.n_filled].zero_()
            _st.row[: _st.n_filled].zero_()
            _st.n_filled = 0
        return original_dummy(self, *args, **kwargs)

    load_model._steer_patched = True
    GPUModelRunner.load_model = load_model
    GPUModelRunner._prepare_inputs = prepare_inputs
    GPUModelRunner._dummy_run = dummy_run


def apply() -> None:
    """Install the steering patches. Safe to call more than once."""
    config = config_from_env()
    served = store.read(config.vector_dir, config.vectors)
    _st.config = config
    _st.store = served
    _patch_runner(config, served)
    _log(
        f"installed: block={served.block} vectors={', '.join(served.ids)} "
        f"digest={served.digest[:16]} arg='{config.arg}' "
        f"vector_arg='{config.vector_arg}'"
    )
