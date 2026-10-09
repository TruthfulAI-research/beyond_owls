"""Inspect model provider for a vLLM server with per-request activation steering.

Models are addressed as ``steered/<served-model-name>`` and both halves of the
condition are model args::

    inspect eval task.py --model steered/qwen -M steer_vector=0007 -M steer_strength=1.0

which travel to the server as per-request vLLM extra args::

    {"vllm_xargs": {"steer_vector": "0007", "steer_strength": 1.0}}

The vectors themselves live server side, one directory each, and the server
lists what it holds at ``GET /steering/vectors``. The client sends an id and a
scalar, so a sweep over vectors and a sweep over strengths are the same kind of
loop and both are recorded in the eval log.

The two arguments go together: the server rejects one without the other with a
400, because a strength with no vector is a request that means to steer and
would generate from the base model instead. That pairing is checked here too, at
construction, so a mistyped sweep fails before the first sample rather than on
every one of them.

Strength is in the unit the vectors' metadata calls the relative perturbation:
``1.0`` adds a delta the size of a typical residual-stream row at the vector's
layer, whichever vector was named. A request with neither arg is unsteered.
"""

from __future__ import annotations

import os
from typing import Any

from inspect_ai.model import GenerateConfig, modelapi
from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from typing_extensions import override

from steering_vectors import vectorfmt

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"

XARGS_FIELD = "vllm_xargs"
VECTOR_ARG = "steer_vector"
STRENGTH_ARG = "steer_strength"
TEMPLATE_KWARGS_FIELD = "chat_template_kwargs"


def _as_bool(value: bool | str) -> bool:
    """Coerce a CLI-supplied (``-M key=value``) or Python value to a bool."""
    if isinstance(value, bool):
        return value
    low = str(value).strip().lower()
    if low in ("true", "1", "yes", "on"):
        return True
    if low in ("false", "0", "no", "off"):
        return False
    raise ValueError(f"cannot interpret {value!r} as a boolean")


class SteeredAPI(OpenAICompatibleAPI):
    """OpenAI-compatible provider carrying a per-request vector and strength."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        steer_vector: str | int | None = None,
        steer_strength: float | str | None = None,
        enable_thinking: bool | str | None = None,
        **model_args: Any,
    ) -> None:
        # All three are consumed here rather than forwarded: whatever remains in
        # model_args reaches AsyncOpenAI(), which raises on unexpected keywords.
        if steer_vector is None:
            self.steer_vector: str | None = None
        else:
            # Normalised to the four-digit form the server addresses vectors by,
            # so `-M steer_vector=7` and `-M steer_vector=0007` are one
            # condition in the eval log rather than two.
            self.steer_vector = vectorfmt.vector_id(steer_vector)

        if steer_strength is None:
            self.steer_strength: float | None = None
        else:
            try:
                self.steer_strength = float(steer_strength)
            except (TypeError, ValueError) as ex:
                raise ValueError(
                    f"steer_strength must be a number, got {steer_strength!r}"
                ) from ex

        if (self.steer_vector is None) != (self.steer_strength is None):
            raise ValueError(
                f"steer_vector and steer_strength go together: got "
                f"steer_vector={self.steer_vector!r}, "
                f"steer_strength={self.steer_strength!r}. A strength with no "
                f"vector has nothing to apply and a vector with no strength "
                f"applies it at 0; either way the eval would run against the "
                f"unsteered model under a name that says otherwise. Pass both, "
                f"or neither for the base model."
            )

        self.enable_thinking = (
            None if enable_thinking is None else _as_bool(enable_thinking)
        )

        super().__init__(
            model_name=model_name,
            base_url=base_url,
            # The server does not authenticate, but AsyncOpenAI requires a key.
            api_key=api_key or os.environ.get("STEERED_API_KEY") or "EMPTY",
            config=config,
            service="steered",
            service_base_url=DEFAULT_BASE_URL,
            api_key_var="STEERED_API_KEY",
            **model_args,
        )

    @override
    def completion_params(self, config: GenerateConfig, tools: bool) -> dict[str, Any]:
        params = super().completion_params(config, tools)
        if self.steer_vector is None and self.enable_thinking is None:
            return params

        # Merge rather than replace: the base class may already have populated
        # extra_body from config.extra_body and prompt_logprobs.
        extra_body: dict[str, Any] = dict(params.get("extra_body") or {})

        if self.steer_vector is not None:
            xargs: dict[str, Any] = dict(extra_body.get(XARGS_FIELD) or {})
            xargs[VECTOR_ARG] = self.steer_vector
            xargs[STRENGTH_ARG] = self.steer_strength
            extra_body[XARGS_FIELD] = xargs

        if self.enable_thinking is not None:
            template_kwargs: dict[str, Any] = dict(
                extra_body.get(TEMPLATE_KWARGS_FIELD) or {}
            )
            template_kwargs["enable_thinking"] = self.enable_thinking
            extra_body[TEMPLATE_KWARGS_FIELD] = template_kwargs

        params["extra_body"] = extra_body
        return params

    @override
    def connection_key(self) -> str:
        # Scope adaptive concurrency per condition. The base class keys on
        # (api_key, model), which is identical across a sweep, so one pool would
        # be tuned by whichever condition happens to run fastest. The vector is
        # in the key as well as the strength: two vectors at one strength are
        # two different interventions and need not generate at the same rate.
        return f"steered:{self.model_name}:{self.steer_vector}:{self.steer_strength}"


@modelapi(name="steered")
def steered() -> type[SteeredAPI]:
    """Register the `steered` provider."""
    return SteeredAPI
