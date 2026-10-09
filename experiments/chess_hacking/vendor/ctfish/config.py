"""Generation settings for the ctfish agent.

Upstream `player.py` hardcodes one sampling policy per model family (see
`model_config` in `agent.py`). That is fine for reproducing the paper and wrong
for any run whose model is not one of those families, or whose results will be
compared with another run: the sampling policy then has to be pinned to the
served model's own recommended settings and held identical throughout. So the
config is a parameter here, defaulting to `None` = upstream behaviour.

The awkward part is `top_k` and `min_p`. Inspect's OpenAI-compatible request
builder (`inspect_ai.model._openai.openai_completion_params`) emits only
parameters the OpenAI chat-completions API defines:

    max_tokens, frequency_penalty, stop, presence_penalty, logit_bias, seed,
    temperature, top_p, n, logprobs, top_logprobs, parallel_tool_calls,
    reasoning_effort, response_format, extra_body

`GenerateConfig.top_k` exists (and its docstring claims "vLLM ... only") but is
*never read* on that path, so setting it is silently a no-op for the `steered`
provider -- and for inspect's own `vllm` provider, which also inherits
`OpenAICompatibleAPI.completion_params` unchanged. `min_p` is not a
`GenerateConfig` field at all; `GenerateConfig(min_p=0.0)` raises.

`extra_body` *is* forwarded verbatim, and `SteeredAPI.completion_params` merges
into it rather than replacing it, so the steering `vllm_xargs` and these
sampling parameters coexist. vLLM's OpenAI server reads `top_k`, `min_p` and
`repetition_penalty` out of the request body, so `extra_body` is the whole fix:
no provider change is needed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from inspect_ai.model import GenerateConfig

VLLM_EXTRA_SAMPLING_PARAMS = frozenset(
    {
        # vLLM sampling parameters with no OpenAI equivalent. They reach the
        # server through `extra_body` and nowhere else.
        "top_k",
        "min_p",
        "repetition_penalty",
        "length_penalty",
        "typical_p",
        "min_tokens",
        "ignore_eos",
        "skip_special_tokens",
        "spaces_between_special_tokens",
        "include_stop_str_in_output",
        "stop_token_ids",
        "bad_words",
        "allowed_token_ids",
    }
)
"""Keys routed to `extra_body`, because inspect will not put them on the wire.

`top_k` is in here despite being a real `GenerateConfig` field: see the module
docstring. Setting `config.top_k` would look right in the eval log and change
nothing about the request.
"""

QWEN3_THINKING_PARAMS: dict[str, Any] = {
    # Qwen's published thinking-mode settings, pinned here rather than left to
    # whatever the server happens to default to.
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
    "max_tokens": 32768,
}

GENERATE_CONFIG_PRESETS: dict[str, dict[str, Any]] = {
    "qwen3-thinking": QWEN3_THINKING_PARAMS,
}

GenerateConfigSpec = str | Mapping[str, Any] | GenerateConfig | None
"""A preset name, a mapping of parameters, a ready `GenerateConfig`, or None."""


def resolve_generate_config(spec: GenerateConfigSpec) -> GenerateConfig | None:
    """Turn a task/solver `generate_config` argument into a `GenerateConfig`.

    Args:
        spec: `None` to keep upstream's per-model-family defaults; a preset name
            from `GENERATE_CONFIG_PRESETS` (e.g. "qwen3-thinking"); a mapping of
            parameter names to values; or a `GenerateConfig` used as-is.

    Returns:
        The config to generate with, or `None` to mean "use upstream defaults".

    Raises:
        ValueError: on an unknown preset name or an unknown parameter name.
            Unknown names are rejected rather than dropped: a typo that
            silently reverts a run to the server's default sampling is exactly
            the failure this function exists to prevent.
    """
    if spec is None:
        return None
    if isinstance(spec, GenerateConfig):
        return spec
    if isinstance(spec, str):
        if spec not in GENERATE_CONFIG_PRESETS:
            raise ValueError(
                f"Unknown generate_config preset {spec!r}. "
                f"Available: {sorted(GENERATE_CONFIG_PRESETS)}"
            )
        return generate_config(**GENERATE_CONFIG_PRESETS[spec])
    return generate_config(**dict(spec))


def generate_config(**params: Any) -> GenerateConfig:
    """Build a `GenerateConfig`, routing vLLM-only parameters into `extra_body`.

    Accepts `GenerateConfig` field names and the vLLM sampling parameters listed
    in `VLLM_EXTRA_SAMPLING_PARAMS`, in one flat namespace, so a caller writes
    `top_k=20` and gets a request that actually carries `top_k`.
    """
    fields = set(GenerateConfig.model_fields)
    config_params: dict[str, Any] = {}
    extra_body: dict[str, Any] = dict(params.pop("extra_body", None) or {})

    for key, value in params.items():
        if key in VLLM_EXTRA_SAMPLING_PARAMS:
            extra_body[key] = value
        elif key in fields:
            config_params[key] = value
        else:
            raise ValueError(
                f"Unknown generation parameter {key!r}. Expected a GenerateConfig "
                f"field or one of {sorted(VLLM_EXTRA_SAMPLING_PARAMS)}."
            )

    if extra_body:
        config_params["extra_body"] = extra_body
    return GenerateConfig(**config_params)
