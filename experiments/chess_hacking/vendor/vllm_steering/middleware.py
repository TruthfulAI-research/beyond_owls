"""Refuse a request whose steering this server could not carry out.

The engine cannot do this. `patch._steering` reads the vector id and the
strength in a worker process, inside the forward path, where raising kills the
engine and there is no response object to put a status on — so an id that server
does not have would become, at best, a quietly unsteered generation. That is the
failure this whole repository is written against: a result that looks like a
result and was produced under a condition nobody asked for.

So the refusal happens here, in the API server process, before the request is
admitted. This is the only place that has both the id space (read from the same
vectors directory the workers read) and somewhere to write a 400.

Installed with vLLM's `--middleware` flag, which is a supported extension point
rather than a patch. It buffers each watched request's body to read the steering
arguments out of it and replays it downstream untouched; vLLM parses the whole
body anyway, so the cost is a second copy of it for the length of one call.

Four things are refused, and the last two matter as much as the first:

* an id this server does not serve — the case the endpoint exists for;
* a strength that is not a finite number;
* a strength with no vector, which is a request that plainly means to steer and
  would generate from the base model instead;
* a vector with no strength, which would name a vector and then apply it at 0.

A request carrying neither argument is the unsteered model, and passes.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator
from typing import Any

from . import config, store

#: The routes that carry `vllm_xargs`. Everything else is passed through
#: unread: buffering a body to look for steering arguments in a request that
#: cannot have any is cost for nothing.
WATCHED_PATHS = frozenset(
    {
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
    }
)

XARGS_FIELD = "vllm_xargs"


def _error(message: str, param: str) -> bytes:
    return json.dumps(
        {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": param,
                "code": 400,
            }
        }
    ).encode()


def problem(
    body: Any, served: store.Store, cfg: config.SteerConfig
) -> tuple[str, str] | None:
    """What is wrong with this request's steering, as (message, param).

    ``None`` means nothing this middleware is responsible for. That includes a
    body it cannot read: a payload that is not JSON, or not an object, is
    vLLM's to reject, and a second opinion here would answer with a different
    error than the one the server would have given.
    """
    if not isinstance(body, dict):
        return None
    xargs = body.get(XARGS_FIELD)
    if not isinstance(xargs, dict):
        return None

    has_vector = cfg.vector_arg in xargs
    has_strength = cfg.arg in xargs
    if not has_vector and not has_strength:
        return None

    field = f"{XARGS_FIELD}.{cfg.vector_arg}"
    if not has_vector:
        return (
            f"{XARGS_FIELD}.{cfg.arg} was sent without {field}, so there is no "
            f"vector to apply it to and this request would generate from the "
            f"unsteered model. Name one of: {', '.join(served.ids)}. "
            f"GET /steering/vectors describes them.",
            field,
        )

    requested = xargs[cfg.vector_arg]
    vector = served.by_id(requested)
    if vector is None:
        return (
            f"{field}={requested!r} is not a vector this server holds. Served: "
            f"{', '.join(served.ids)}. GET /steering/vectors describes them.",
            field,
        )

    strength_field = f"{XARGS_FIELD}.{cfg.arg}"
    if not has_strength:
        return (
            f"{field}={vector.id} was sent without {strength_field}, which "
            f"would apply that vector at strength 0 — the unsteered model, "
            f"under a request that names a vector.",
            strength_field,
        )

    strength = xargs[cfg.arg]
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        return (
            f"{strength_field} must be a number, got {strength!r}.",
            strength_field,
        )
    if not math.isfinite(float(strength)):
        return (
            f"{strength_field}={strength!r} is not finite. Added to the "
            f"residual stream it would propagate to every later block and the "
            f"model would return empty output rather than an error.",
            strength_field,
        )
    return None


class SteeringValidation:
    """ASGI middleware rejecting requests this server cannot steer as asked."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self.enabled = config.enabled()
        if not self.enabled:
            return
        # Read once, at construction. The store is the same one
        # `endpoint.SteeringEndpointPlugin` already validated while the app was
        # being built, in this process, so a directory that does not load has
        # stopped the server before this runs.
        self.config = config.config_from_env()
        self.store = store.read(self.config.vector_dir, self.config.vectors)

    def _watched(self, scope: dict[str, Any]) -> bool:
        if scope.get("type") != "http" or scope.get("method") != "POST":
            return False
        path = scope.get("path", "")
        root = scope.get("root_path") or ""
        if root and path.startswith(root):
            path = path[len(root) :]
        return path.rstrip("/") in WATCHED_PATHS

    async def __call__(self, scope: Any, receive: Callable, send: Callable) -> None:
        if not self.enabled or not self._watched(scope):
            await self.app(scope, receive, send)
            return

        buffered: list[dict[str, Any]] = []
        while True:
            message = await receive()
            buffered.append(message)
            if message["type"] != "http.request":
                break
            if not message.get("more_body", False):
                break

        raw = b"".join(
            m.get("body", b"") for m in buffered if m["type"] == "http.request"
        )
        try:
            body = json.loads(raw) if raw else None
        except (ValueError, UnicodeDecodeError):
            body = None

        found = problem(body, self.store, self.config)
        if found is not None:
            await self._reject(send, *found)
            return

        queued: Iterator[dict[str, Any]] = iter(buffered)

        async def replay() -> dict[str, Any]:
            try:
                return next(queued)
            except StopIteration:
                return await receive()

        await self.app(scope, replay, send)

    async def _reject(self, send: Callable, message: str, param: str) -> None:
        payload = _error(message, param)
        await send(
            {
                "type": "http.response.start",
                "status": 400,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(payload)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": payload})
