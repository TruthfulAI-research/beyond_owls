"""`GET /steering/vectors`: what this server can be asked to steer with.

A client cannot address a vector it cannot name, and until this endpoint existed
the only way to know what a server held was to have started it. So the manifest
is served from the same directory the workers load their arrays from, and it
carries, per vector, everything a caller needs to choose one and to say
afterwards which one it used: the id, the name, what the two prompt sets
contrast, the layer, and the sha256 of the array.

Attached through vLLM's `vllm.endpoint_plugins` entry point, which is the
supported way to put a route on the OpenAI-compatible app — the alternative,
reaching into `vllm.entrypoints.openai.api_server` from `sitecustomize`, would
be a second private-module patch to keep working across releases. Endpoint
plugins are opt-in: vLLM loads one only when it is named in `VLLM_PLUGINS`, and
logs a warning and loads nothing when that variable is unset. `serve.sh` sets
it.

**This process is not the process that steers.** The API server runs separately
from the workers that hold the arrays, and this plugin deliberately does not try
to bridge that — the endpoint plugin contract says as much, and asking the
engine would mean opening an engine path for a question that is answerable from
disk. Both sides read the same directory, and `digest` is over the served set,
so the one thing the arrangement cannot rule out — the directory changing
between the two reads — is comparable between this payload and the workers'
startup log rather than invisible.
"""

from __future__ import annotations

from argparse import Namespace
from typing import Any

from fastapi import APIRouter, FastAPI
from starlette.datastructures import State

from . import config, store

#: Where the manifest is served. Not under `/v1`: that prefix is OpenAI's
#: namespace, and nothing here is part of that API.
ROUTE = "/steering/vectors"


def manifest(served: store.Store, cfg: config.SteerConfig) -> dict[str, Any]:
    """The payload, built from the store and nothing else.

    Every figure is a fact about a vector directory on disk, and the two `arg`
    names are what this server was configured to read off a request — so a
    client can construct a valid request out of this response alone, rather than
    out of this response plus an assumption about the field names.
    """
    return {
        "model": served.model,
        "block": served.block,
        "digest": served.digest,
        "arg": {
            "field": "vllm_xargs",
            "vector": cfg.vector_arg,
            "strength": cfg.arg,
        },
        "vectors": [
            {
                "id": vector.id,
                "name": vector.name,
                "description": vector.description,
                "layer": vector.layer,
                "block": vector.block,
                "scale": vector.scale,
                "vector_norm": vector.norm,
                "relative_per_unit": vector.relative_per_unit,
                "sha256": vector.sha256,
            }
            for vector in served.vectors
        ],
    }


class SteeringEndpointPlugin:
    """The `vllm.endpoint_plugins` entry point for this package."""

    name = "steering"

    #: No task requirement. The manifest describes a directory, so it is
    #: answerable on any server this patch is installed in — including one that
    #: is not generating, where knowing what it *would* steer with is still the
    #: question a client is asking.
    required_tasks = None

    def attach_router(self, app: FastAPI) -> None:
        """Read the store and register the route. Fatal if the store is bad.

        The store is read here rather than in `__init__` on purpose. vLLM wraps
        plugin *construction* in a try/except that logs the failure and carries
        on without the plugin, so a vectors directory that does not validate
        would produce a server that starts, serves, steers nothing a client can
        name, and says so only in a line of the startup log. `attach_router` is
        called without that guard, so a refusal here stops the server coming up
        — which is what a bad vectors directory should do.
        """
        if not config.enabled():
            return
        cfg = config.config_from_env()
        served = store.read(cfg.vector_dir, cfg.vectors)
        payload = manifest(served, cfg)

        router = APIRouter()

        @router.get(ROUTE)
        async def steering_vectors() -> dict[str, Any]:
            return payload

        app.include_router(router)
        print(
            f"[steer][api] {ROUTE} lists {len(served.vectors)} vector(s) "
            f"from {served.root}, digest={served.digest[:16]}",
            flush=True,
        )

    async def init_state(
        self, engine_client: Any, state: State, args: Namespace
    ) -> None:
        """Nothing to initialise: the manifest is a function of the filesystem.

        Part of the plugin contract, and empty on purpose rather than by
        oversight. The other half of the contract is where a plugin would reach
        the engine, and this one has no reason to: what it reports is what is on
        disk, which is the same thing the workers read, and asking the engine
        instead would make the endpoint unavailable exactly when the engine is
        unhealthy.
        """
        return None
