"""Installs the steering patch into every vLLM process.

Python imports `sitecustomize` automatically at interpreter start-up for any
directory on `PYTHONPATH`, which is how the patch reaches the spawned engine and
worker processes. That indirection is the whole point: the model lives in a
different process from the API server, so patching the API server alone would
silently do nothing.

The hook is lazy — it fires when vLLM's GPU model runner is imported, after vLLM
exists but before the model is built — so no process pays to import torch at
start-up, and the module is a complete no-op unless STEER_ENABLE=1.
"""

import importlib.abc
import importlib.machinery
import os
import sys
from collections.abc import Sequence
from types import ModuleType

_TRIGGER = "vllm.v1.worker.gpu_model_runner"

if os.environ.get("STEER_ENABLE") == "1":

    class _SteerFinder(importlib.abc.MetaPathFinder):
        _armed = True

        def find_spec(
            self,
            fullname: str,
            path: Sequence[str] | None = None,
            target: ModuleType | None = None,
        ) -> importlib.machinery.ModuleSpec | None:
            if fullname != _TRIGGER or not self._armed:
                return None

            # Disarm before delegating, so the lookup below does not re-enter.
            self._armed = False
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                self._armed = True
                return None

            exec_module = spec.loader.exec_module

            def exec_and_patch(module: ModuleType) -> None:
                exec_module(module)
                from vllm_steering.patch import apply

                apply()

            spec.loader.exec_module = exec_and_patch  # type: ignore[method-assign]
            return spec

    sys.meta_path.insert(0, _SteerFinder())
