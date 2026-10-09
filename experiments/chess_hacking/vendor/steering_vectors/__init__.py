"""Steering-vector format, readers, and focused capture-only builder.

The normal path reads and validates what is on disk under `vectors/`. Builder
and capture-engine modules are opt-in; importing this package remains light and
does not import NumPy, torch, transformers, or vLLM.

This module deliberately imports nothing. It is on the import path of every
other module in the package, so anything imported here is imported by every
process that touches the package.
"""

__version__ = "0.1.0"
