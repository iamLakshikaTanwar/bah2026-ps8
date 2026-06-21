"""Tiny on-disk memoisation layer built on :mod:`joblib`.

Expensive deterministic computations (horizon maps, sky-view factors, LUT
builds, Monte-Carlo volume runs) can be wrapped with :func:`cached` so repeated
runs read from ``data/cache`` instead of recomputing. This realises part of the
"fastest-platform" goal: O(1) re-access of previously computed artefacts.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from joblib import Memory

__all__ = ["memory", "cached", "CACHE_DIR", "clear_cache"]

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

#: Shared joblib Memory instance pointing at the on-disk cache directory.
memory = Memory(str(CACHE_DIR), verbose=0)

_F = TypeVar("_F", bound=Callable[..., Any])


def cached(func: _F) -> _F:
    """Decorator memoising ``func`` to the shared :data:`memory` cache.

    Thin wrapper over :meth:`joblib.Memory.cache` so call sites read
    ``@cached`` without importing joblib. Inputs must be joblib-hashable
    (numpy arrays are fine).

    Examples
    --------
    >>> from lunaris.io.cache import cached
    >>> @cached
    ... def expensive(x):
    ...     return x * x
    """
    return memory.cache(func)  # type: ignore[return-value]


def clear_cache() -> None:
    """Flush the entire on-disk cache (use between incompatible code changes)."""
    memory.clear(warn=False)
