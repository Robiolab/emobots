"""Backend selection through entry points, and a small engine clock for standalone runs."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any

from affect_core import AffectEngine


def load_backend(group: str, name: str, **kwargs: Any) -> Any:
    """Load a plugin by name, so a lab can add a backend without forking."""
    matches = entry_points(group=f"affect.{group}")
    for entry in matches:
        if entry.name == name:
            return entry.load()(**kwargs)
    known = ", ".join(sorted(e.name for e in matches)) or "none"
    raise ValueError(f"unknown {group} backend {name!r}. Installed: {known}")


async def run_engine(
    engine: AffectEngine,
    clock: Callable[[], float] = time.monotonic,
    start: float | None = None,
) -> None:
    """Keep an engine on wall-clock time until cancelled."""
    origin = clock() if start is None else start
    interval = 1.0 / engine.config.tick_hz
    while True:
        engine.run_until(clock() - origin)
        await asyncio.sleep(interval)
