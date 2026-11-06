"""Robot adapters: capabilities(), apply(command), state() and stop().

The engine only requests. Each adapter holds its robot's limits and clamps or refuses
what it receives, so no expression can override a safety limit. A hardware e-stop
bypasses all of this.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import yaml
from affect_core import ExpressionCommand
from affect_express import EmbodimentProfile
from pydantic import BaseModel, ConfigDict, model_validator

LIMITS_DIR = Path(__file__).parent / "limits"


class ChannelLimit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    min: float
    max: float
    rate_limit: float
    rest: float

    @model_validator(mode="after")
    def _ordered(self) -> ChannelLimit:
        if not self.min <= self.rest <= self.max:
            raise ValueError("limits must satisfy min <= rest <= max")
        if self.rate_limit <= 0:
            raise ValueError("rate_limit must be positive")
        return self


@dataclass
class ApplyResult:
    applied: dict[str, float] = field(default_factory=dict)
    clamped: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    rejected: str | None = None


class Adapter(Protocol):
    def capabilities(self) -> dict[str, ChannelLimit]: ...
    def apply(self, command: ExpressionCommand) -> ApplyResult: ...
    def state(self, now: float | None = None) -> dict[str, float]: ...
    def stop(self) -> None: ...


def load_limits(name_or_path: str | Path) -> dict[str, ChannelLimit]:
    path = Path(name_or_path)
    if path.suffix not in (".yaml", ".yml"):
        path = LIMITS_DIR / f"{name_or_path}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {name: ChannelLimit.model_validate(v) for name, v in data["channels"].items()}


class BaseAdapter:
    """Enforces limits, then hands the safe values to `_send`. Subclass for a real robot."""

    def __init__(self, limits: dict[str, ChannelLimit]) -> None:
        self._limits = limits
        self._values = {name: limit.rest for name, limit in limits.items()}
        self._t: float | None = None
        self._expires: float | None = None
        self._stopped = False

    def capabilities(self) -> dict[str, ChannelLimit]:
        return dict(self._limits)

    def apply(self, command: ExpressionCommand) -> ApplyResult:
        if self._stopped:
            return ApplyResult(rejected="stopped")
        result = ApplyResult()
        dt = 0.0 if self._t is None else max(0.0, command.t - self._t)
        first = self._t is None
        for name, requested in command.channels.items():
            limit = self._limits.get(name)
            if limit is None:
                result.dropped.append(name)
                continue
            value = min(limit.max, max(limit.min, requested))
            current = self._values[name]
            # The first command too must move from rest at a safe speed, so allow one tick.
            step = limit.rate_limit * (dt if not first else command.ttl_s)
            value = min(current + step, max(current - step, value))
            if value != requested:
                result.clamped.append(name)
            self._values[name] = value
            result.applied[name] = value
        self._t = command.t
        self._expires = command.t + command.ttl_s
        self._send(result.applied, command.cues)
        return result

    def state(self, now: float | None = None) -> dict[str, float]:
        """Current values. With `now`, a command past its ttl reads as rest."""
        if now is not None and (self._expires is None or now > self._expires):
            return {name: limit.rest for name, limit in self._limits.items()}
        return dict(self._values)

    def stop(self) -> None:
        """Refuse every later command until `resume`, and return to rest."""
        self._stopped = True
        self._values = {name: limit.rest for name, limit in self._limits.items()}
        self._send(self._values, [])

    def resume(self) -> None:
        self._stopped = False
        self._t = None

    @property
    def stopped(self) -> bool:
        return self._stopped

    def _send(self, values: dict[str, float], cues: list[str]) -> None:
        raise NotImplementedError


class SimulatorAdapter(BaseAdapter):
    """Records what a robot would have received, for development without hardware."""

    def __init__(self, limits: dict[str, ChannelLimit], keep: int = 100_000) -> None:
        super().__init__(limits)
        self.history: list[dict[str, float]] = []
        self.cues: list[str] = []
        self._keep = keep

    @classmethod
    def for_robot(cls, name: str) -> SimulatorAdapter:
        return cls(load_limits(name))

    def _send(self, values: dict[str, float], cues: list[str]) -> None:
        self.history.append(dict(values))
        if len(self.history) > self._keep:
            del self.history[: self._keep // 2]
        self.cues.extend(cues)


class AvatarAdapter(BaseAdapter):
    """An on-screen face. It has no physical limits, so the profile's own apply."""

    def __init__(self, limits: dict[str, ChannelLimit]) -> None:
        super().__init__(limits)
        self.last_cues: list[str] = []

    @classmethod
    def from_profile(cls, profile: EmbodimentProfile) -> AvatarAdapter:
        return cls(
            {
                name: ChannelLimit(
                    min=ch.limits[0], max=ch.limits[1], rate_limit=ch.rate_limit, rest=ch.rest
                )
                for name, ch in profile.channels.items()
            }
        )

    def _send(self, values: dict[str, float], cues: list[str]) -> None:
        if cues:
            self.last_cues = list(cues)


class SdkAdapter(BaseAdapter):
    """For any robot with its own Python API: pass a function that sends safe values to it."""

    def __init__(
        self,
        limits: dict[str, ChannelLimit],
        send: Callable[[dict[str, float], list[str]], None],
    ) -> None:
        super().__init__(limits)
        self._send_fn = send

    def _send(self, values: dict[str, float], cues: list[str]) -> None:
        self._send_fn(values, cues)


__all__ = [
    "Adapter",
    "ApplyResult",
    "AvatarAdapter",
    "BaseAdapter",
    "ChannelLimit",
    "SdkAdapter",
    "SimulatorAdapter",
    "load_limits",
]
