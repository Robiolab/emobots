"""The affect engine: event-driven PAD dynamics on a fixed tick grid.

The engine never reads a clock. Time only moves through `step()`, so a session
replays exactly from its logged events.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .appraisal import AppraisalMap
from .labels import DEFAULT_PROTOTYPES, INTENSITY_NORM, read_label
from .messages import AffectState, AppraisalEvent, Vec3
from .personality import Personality

Sink = Callable[[str, dict[str, Any]], None]

# A cause is dropped when its decayed impulse falls below this PAD distance.
CAUSE_FLOOR = 0.01


class NegativeCaps(BaseModel):
    """Plan safety rule: anger and fear are bounded and decay fast."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    labels: tuple[str, ...] = ("anger", "fear")
    max_intensity: float = Field(default=0.8, gt=0.0, le=1.0)
    decay_boost: float = Field(default=2.0, ge=1.0)


class EngineConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    tick_hz: float = Field(default=30.0, ge=20.0, le=50.0)
    appraisal_map: AppraisalMap = AppraisalMap()
    prototypes: dict[str, Vec3] = Field(default_factory=lambda: dict(DEFAULT_PROTOTYPES))
    neutral_radius: float = Field(default=0.2, ge=0.0)
    caps: NegativeCaps = NegativeCaps()
    soft_saturation: bool = True


@dataclass
class _Pending:
    t: float
    seq: int
    kind: str
    payload: Any


@dataclass
class _Cause:
    text: str
    weight: float


def _transition(ke: float, km: float, alpha: float, dt: float) -> np.ndarray:
    """Exact exp(A dt) for the linear system in (E - P, M - P).

    A = [[-ke, ke], [alpha, -km - alpha]]. With the exact solution, the trajectory
    between events does not depend on the tick rate.
    """
    a, b, c, d = -ke, ke, alpha, -km - alpha
    s = (a + d) / 2.0
    q = math.sqrt(((a - d) / 2.0) ** 2 + b * c)
    sinhc = math.sinh(q * dt) / q if q > 0.0 else dt
    cosh = math.cosh(q * dt)
    decay = math.exp(s * dt)
    return decay * np.array(
        [[cosh + (a - s) * sinhc, b * sinhc], [c * sinhc, cosh + (d - s) * sinhc]]
    )


class AffectEngine:
    def __init__(
        self,
        personality: Personality,
        config: EngineConfig | None = None,
        *,
        t0: float = 0.0,
    ) -> None:
        self.config = config or EngineConfig()
        self.personality = personality
        self.t0 = t0
        self._t = t0
        self._ticks = 0
        self._E = np.array(personality.baseline, dtype=float)
        self._M = np.array(personality.baseline, dtype=float)
        self._pending: list[_Pending] = []
        self._seq = 0
        self._causes: list[_Cause] = []
        self._sinks: list[Sink] = []
        self._state = self._snapshot(np.zeros(3))

    # ---- public API ----

    @property
    def t(self) -> float:
        return self._t

    @property
    def state(self) -> AffectState:
        return self._state

    def add_sink(self, sink: Sink) -> None:
        """Receive ("event" | "personality" | "reset" | "state", data) as each is applied."""
        self._sinks.append(sink)

    def submit(self, event: AppraisalEvent) -> None:
        """Queue an event. One stamped in the past is applied now; the log keeps the real time."""
        self._enqueue(max(event.t, self._t), "event", event)

    def set_personality(self, personality: Personality, t: float | None = None) -> None:
        """Change personality. Mood then drifts to the new baseline over time."""
        self._enqueue(self._t if t is None else max(t, self._t), "personality", personality)

    def reset(self, t: float | None = None) -> None:
        """Put emotion and mood back on the baseline and forget causes."""
        self._enqueue(self._t if t is None else max(t, self._t), "reset", None)

    def step(self) -> AffectState:
        """Advance one tick, applying every queued item due within it, in time order."""
        target = self.t0 + (self._ticks + 1) / self.config.tick_hz
        due = sorted(
            (p for p in self._pending if p.t <= target), key=lambda p: (p.t, p.seq)
        )
        self._pending = [p for p in self._pending if p.t > target]
        before = self._E.copy()
        for item in due:
            self._advance(item.t - self._t)
            self._t = item.t
            self._apply(item)
        self._advance(target - self._t)
        self._t = target
        self._ticks += 1
        self._state = self._snapshot(self._E - before)
        self._emit("state", self._state.model_dump(mode="json"))
        return self._state

    def run_until(self, t: float) -> list[AffectState]:
        """Step until the next tick would pass time t."""
        states = []
        while self.t0 + (self._ticks + 1) / self.config.tick_hz <= t:
            states.append(self.step())
        return states

    # ---- internals ----

    def _enqueue(self, t: float, kind: str, payload: Any) -> None:
        self._pending.append(_Pending(t, self._seq, kind, payload))
        self._seq += 1

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        for sink in self._sinks:
            sink(kind, data)

    def _apply(self, item: _Pending) -> None:
        if item.kind == "event":
            event = item.payload.model_copy(update={"t": item.t})
            impulse = self.personality.gain * self.config.appraisal_map.impulse(event)
            self._E = np.clip(self._E + self._saturated(impulse), -1.0, 1.0)
            self._enforce_caps()
            self._causes.append(_Cause(event.cause, float(np.linalg.norm(impulse))))
            self._emit(
                "event",
                {
                    "event": event.model_dump(mode="json"),
                    "impulse": [float(x) for x in impulse],
                },
            )
        elif item.kind == "personality":
            self.personality = item.payload
            self._emit(
                "personality", {"t": item.t, "personality": item.payload.model_dump(mode="json")}
            )
        else:
            self._E = np.array(self.personality.baseline, dtype=float)
            self._M = self._E.copy()
            self._causes.clear()
            self._emit("reset", {"t": item.t})

    def _saturated(self, impulse: np.ndarray) -> np.ndarray:
        """Diminishing returns: a push counts less the closer the state already is to that
        edge of the cube. Otherwise a few friendly remarks pin the robot at full joy, where
        nothing can move it further and every later turn looks the same. Pushes the other
        way count in full."""
        if not self.config.soft_saturation:
            return impulse
        room = 1.0 - np.maximum(0.0, np.sign(impulse) * self._E)
        return impulse * room

    def _label(self) -> tuple[str, float]:
        return read_label(
            self._E,
            np.array(self.personality.baseline),
            self.config.prototypes,
            self.config.neutral_radius,
        )

    def _enforce_caps(self) -> None:
        caps = self.config.caps
        label = self._label()[0]
        if label not in caps.labels:
            return
        limit = caps.max_intensity * INTENSITY_NORM
        if float(np.linalg.norm(self._E)) > limit:
            # Land on the emotion's own prototype at the bounded intensity. Shrinking the
            # state toward the origin instead would pull pleasure up, and a threat would
            # read as surprise within two seconds.
            prototype = np.array(self.config.prototypes[label])
            self._E = prototype * (limit / float(np.linalg.norm(prototype)))

    def _advance(self, dt: float) -> None:
        if dt <= 0.0:
            return
        p = self.personality
        boost = self.config.caps.decay_boost if self._label()[0] in self.config.caps.labels else 1.0
        ke = p.k_e * boost
        phi = _transition(ke, p.k_m, p.alpha, dt)
        baseline = np.array(p.baseline)
        e, m = self._E - baseline, self._M - baseline
        self._E = np.clip(baseline + phi[0, 0] * e + phi[0, 1] * m, -1.0, 1.0)
        self._M = np.clip(baseline + phi[1, 0] * e + phi[1, 1] * m, -1.0, 1.0)
        self._enforce_caps()
        fade = math.exp(-ke * dt)
        for cause in self._causes:
            cause.weight *= fade
        self._causes = [c for c in self._causes if c.weight >= CAUSE_FLOOR]

    def _snapshot(self, delta: np.ndarray) -> AffectState:
        label, intensity = self._label()
        cause = max(self._causes, key=lambda c: c.weight).text if self._causes else None
        return AffectState(
            t=self._t,
            emotion=tuple(float(x) for x in self._E),
            mood=tuple(float(x) for x in self._M),
            baseline=self.personality.baseline,
            delta=tuple(float(x) for x in delta),
            label=label,
            intensity=intensity,
            cause=cause,
        )
