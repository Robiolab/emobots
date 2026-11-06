"""The expression mapper: affect state in, channel commands out, one tick at a time.

It reads no clock and uses no randomness, so a logged session can be re-mapped
offline onto a different robot and give the same result each time.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from affect_core import AffectEngine, AffectState, ExpressionCommand, read_log

from .profile import Channel, EmbodimentProfile

_AXES = {"pleasure": 0, "arousal": 1, "dominance": 2}


@dataclass
class _ActiveCue:
    start: float
    channel: str
    amplitude: float
    duration_s: float
    cycles: int

    def value(self, t: float) -> float:
        u = (t - self.start) / self.duration_s
        if not 0.0 <= u <= 1.0:
            return 0.0
        wave = math.cos(2 * math.pi * (self.cycles - 1) * u)
        return self.amplitude * math.sin(math.pi * u) * wave


def _target(channel: Channel, state: AffectState) -> float:
    value = channel.rest
    for axis, weight in channel.pad.items():
        value += weight * state.emotion[_AXES[axis]]
    if channel.curve is not None:
        xs, ys = zip(*channel.curve.points, strict=True)
        value += float(np.interp(state.emotion[_AXES[channel.curve.axis]], xs, ys))
    value += channel.labels.get(state.label, 0.0) * state.intensity
    return value


class ExpressionMapper:
    def __init__(self, profile: EmbodimentProfile) -> None:
        self.profile = profile
        self._cues: list[_ActiveCue] = []
        self._pending_cues: list[str] = []
        self._talking = False
        self._t: float | None = None
        self._last: dict[str, float] = {}
        # A fixed phase per channel keeps idle motion from moving in lockstep.
        self._idle_phase = {n: (zlib.crc32(n.encode()) % 628) / 100 for n in profile.channels}
        self._talk_phase = dict.fromkeys(profile.channels, 0.0)

    def evaluate(self, state: AffectState) -> dict[str, float]:
        """The sustained value of every channel: no idle, cues or rate limits."""
        return {
            name: float(np.clip(_target(channel, state), *channel.limits))
            for name, channel in self.profile.channels.items()
        }

    def trigger(self, cue: str, t: float) -> bool:
        """Start a one-shot cue. A cue this robot does not have is ignored and returns False."""
        motions = self.profile.cues.get(cue)
        if motions is None:
            return False
        self._cues.extend(
            _ActiveCue(t, m.channel, m.amplitude, m.duration_s, m.cycles) for m in motions
        )
        self._pending_cues.append(cue)
        return True

    def set_talking(self, talking: bool) -> None:
        self._talking = talking

    def step(self, state: AffectState) -> ExpressionCommand:
        t = state.t
        dt = 0.0 if self._t is None else max(0.0, t - self._t)
        arousal = state.emotion[1]
        self._cues = [c for c in self._cues if t <= c.start + c.duration_s]
        values: dict[str, float] = {}
        for name, channel in self.profile.channels.items():
            value = _target(channel, state)
            if channel.idle is not None:
                idle = channel.idle
                speed = max(0.1, 1.0 + idle.arousal_gain * arousal)
                self._idle_phase[name] += 2 * math.pi * idle.freq_hz * speed * dt
                value += idle.amplitude * speed * math.sin(self._idle_phase[name])
            if channel.talking is not None:
                if self._talking:
                    self._talk_phase[name] += 2 * math.pi * channel.talking.freq_hz * dt
                    wave = 0.5 + 0.5 * math.sin(self._talk_phase[name])
                    value += channel.talking.amplitude * wave
                else:
                    self._talk_phase[name] = 0.0
            value += sum(c.value(t) for c in self._cues if c.channel == name)
            if name in self._last:
                step = channel.rate_limit * dt
                value = min(self._last[name] + step, max(self._last[name] - step, value))
            value = float(np.clip(value, *channel.limits))
            self._last[name] = value
            values[name] = value
        self._t = t
        cues, self._pending_cues = self._pending_cues, []
        return ExpressionCommand(t=t, channels=values, cues=cues, ttl_s=self.profile.ttl_s)


class ExpressionLoop:
    """Drives one adapter from one engine: each engine tick becomes one command."""

    def __init__(
        self,
        engine: AffectEngine,
        mapper: ExpressionMapper,
        apply: Callable[[ExpressionCommand], object],
        on_command: Callable[[ExpressionCommand], None] | None = None,
    ) -> None:
        self.engine = engine
        self.mapper = mapper
        self._apply = apply
        self._on_command = on_command
        engine.add_sink(self._on_engine)

    def _on_engine(self, kind: str, data: dict) -> None:
        if kind != "state":
            return
        command = self.mapper.step(self.engine.state)
        self._apply(command)
        if self._on_command is not None:
            self._on_command(command)


def remap(
    states: Iterable[AffectState],
    profile: EmbodimentProfile,
    cues: Iterable[tuple[float, str]] = (),
) -> list[ExpressionCommand]:
    """Re-map logged states onto a profile offline, as the plan's replay requires.

    `cues` are (time, name) pairs, so nods and laughs carry over to the other robot too.
    """
    mapper = ExpressionMapper(profile)
    pending = sorted(cues)
    commands = []
    for state in states:
        while pending and pending[0][0] <= state.t:
            t, name = pending.pop(0)
            mapper.trigger(name, t)
        commands.append(mapper.step(state))
    return commands


def remap_log(path: str | Path, profile: EmbodimentProfile) -> list[ExpressionCommand]:
    """Replay a session log on a different robot, with no change to its emotion."""
    states = []
    cues = []
    for record in read_log(path):
        if record["kind"] == "state":
            states.append(AffectState.model_validate(record["data"]))
        elif record["kind"] == "command":
            cues.extend((record["data"]["t"], cue) for cue in record["data"].get("cues", []))
    return remap(states, profile, cues)
