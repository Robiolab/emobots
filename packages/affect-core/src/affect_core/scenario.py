"""Scripted event sequences with expected read-outs, stored as JSON."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .catalog import CATALOG
from .engine import AffectEngine, EngineConfig
from .messages import AffectState, AppraisalEvent
from .personality import PRESETS, Personality
from .session import SessionLogger


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class ScriptedEvent(_Strict):
    t: float = Field(ge=0.0)
    event: str


class Expectation(_Strict):
    t: float = Field(ge=0.0)
    label: str | None = None
    cause: str | None = None


class Scenario(_Strict):
    name: str
    description: str = ""
    personality: str = "pet"
    tick_hz: float = 30.0
    tau_mood: float | None = None
    duration_s: float = Field(gt=0.0)
    events: list[ScriptedEvent] = Field(default_factory=list)
    expect: list[Expectation] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> Scenario:
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    def build_personality(self) -> Personality:
        personality = PRESETS[self.personality]
        return personality if self.tau_mood is None else personality.with_tau_mood(self.tau_mood)


def run_scenario(scenario: Scenario, log_path: str | Path | None = None) -> list[AffectState]:
    engine = AffectEngine(scenario.build_personality(), EngineConfig(tick_hz=scenario.tick_hz))
    logger = SessionLogger(log_path, engine, session=scenario.name) if log_path else None
    try:
        for scripted in scenario.events:
            template: AppraisalEvent = CATALOG[scripted.event]
            engine.submit(template.model_copy(update={"t": scripted.t, "source": "scenario"}))
        return engine.run_until(scenario.duration_s)
    finally:
        if logger:
            logger.close()


def check_expectations(scenario: Scenario, states: list[AffectState]) -> list[str]:
    """Return a readable failure for each expectation the run missed."""
    failures = []
    for want in scenario.expect:
        state = min(states, key=lambda s: abs(s.t - want.t))
        if want.label is not None and state.label != want.label:
            failures.append(f"t={want.t}: label {state.label!r}, wanted {want.label!r}")
        if want.cause is not None and state.cause != want.cause:
            failures.append(f"t={want.t}: cause {state.cause!r}, wanted {want.cause!r}")
    return failures
