"""The three versioned messages every part of the engine speaks."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1

Unit = Annotated[float, Field(ge=-1.0, le=1.0)]
Vec3 = tuple[Unit, Unit, Unit]
Agency = Literal["self", "user", "world"]


class Message(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    v: Literal[1] = SCHEMA_VERSION


class AppraisalEvent(Message):
    """Something that happened, as judged against the robot's goals.

    Relevance is a gate from 0 to 1. The other scales are signed: novelty below
    zero means familiar, control below zero means the robot cannot cope.
    """

    t: float = 0.0
    source: str = "manual"
    cause: str
    relevance: float = Field(ge=0.0, le=1.0)
    congruence: Unit
    novelty: Unit
    agency: Agency = "world"
    control: Unit
    norm: Unit


class AffectState(Message):
    """Published every tick. Emotion, mood and baseline are PAD points."""

    t: float
    emotion: Vec3
    mood: Vec3
    baseline: Vec3
    delta: tuple[float, float, float]
    label: str
    intensity: float = Field(ge=0.0, le=1.0)
    cause: str | None = None


class ExpressionCommand(Message):
    """Sent to a robot adapter, which may clamp or refuse it."""

    t: float
    channels: dict[str, float]
    cues: list[str] = Field(default_factory=list)
    ttl_s: float = Field(gt=0.0)
