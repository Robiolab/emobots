"""Embodiment profiles: what a robot can express, and how PAD maps onto it."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

BUILTIN_DIR = Path(__file__).parent / "profiles"
Axis = Literal["pleasure", "arousal", "dominance"]


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Curve(_Strict):
    """A lookup curve on one PAD axis, for mappings a straight line cannot express."""

    axis: Axis
    points: list[tuple[float, float]] = Field(min_length=2)


class Idle(_Strict):
    """Background motion, so the robot never freezes. Arousal raises its size and speed."""

    amplitude: float = Field(ge=0.0)
    freq_hz: float = Field(gt=0.0)
    arousal_gain: float = 0.5


class Talking(_Strict):
    """Motion added while the robot is speaking."""

    amplitude: float = Field(ge=0.0)
    freq_hz: float = Field(gt=0.0)


class Channel(_Strict):
    limits: tuple[float, float]
    rest: float
    rate_limit: float = Field(gt=0.0, description="Largest change per second")
    pad: dict[Axis, float] = Field(default_factory=dict)
    labels: dict[str, float] = Field(
        default_factory=dict, description="Offset for a label, scaled by its intensity"
    )
    curve: Curve | None = None
    idle: Idle | None = None
    talking: Talking | None = None

    @model_validator(mode="after")
    def _rest_within_limits(self) -> Channel:
        low, high = self.limits
        if not low <= self.rest <= high:
            raise ValueError(f"rest {self.rest} is outside limits {self.limits}")
        return self


class CueMotion(_Strict):
    channel: str
    amplitude: float
    duration_s: float = Field(gt=0.0)
    cycles: int = Field(default=1, ge=1)


class EmbodimentProfile(_Strict):
    name: str
    ttl_s: float = Field(default=0.2, gt=0.0)
    channels: dict[str, Channel]
    cues: dict[str, list[CueMotion]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _cues_use_known_channels(self) -> EmbodimentProfile:
        for name, motions in self.cues.items():
            for motion in motions:
                if motion.channel not in self.channels:
                    raise ValueError(f"cue {name!r} uses unknown channel {motion.channel!r}")
        return self


def load_profile(name_or_path: str | Path) -> EmbodimentProfile:
    """Load a built-in profile by name, or any YAML file by path."""
    path = Path(name_or_path)
    if path.suffix not in (".yaml", ".yml"):
        path = BUILTIN_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no embodiment profile at {path}")
    return EmbodimentProfile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
