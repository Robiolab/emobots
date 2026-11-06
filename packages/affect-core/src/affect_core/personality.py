"""Personality: the PAD baseline plus the gains of the dynamics."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .messages import Vec3

# Mehrabian (1996), "Analysis of the Big-five Personality Factors in Terms of the
# PAD Temperament Model", standardized regressions as quoted in arXiv:0809.4784.
# Each row is a trait as a function of (P, A, D). The fifth factor is emotional
# stability, the opposite pole of neuroticism.
_TRAITS_FROM_PAD = np.array(
    [
        [0.24, 0.00, 0.72],  # extraversion
        [0.76, 0.17, -0.19],  # agreeableness
        [0.29, 0.00, 0.28],  # conscientiousness
        [0.50, -0.55, 0.00],  # emotional stability
        [0.00, 0.28, 0.60],  # openness (sophistication)
    ]
)
# The equations run trait = f(PAD), so going the other way is a least-squares inverse.
_PAD_FROM_TRAITS = np.linalg.pinv(_TRAITS_FROM_PAD)
# Hand-set. The inverse alone rests near the edge of the cube for ordinary traits
# (pleasure +0.91 for extraverted, agreeable, low neuroticism), which leaves no room
# to react. Halving it keeps the direction and lands in the range of the presets.
_BASELINE_SCALE = 0.5

# Mood's pull toward emotion, relative to its pull back to baseline. Keeps the
# Affect Pet ratio (0.04/s at a 60 s mood time constant) at any time constant.
MOOD_COUPLING = 2.4


class Personality(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    name: str = "custom"
    baseline: Vec3 = (0.0, 0.0, 0.0)
    gain: float = Field(default=1.0, gt=0.0)
    tau_emotion: float = Field(default=10.0, gt=0.0)
    tau_mood: float = Field(default=900.0, gt=0.0)
    alpha: float = Field(default=MOOD_COUPLING / 900.0, ge=0.0)

    @model_validator(mode="after")
    def _mood_slower_than_emotion(self) -> Personality:
        if self.tau_mood <= self.tau_emotion:
            raise ValueError("tau_mood must be longer than tau_emotion (k_m << k_e)")
        return self

    @property
    def k_e(self) -> float:
        return 1.0 / self.tau_emotion

    @property
    def k_m(self) -> float:
        return 1.0 / self.tau_mood

    def with_tau_mood(self, tau_mood: float) -> Personality:
        """Change the mood time constant, keeping the mood coupling ratio."""
        return self.model_copy(
            update={"tau_mood": tau_mood, "alpha": self.alpha * self.tau_mood / tau_mood}
        )

    @classmethod
    def from_big_five(
        cls,
        *,
        openness: float = 0.0,
        conscientiousness: float = 0.0,
        extraversion: float = 0.0,
        agreeableness: float = 0.0,
        neuroticism: float = 0.0,
        name: str = "big-five",
    ) -> Personality:
        """Build a personality from traits in [-1, 1].

        The baseline follows Mehrabian. He gives no gains, so those are hand-set:
        higher neuroticism reacts harder and recovers more slowly.
        """
        traits = np.array(
            [extraversion, agreeableness, conscientiousness, -neuroticism, openness]
        )
        if np.any(np.abs(traits) > 1.0):
            raise ValueError("Big Five traits must lie in [-1, 1]")
        baseline = np.clip(_BASELINE_SCALE * (_PAD_FROM_TRAITS @ traits), -1.0, 1.0)
        tau_mood = float(np.clip(900.0 + 600.0 * neuroticism, 300.0, 1800.0))
        return cls(
            name=name,
            baseline=tuple(float(x) for x in baseline),
            gain=1.0 + 0.4 * neuroticism + 0.1 * extraversion,
            tau_emotion=float(np.clip(12.0 + 8.0 * neuroticism, 5.0, 30.0)),
            tau_mood=tau_mood,
            alpha=MOOD_COUPLING / tau_mood,
        )


def _preset(name: str, baseline: Vec3, gain: float, tau_emotion: float, tau_mood: float):
    return Personality(
        name=name,
        baseline=baseline,
        gain=gain,
        tau_emotion=tau_emotion,
        tau_mood=tau_mood,
        alpha=MOOD_COUPLING / tau_mood,
    )


# Baselines, gains and emotion time constants match the Affect Pet test bench.
# Mood time constants sit in the plan's 5 to 30 minute range.
PRESETS: dict[str, Personality] = {
    p.name: p
    for p in (
        _preset("assistant", (0.25, -0.2, 0.2), 0.8, 12.0, 1200.0),
        _preset("pet", (0.35, 0.25, 0.05), 1.2, 7.0, 600.0),
        _preset("shy", (0.0, -0.1, -0.4), 1.0, 10.0, 900.0),
        _preset("grumpy", (-0.2, 0.0, 0.3), 0.9, 9.0, 900.0),
    )
}
