"""Discrete labels: a read-out of the PAD point, never part of the state."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .messages import Vec3

NEUTRAL = "neutral"

# From the Affect Pet test bench, plus "interest", which the plan names and the
# bench lacks. All hand-set; Phase 3 checks them against rater labels.
DEFAULT_PROTOTYPES: dict[str, Vec3] = {
    "joy": (0.76, 0.48, 0.35),
    "calm": (0.55, -0.45, 0.25),
    "interest": (0.35, 0.30, 0.10),
    "surprise": (0.25, 0.80, -0.15),
    "anger": (-0.55, 0.60, 0.35),
    "fear": (-0.60, 0.60, -0.45),
    "sadness": (-0.60, -0.35, -0.35),
    "sleepy": (0.05, -0.80, -0.05),
}

# PAD norm that maps to full intensity.
INTENSITY_NORM = 1.1


def read_label(
    emotion: np.ndarray,
    baseline: np.ndarray,
    prototypes: Mapping[str, Vec3],
    neutral_radius: float,
) -> tuple[str, float]:
    """Return (label, intensity).

    At rest the robot is neutral for its own personality, wherever the baseline
    sits: a grumpy robot at rest is not "angry". That is a change from the test
    bench, which measured neutral only around the origin.
    """
    norm = float(np.linalg.norm(emotion))
    if norm < neutral_radius or float(np.linalg.norm(emotion - baseline)) < neutral_radius:
        return NEUTRAL, 0.0
    best = min(prototypes, key=lambda k: float(np.linalg.norm(emotion - np.array(prototypes[k]))))
    return best, min(1.0, norm / INTENSITY_NORM)
