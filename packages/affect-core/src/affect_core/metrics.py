"""Affect stability from the evaluation plan: no label flip without a logged cause."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .messages import AffectState

# Mood further than this from baseline still carries the effect of logged events.
MOOD_RESIDUE = 0.01


@dataclass(frozen=True)
class LabelFlip:
    t: float
    before: str
    after: str
    cause: str | None
    explained: bool


def label_flips(states: Sequence[AffectState]) -> list[LabelFlip]:
    """List every label change.

    A flip is explained if a logged event is still the state's cause, or if mood
    is still displaced from baseline, since only logged events can displace it.
    """
    flips = []
    for prev, cur in zip(states, states[1:], strict=False):
        if prev.label == cur.label:
            continue
        residue = float(np.linalg.norm(np.array(cur.mood) - np.array(cur.baseline)))
        explained = cur.cause is not None or residue > MOOD_RESIDUE
        flips.append(LabelFlip(cur.t, prev.label, cur.label, cur.cause, explained))
    return flips
