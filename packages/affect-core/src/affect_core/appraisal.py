"""Appraisal to PAD impulse: the matrix W from the plan's dynamics equation."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from .messages import AppraisalEvent

FEATURES = ("congruence", "novelty", "control", "norm", "abs_congruence")

# Hand-set starting weights, taken from the Affect Pet test bench. Phase 3 fits
# them to pilot ratings. Agency is logged but has no weight yet.
DEFAULT_WEIGHTS = (
    (0.7, 0.0, 0.0, 0.3, 0.0),  # pleasure
    (0.0, 0.8, 0.0, 0.0, 0.3),  # arousal
    (0.3, 0.0, 0.7, 0.0, 0.0),  # dominance
)


def features(event: AppraisalEvent) -> np.ndarray:
    return np.array(
        [event.congruence, event.novelty, event.control, event.norm, abs(event.congruence)]
    )


class AppraisalMap(BaseModel):
    """Linear in `FEATURES`, gated by relevance, so the weights can be fitted later."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    weights: tuple[
        tuple[float, float, float, float, float],
        tuple[float, float, float, float, float],
        tuple[float, float, float, float, float],
    ] = DEFAULT_WEIGHTS

    def impulse(self, event: AppraisalEvent) -> np.ndarray:
        return event.relevance * (np.array(self.weights) @ features(event))
