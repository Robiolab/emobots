"""Named events for scenarios and the dashboard, from the Affect Pet test bench."""

from __future__ import annotations

from .messages import AppraisalEvent


def _ev(cause, agency, relevance, congruence, novelty, control, norm) -> AppraisalEvent:
    return AppraisalEvent(
        cause=cause,
        agency=agency,
        relevance=relevance,
        congruence=congruence,
        novelty=novelty,
        control=control,
        norm=norm,
    )


CATALOG: dict[str, AppraisalEvent] = {
    "pet": _ev("you pet it", "user", 0.6, 0.6, -0.5, 0.2, 0.4),
    "praise": _ev("you praised it", "user", 0.8, 0.8, 0.2, 0.3, 0.6),
    "play": _ev("you played with it", "user", 0.7, 0.7, 0.7, 0.4, 0.3),
    "noise": _ev("a sudden loud noise", "world", 0.7, -0.1, 1.0, -0.3, 0.0),
    "ignore": _ev("you ignored it", "user", 0.4, -0.4, -0.6, -0.2, -0.2),
    "scold": _ev("you scolded it", "user", 0.8, -0.7, -0.2, -0.4, -0.5),
    "insult": _ev("an unfair insult", "user", 0.8, -0.6, 0.4, 0.6, -0.9),
    "threat": _ev("a threat", "user", 1.0, -0.8, 0.8, -0.9, -0.3),
}
