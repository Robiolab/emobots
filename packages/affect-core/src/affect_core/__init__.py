"""Robot-agnostic affect core: appraisal to PAD, dynamics, personality, labels."""

from .appraisal import AppraisalMap
from .catalog import CATALOG
from .engine import AffectEngine, EngineConfig, NegativeCaps
from .messages import SCHEMA_VERSION, AffectState, AppraisalEvent, ExpressionCommand
from .metrics import LabelFlip, label_flips
from .personality import PRESETS, Personality
from .scenario import Scenario, check_expectations, run_scenario
from .session import ReplayResult, SessionLogger, read_log, replay

__all__ = [
    "CATALOG",
    "PRESETS",
    "SCHEMA_VERSION",
    "AffectEngine",
    "AffectState",
    "AppraisalEvent",
    "AppraisalMap",
    "EngineConfig",
    "ExpressionCommand",
    "LabelFlip",
    "NegativeCaps",
    "Personality",
    "ReplayResult",
    "Scenario",
    "SessionLogger",
    "check_expectations",
    "label_flips",
    "read_log",
    "replay",
    "run_scenario",
]
