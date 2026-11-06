"""Turns affect state into commands for whatever channels a robot has."""

from .mapper import ExpressionLoop, ExpressionMapper, remap, remap_log
from .profile import BUILTIN_DIR, Channel, CueMotion, EmbodimentProfile, load_profile

__all__ = [
    "BUILTIN_DIR",
    "Channel",
    "CueMotion",
    "EmbodimentProfile",
    "ExpressionLoop",
    "ExpressionMapper",
    "load_profile",
    "remap",
    "remap_log",
]
