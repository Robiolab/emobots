"""ROS 2 wrapper. The bridge holds the logic and needs no ROS; `node` adds rclpy."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from affect_adapters import BaseAdapter, ChannelLimit, load_limits
from affect_core import (
    PRESETS,
    AffectEngine,
    AppraisalEvent,
    EngineConfig,
    ExpressionCommand,
    SessionLogger,
)
from affect_express import ExpressionLoop, ExpressionMapper, load_profile
from pydantic import ValidationError

Publish = Callable[[str], None]


class Ros2Adapter(BaseAdapter):
    """Publishes only limit-checked values, as JSON, for the robot's controller to read."""

    def __init__(self, limits: dict[str, ChannelLimit], publish: Publish) -> None:
        super().__init__(limits)
        self._publish = publish
        self._t_sent = 0.0

    def apply(self, command: ExpressionCommand):
        self._t_sent = command.t
        self._ttl = command.ttl_s
        return super().apply(command)

    def _send(self, values: dict[str, float], cues: list[str]) -> None:
        message = ExpressionCommand(
            t=self._t_sent, channels=values, cues=cues, ttl_s=getattr(self, "_ttl", 0.2)
        )
        self._publish(message.model_dump_json())


class AffectBridge:
    """Engine, mapper and adapter, wired to plain publish functions."""

    def __init__(
        self,
        publish_state: Publish,
        publish_command: Publish,
        personality: str = "pet",
        profile: str = "quadruped",
        limits: str | None = None,
        tick_hz: float = 30.0,
        log_dir: str | Path | None = None,
        warn: Callable[[str], None] = print,
    ) -> None:
        self.engine = AffectEngine(PRESETS[personality], EngineConfig(tick_hz=tick_hz))
        self.adapter = Ros2Adapter(load_limits(limits or profile), publish_command)
        self.mapper = ExpressionMapper(load_profile(profile))
        self.logger = None
        if log_dir is not None:
            self.logger = SessionLogger(Path(log_dir) / "ros-session.jsonl", self.engine)
        self._publish_state = publish_state
        self._warn = warn
        self.rejected = 0
        ExpressionLoop(self.engine, self.mapper, self.adapter.apply)

    def tick(self, now: float) -> None:
        """Advance the engine to `now`, in seconds since the start, and publish the state."""
        if self.engine.run_until(now):
            self._publish_state(self.engine.state.model_dump_json())
            if self.logger is not None:
                self.logger.flush()

    def on_appraisal(self, text: str) -> bool:
        """An appraisal from another node. Bad input is logged and never reaches the engine."""
        try:
            event = AppraisalEvent.model_validate(json.loads(text))
        except (ValueError, ValidationError) as exc:
            self.rejected += 1
            self._warn(f"rejected appraisal: {str(exc)[:200]}")
            return False
        # Stamped by the engine on arrival, so another node's clock cannot reorder a session.
        self.engine.submit(event.model_copy(update={"t": 0.0}))
        return True

    def on_cue(self, cue: str) -> bool:
        return self.mapper.trigger(cue.strip(), self.engine.t)

    def stop(self) -> None:
        self.adapter.stop()

    def resume(self) -> None:
        self.adapter.resume()

    def close(self) -> None:
        if self.logger is not None:
            self.logger.close()
