"""JSONL session logs and exact replay."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from .engine import AffectEngine, EngineConfig
from .messages import AffectState, AppraisalEvent
from .personality import Personality

LOG_VERSION = 1


class SessionLogger:
    """Writes one record per line: the session header, then everything the engine applies."""

    def __init__(self, path: str | Path, engine: AffectEngine, session: str | None = None) -> None:
        self.session = session or uuid.uuid4().hex[:12]
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO | None = self.path.open("w", encoding="utf-8", newline="\n")
        self._seq = 0
        self._write(
            "session",
            {
                "t0": engine.t0,
                "personality": engine.personality.model_dump(mode="json"),
                "config": engine.config.model_dump(mode="json"),
            },
        )
        engine.add_sink(self._write)

    def _write(self, kind: str, data: dict[str, Any]) -> None:
        if self._file is None:
            return
        record = {"v": LOG_VERSION, "session": self.session, "seq": self._seq, "kind": kind,
                  "data": data}
        self._file.write(json.dumps(record, allow_nan=False) + "\n")
        self._seq += 1

    def write(self, kind: str, data: dict[str, Any]) -> None:
        """Add another layer's record, such as a command or a turn. Replay skips these."""
        self._write(kind, data)

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> SessionLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_log(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield records. A cut-off last line (a crash) is dropped; any other bad line raises."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise


@dataclass
class ReplayResult:
    states: list[AffectState] = field(default_factory=list)
    compared: int = 0
    first_mismatch: int | None = None
    mismatches: int = 0

    @property
    def exact(self) -> bool:
        return self.compared > 0 and self.mismatches == 0


def replay(
    path: str | Path,
    personality: Personality | None = None,
    config: EngineConfig | None = None,
) -> ReplayResult:
    """Re-run a session from its logged inputs and compare each state to the log, bit for bit.

    Passing a different personality or config gives the counterfactual run. Those
    states are expected to differ, and personality changes logged during the
    session are skipped so the override holds for the whole run.
    """
    records = read_log(path)
    header = next(records)
    if header["kind"] != "session" or header["v"] != LOG_VERSION:
        raise ValueError(f"{path}: not a v{LOG_VERSION} session log")
    engine = AffectEngine(
        personality or Personality.model_validate(header["data"]["personality"]),
        config or EngineConfig.model_validate(header["data"]["config"]),
        t0=header["data"]["t0"],
    )
    result = ReplayResult()
    for record in records:
        kind, data = record["kind"], record["data"]
        if kind == "event":
            engine.submit(AppraisalEvent.model_validate(data["event"]))
        elif kind == "personality":
            if personality is None:
                engine.set_personality(Personality.model_validate(data["personality"]), t=data["t"])
        elif kind == "reset":
            engine.reset(t=data["t"])
        elif kind == "state":
            state = engine.step()
            result.states.append(state)
            if state.model_dump(mode="json") != data:
                if result.first_mismatch is None:
                    result.first_mismatch = result.compared
                result.mismatches += 1
            result.compared += 1
    return result
