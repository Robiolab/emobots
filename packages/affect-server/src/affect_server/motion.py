"""Play one recorded arm movement for the robot's emotion.

Movements are recorded by the separate `movements` project: one folder per recording,
`meta.json` (its first emotion is the type) and `thetas.csv` (t, theta1..theta4 in
degrees). The server picks one of that type at random and plays it on an arm, while the
host holds the emotion for the length of the movement.
"""

from __future__ import annotations

import csv
import json
import random
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

# Engine label -> type of recording. The engine has no "disgust" label, so fear, the other
# aversive withdrawal emotion, plays the disgust recordings.
DEFAULT_MOTION_MAP: dict[str, str] = {
    "joy": "happy",
    "sadness": "sad",
    "anger": "angry",
    "fear": "disgust",
}
MIN_INTENSITY = 0.25  # a faint emotion does not move the arm
DEFAULT_SPEED = 20  # the movements project tested playback at this speed on the real arm
DEFAULT_PORT = "/dev/ttyACM0"
REAL_SETTLE_S = 2.0  # time for a real arm to reach a recording's first pose


class Arm(Protocol):
    def send_thetas(self, thetas: Sequence[float], speed: int = 50) -> None: ...


@dataclass(frozen=True)
class Movement:
    kind: str
    name: str
    folder: Path
    duration: float
    samples: tuple[tuple[float, tuple[float, ...]], ...]


def parse_motion_map(text: str) -> dict[str, str]:
    """`joy=happy,anger=angry` -> {"joy": "happy", "anger": "angry"}."""
    pairs = (item.split("=", 1) for item in text.split(",") if item.strip())
    result = {label.strip(): kind.strip() for label, kind in pairs}
    if not result or not all(result.values()):
        raise ValueError(f"bad motion map {text!r}; expected label=type,label=type")
    return result


class MotionLibrary:
    """The recordings, grouped by their first emotion."""

    def __init__(self, root: str | Path, rng: random.Random | None = None) -> None:
        self.root = Path(root)
        self._rng = rng or random.Random()
        self._by_kind: dict[str, list[Path]] = {}
        for folder in sorted(self.root.iterdir()) if self.root.is_dir() else []:
            meta_file = folder / "meta.json"
            if not (meta_file.exists() and (folder / "thetas.csv").exists()):
                continue
            emotions = json.loads(meta_file.read_text()).get("emotions") or []
            if emotions:
                self._by_kind.setdefault(emotions[0], []).append(folder)

    def counts(self) -> dict[str, int]:
        return {kind: len(folders) for kind, folders in sorted(self._by_kind.items())}

    def pick(self, kind: str) -> Movement | None:
        folders = self._by_kind.get(kind)
        return self.load(self._rng.choice(folders), kind) if folders else None

    @staticmethod
    def load(folder: Path, kind: str) -> Movement:
        meta = json.loads((folder / "meta.json").read_text())
        with open(folder / "thetas.csv", newline="") as f:
            rows = list(csv.reader(f))[1:]
        samples = tuple((float(r[0]), tuple(float(x) for x in r[1:])) for r in rows)
        duration = samples[-1][0] if samples else 0.0
        return Movement(kind, meta.get("name", folder.name), folder, duration, samples)


class MotionPlayer:
    """Plays a movement on an arm in a thread, so the server's event loop never waits."""

    def __init__(
        self,
        library: MotionLibrary,
        arm: Arm,
        mapping: Mapping[str, str] | None = None,
        speed: int = DEFAULT_SPEED,
        min_intensity: float = MIN_INTENSITY,
        settle_s: float = 0.0,
    ) -> None:
        self.library = library
        self.arm = arm
        self.mapping = dict(DEFAULT_MOTION_MAP if mapping is None else mapping)
        self.speed = speed
        self.min_intensity = min_intensity
        self.settle_s = settle_s
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self.error: str | None = None

    @property
    def playing(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def choose(self, label: str, intensity: float) -> Movement | None:
        kind = self.mapping.get(label)
        if kind is None or intensity < self.min_intensity or self.playing:
            return None
        return self.library.pick(kind)

    def window_s(self, movement: Movement) -> float:
        """How long the arm is busy: reaching the first pose, then the recording itself."""
        return self.settle_s + movement.duration

    def start(self, movement: Movement, on_done: Callable[[], None] | None = None) -> None:
        self._cancel.clear()
        self.error = None
        self._thread = threading.Thread(
            target=self._run, args=(movement, on_done), name="arm-motion", daemon=True
        )
        self._thread.start()

    def _run(self, movement: Movement, on_done: Callable[[], None] | None) -> None:
        try:
            if not movement.samples:
                return
            # The first command moves the arm from wherever it is to the first pose, which
            # can be far away. Go there first and only then start the recording's clock.
            self.arm.send_thetas(movement.samples[0][1], self.speed)
            if self._cancel.wait(self.settle_s):
                return
            origin = time.monotonic()
            for t, thetas in movement.samples:
                wait = t - (time.monotonic() - origin)
                if wait > 0 and self._cancel.wait(wait):
                    return
                if self._cancel.is_set():
                    return
                self.arm.send_thetas(thetas, self.speed)
        except Exception as exc:  # a serial error must not take the server down
            self.error = str(exc)
        finally:
            if on_done is not None:
                on_done()

    def stop(self) -> None:
        """Abandon the movement in progress. The arm stays where it is."""
        self._cancel.set()


def create_arm(kind: str, port: str = DEFAULT_PORT) -> Any:
    """`mock` is a simulated arm; `real` opens the serial port of a myPalletizer."""
    try:
        from palletizer_movements.connector import MockConnector, PalletizerConnector
    except ImportError as exc:
        raise RuntimeError(
            "the movements project is not installed; run with "
            "`uv run --with-editable <path to movements> ...`"
        ) from exc
    if kind == "mock":
        return MockConnector().connect()
    if kind == "real":
        arm = PalletizerConnector(port)
        arm.connect()
        arm.lock()
        return arm
    raise ValueError(f"unknown arm {kind!r}; use mock or real")
