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
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

# Engine label -> type of recording. The engine has no "disgust" label, so fear, the other
# aversive withdrawal emotion, plays the disgust recordings.
ANY = "any"  # a recording type that means "one at random from every type"
DEFAULT_MOTION_MAP: dict[str, str] = {
    "joy": "happy",
    "sadness": "sad",
    "anger": "angry",
    "fear": "disgust",
    "interest": ANY,
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
        self._kind_of: dict[Path, str] = {}
        self._last: dict[str, Path] = {}  # per requested type, to avoid an immediate repeat
        for folder in sorted(self.root.iterdir()) if self.root.is_dir() else []:
            meta_file = folder / "meta.json"
            if not (meta_file.exists() and (folder / "thetas.csv").exists()):
                continue
            emotions = json.loads(meta_file.read_text()).get("emotions") or []
            if emotions:
                self._by_kind.setdefault(emotions[0], []).append(folder)
                self._kind_of[folder] = emotions[0]

    def counts(self) -> dict[str, int]:
        return {kind: len(folders) for kind, folders in sorted(self._by_kind.items())}

    def pick(self, kind: str) -> Movement | None:
        """A random recording of this type (or of any type for ANY), never the one this
        request got last time unless it is the only choice."""
        folders = list(self._kind_of) if kind == ANY else self._by_kind.get(kind)
        if not folders:
            return None
        fresh = [f for f in folders if f != self._last.get(kind)] or folders
        chosen = self._rng.choice(fresh)
        self._last[kind] = chosen
        return self.load(chosen, self._kind_of[chosen])

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


# myPalletizer 260 joint ranges in degrees (pymycobot.robot_info). The firmware silently
# drops any command with a joint outside its range, and arms taught by hand can be moved
# past it (theta2 below -2 in 26 of 41 recordings), so the arm would freeze mid-movement.
PALLETIZER_260_LIMITS = ((-162.0, 162.0), (-2.0, 90.0), (-92.0, 60.0), (-180.0, 180.0))
LIMIT_MARGIN = 0.2  # stay just inside the range: a value on the edge may be dropped too


class LimitedArm:
    """Keeps every command inside the arm's joint ranges, so the arm follows a movement as
    closely as it can instead of ignoring the samples that leave its range."""

    def __init__(
        self,
        arm: Any,
        limits: Sequence[tuple[float, float]] = PALLETIZER_260_LIMITS,
        margin: float = LIMIT_MARGIN,
    ) -> None:
        self.arm = arm
        self.limits = tuple(limits)
        self.margin = margin
        self.clamped = 0  # commands that had to be brought back into range

    def clamp(self, thetas: Sequence[float]) -> tuple[float, ...]:
        return tuple(
            min(max(float(t), low + self.margin), high - self.margin)
            for t, (low, high) in zip(thetas, self.limits, strict=True)
        )

    def send_thetas(self, thetas: Sequence[float], speed: int = DEFAULT_SPEED) -> None:
        safe = self.clamp(thetas)
        if safe != tuple(float(t) for t in thetas):
            self.clamped += 1
        self.arm.send_thetas(safe, speed)

    def get_thetas(self) -> Any:
        return self.arm.get_thetas()

    def lock(self) -> None:
        self.arm.lock()

    def disconnect(self) -> None:
        self.arm.disconnect()


class MultiArm:
    """Several arms driven as one: every command goes to each, so they move together.

    One arm failing (unplugged, a serial error) must not stop the others, so errors are
    kept per port and only raised when no arm took the command.
    """

    def __init__(self, arms: Mapping[str, Any]) -> None:
        if not arms:
            raise ValueError("MultiArm needs at least one arm")
        self.arms = dict(arms)
        self.errors: dict[str, str] = {}
        self.silent: list[str] = []  # ports that never answered, left out

    def send_thetas(self, thetas: Sequence[float], speed: int = DEFAULT_SPEED) -> None:
        delivered = 0
        for name, arm in self.arms.items():
            try:
                arm.send_thetas(thetas, speed)
                delivered += 1
            except Exception as exc:
                self.errors[name] = str(exc)
        if not delivered:
            raise RuntimeError(f"no arm took the command: {self.errors}")

    @property
    def clamped(self) -> int:
        return sum(getattr(arm, "clamped", 0) for arm in self.arms.values())

    def disconnect(self) -> None:
        for arm in self.arms.values():
            arm.disconnect()


def split_ports(port: str) -> list[str]:
    """`/dev/ttyACM0,/dev/ttyACM1` -> both. Duplicates are an error: two handles, one arm."""
    ports = [p.strip() for p in port.split(",") if p.strip()]
    if len(set(ports)) != len(ports):
        raise ValueError(f"the same port is listed twice in {port!r}")
    return ports


def _answers(arm: Any, attempts: int, pause: float) -> bool:
    """A powered-off or mismatched arm still opens its serial port, and sending to it never
    fails, so ask it where it is: a live arm answers with its joint angles."""
    for _ in range(attempts):
        if arm.get_thetas() is not None:
            return True
        time.sleep(pause)
    return False


def open_arms(
    ports: Sequence[str],
    factory: Callable[[str], Any],
    attempts: int = 4,
    pause: float = 0.5,
) -> tuple[dict[str, Any], list[str]]:
    """Open every port; keep the arms that answer. Returns (arms, ports that stayed silent)."""
    opened: dict[str, Any] = {}
    silent: list[str] = []
    try:
        for path in ports:
            arm = factory(path)
            arm.connect()
            arm.lock()
            if _answers(arm, attempts, pause):
                opened[path] = arm
            else:
                silent.append(path)
                arm.disconnect()
    except Exception as exc:
        for arm in opened.values():
            arm.disconnect()
        raise RuntimeError(f"could not open the arm on {path}: {exc}") from exc
    if not opened:
        raise RuntimeError(f"no arm answered on {', '.join(ports)}: is it powered and flashed?")
    return opened, silent


def create_arm(kind: str, port: str = DEFAULT_PORT) -> Any:
    """`mock` is a simulated arm; `real` opens each serial port (comma separated) of a
    myPalletizer, and drives all that answer together when there are several."""
    try:
        from palletizer_movements.connector import MockConnector, PalletizerConnector
    except ImportError as exc:
        raise RuntimeError(
            "the movements project is not installed; run with "
            "`uv run --with-editable <path to movements> ...`"
        ) from exc
    if kind == "mock":
        return MockConnector().connect()
    if kind != "real":
        raise ValueError(f"unknown arm {kind!r}; use mock or real")
    opened, silent = open_arms(split_ports(port), PalletizerConnector)
    opened = {path: LimitedArm(arm) for path, arm in opened.items()}
    for path in silent:
        print(f"warning: no arm answered on {path}; it will not move", file=sys.stderr, flush=True)
    print(f"arms ready: {', '.join(opened)}", file=sys.stderr, flush=True)
    arm = opened[next(iter(opened))] if len(opened) == 1 else MultiArm(opened)
    if isinstance(arm, MultiArm):
        arm.silent = silent
    return arm
