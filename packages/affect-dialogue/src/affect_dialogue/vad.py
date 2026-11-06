"""Voice activity detection and endpointing, on 16 kHz mono float32 audio."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512  # 32 ms, the frame size Silero expects
FRAME_S = FRAME_SAMPLES / SAMPLE_RATE


class Vad(Protocol):
    def probability(self, frame: np.ndarray) -> float: ...
    def reset(self) -> None: ...


class EnergyVad:
    """No dependencies. Good enough for tests and a quiet room, not for a noisy lab."""

    def __init__(self, floor_rms: float = 0.01, full_rms: float = 0.04) -> None:
        self.floor_rms = floor_rms
        self.full_rms = full_rms

    def probability(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame))))
        return float(np.clip((rms - self.floor_rms) / (self.full_rms - self.floor_rms), 0.0, 1.0))

    def reset(self) -> None:
        pass


class SileroVad:
    """Silero through onnxruntime, using the model file faster-whisper ships."""

    _CONTEXT = 64

    def __init__(self) -> None:
        import faster_whisper
        import onnxruntime

        path = Path(faster_whisper.__file__).parent / "assets" / "silero_vad_v6.onnx"
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.log_severity_level = 4
        self._session = onnxruntime.InferenceSession(
            str(path), providers=["CPUExecutionProvider"], sess_options=options
        )
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(self._CONTEXT, dtype=np.float32)

    def probability(self, frame: np.ndarray) -> float:
        audio = np.concatenate([self._context, frame.astype(np.float32)])[None, :]
        output, self._h, self._c = self._session.run(
            None, {"input": audio, "h": self._h, "c": self._c}
        )
        self._context = frame[-self._CONTEXT :].astype(np.float32)
        return float(np.asarray(output).reshape(-1)[0])


class EchoGate:
    """Keeps the robot's own voice, heard through the microphone, from counting as a person.

    This is not echo cancellation. The robot knows what it is playing, so the gate learns
    how much of that reaches the microphone, and passes only frames clearly louder than
    the echo expected at that moment. A person close to the mic can still interrupt.
    With a headset almost nothing reaches the mic, so the gate does nothing. The first
    moments use a cautious guess, and the estimate carries over between turns.
    """

    FRAME = 0.032
    TAIL_S = 0.3  # the room keeps ringing, and the sound device adds delay
    MIN_REFERENCE = 1e-3

    def __init__(self, ratio: float = 2.0, initial_gain: float = 0.25) -> None:
        self.ratio = ratio
        self.initial_gain = initial_gain
        self._playback: deque[tuple[float, float]] = deque()
        self._ratios: deque[float] = deque(maxlen=300)

    def reference(self, samples: np.ndarray, rate: int, t: float) -> None:
        """Note what the speaker is about to play, starting at clock time t."""
        block = max(1, int(rate * self.FRAME))
        for i in range(0, len(samples), block):
            piece = samples[i : i + block]
            self._playback.append((t + i / rate, float(np.sqrt(np.mean(np.square(piece))))))

    @property
    def gain(self) -> float:
        """Median echo level. A person talking over part of the robot barely moves a median."""
        if len(self._ratios) < 20:
            return self.initial_gain
        return float(np.clip(np.median(self._ratios), 0.0, 1.0))

    def process(self, frame: np.ndarray, now: float) -> np.ndarray:
        while self._playback and self._playback[0][0] < now - self.TAIL_S:
            self._playback.popleft()
        levels = [level for t, level in self._playback if t <= now + self.FRAME]
        reference = max(levels, default=0.0)
        if reference < self.MIN_REFERENCE:
            return frame
        rms = float(np.sqrt(np.mean(np.square(frame))))
        passes = rms > self.ratio * self.gain * reference
        self._ratios.append(rms / reference)
        return frame if passes else np.zeros_like(frame)


@dataclass(frozen=True)
class SpeechStarted:
    t: float


@dataclass(frozen=True)
class SpeechEnded:
    t_last_voice: float  # true end of speech, before the endpointing silence
    t_detected: float
    audio: np.ndarray


class Endpointer:
    """Finds where speech starts and ends. Frames carry their capture time."""

    def __init__(
        self,
        vad: Vad,
        threshold: float = 0.5,
        start_s: float = 0.09,
        end_silence_s: float = 0.32,
        preroll_s: float = 0.2,
        max_utterance_s: float = 20.0,
    ) -> None:
        self.vad = vad
        self.threshold = threshold
        self.start_frames = max(1, round(start_s / FRAME_S))
        self.end_frames = max(1, round(end_silence_s / FRAME_S))
        self.max_frames = round(max_utterance_s / FRAME_S)
        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, round(preroll_s / FRAME_S)))
        self._speech: list[np.ndarray] = []
        self._voiced_run = 0
        self._silent_run = 0
        self._t_start = 0.0
        self._t_last_voice = 0.0
        self.in_speech = False

    def speech_duration(self, now: float) -> float:
        return now - self._t_start if self.in_speech else 0.0

    def feed(
        self, frame: np.ndarray, t: float, detect: np.ndarray | None = None
    ) -> list[SpeechStarted | SpeechEnded]:
        """`detect` is what the VAD sees, when it differs from the audio that gets transcribed:
        an echo gate must not cut the person's own words out of the transcript."""
        probe = frame if detect is None else detect
        voiced = self.vad.probability(probe) >= self.threshold
        events: list[SpeechStarted | SpeechEnded] = []
        if not self.in_speech:
            self._voiced_run = self._voiced_run + 1 if voiced else 0
            self._preroll.append(frame)
            if self._voiced_run >= self.start_frames:
                self.in_speech = True
                self._speech = list(self._preroll)
                self._silent_run = 0
                self._t_start = t
                self._t_last_voice = t
                events.append(SpeechStarted(t))
            return events
        self._speech.append(frame)
        if voiced:
            self._silent_run = 0
            self._t_last_voice = t
        else:
            self._silent_run += 1
        if self._silent_run >= self.end_frames or len(self._speech) >= self.max_frames:
            audio = np.concatenate(self._speech).astype(np.float32)
            events.append(SpeechEnded(self._t_last_voice, t, audio))
            self.in_speech = False
            self._speech = []
            self._voiced_run = 0
            self._preroll.clear()
            self.vad.reset()
        return events
