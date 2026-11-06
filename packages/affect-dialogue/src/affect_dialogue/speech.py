"""Local speech: faster-whisper, Piper, and the sound device. Audio stays on this machine."""

from __future__ import annotations

import asyncio
import glob
import os
import re
import site
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

STT_LANGUAGES = ("es", "en")
DEFAULT_VOICES = {"en": "en_US-lessac-medium", "es": "es_MX-claude-high"}
DEFAULT_VOICE_DIR = Path("models/piper")


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str


@dataclass(frozen=True)
class Prosody:
    rate: float = 1.0
    pitch_semitones: float = 0.0
    energy: float = 1.0
    pause_s: float = 0.15

    @classmethod
    def from_channels(cls, channels: dict[str, float]) -> Prosody:
        return cls(
            rate=channels.get("voice.rate", 1.0),
            pitch_semitones=channels.get("voice.pitch", 0.0),
            energy=channels.get("voice.energy", 1.0),
            pause_s=channels.get("voice.pause", 0.15),
        )


class STT(Protocol):
    def transcribe(self, audio: np.ndarray, prior_language: str | None = None) -> Transcript: ...


class TTS(Protocol):
    def synthesize(
        self, text: str, language: str, prosody: Prosody
    ) -> Iterator[tuple[np.ndarray, int]]: ...


class AudioSink(Protocol):
    async def play(self, samples: np.ndarray, sample_rate: int) -> None: ...
    def stop(self) -> None: ...


SHORT_CLIP_S = 1.6  # includes the VAD's preroll and its end-of-speech silence
MIN_LANGUAGE_CONFIDENCE = 0.7
MIN_AVG_LOGPROB = -1.2
# Whisper learned these from video subtitles and invents them from very little audio.
_HALLUCINATIONS = re.compile(
    r"(see you in the next video|thanks? for watching|thank you for watching|subscribe"
    r"|gracias por ver|suscr[ií]bete|nos vemos en el pr[oó]ximo|subt[ií]tulos|amara\.org)",
    re.IGNORECASE,
)


def _drop_hallucination(text: str) -> str:
    return "" if _HALLUCINATIONS.search(text) else text


def _add_cuda_dll_dirs() -> None:
    """On Windows the CUDA libraries from pip are not on the DLL search path by themselves."""
    if os.name != "nt":
        return
    for root in site.getsitepackages():
        for folder in glob.glob(os.path.join(root, "nvidia", "*", "bin")):
            os.add_dll_directory(folder)
            os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")


class WhisperSTT:
    """Transcribes a whole utterance at the endpoint.

    Fast enough on a GPU (about 100 ms) that incremental partials are not needed to
    meet the latency target. Falls back to a smaller model on the CPU, which is slower.
    """

    def __init__(
        self,
        model: str | None = None,
        device: str = "auto",
        languages: tuple[str, ...] = STT_LANGUAGES,
    ) -> None:
        _add_cuda_dll_dirs()
        from faster_whisper import WhisperModel

        self.languages = languages
        self.device = device
        self._model = None
        if device in ("auto", "cuda"):
            try:
                self._model = WhisperModel(model or "small", device="cuda", compute_type="float16")
                # Missing CUDA libraries only show up here, on the first real call.
                self._run(np.zeros(8000, dtype=np.float32))
                self.device = "cuda"
            except (RuntimeError, ValueError):
                if device == "cuda":
                    raise
                self._model = None
        if self._model is None:
            self._model = WhisperModel(model or "base", device="cpu", compute_type="int8")
            self.device = "cpu"
        self._run(np.zeros(8000, dtype=np.float32))  # the first call is slow; do it now

    def _run(self, audio: np.ndarray, language: str | None = None) -> tuple[str, str, float, list]:
        segments, info = self._model.transcribe(
            audio,
            language=language,
            beam_size=1,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
        )
        text = "".join(
            s.text for s in segments if s.no_speech_prob < 0.8 and s.avg_logprob > MIN_AVG_LOGPROB
        ).strip()
        probabilities = list(info.all_language_probs or [])
        return text, info.language, info.language_probability, probabilities

    def transcribe(self, audio: np.ndarray, prior_language: str | None = None) -> Transcript:
        """Language detection is unreliable on a short clip, so "Sí" could come back as
        English nonsense. Short or uncertain clips use the conversation's language."""
        seconds = len(audio) / 16000
        if prior_language in self.languages and seconds < SHORT_CLIP_S:
            text, language = self._run(audio, prior_language)[:2]
            return Transcript(_drop_hallucination(text), language)
        text, language, confidence, probabilities = self._run(audio)
        if language not in self.languages or confidence < MIN_LANGUAGE_CONFIDENCE:
            allowed = [(p, code) for code, p in probabilities if code in self.languages]
            if prior_language in self.languages and confidence < MIN_LANGUAGE_CONFIDENCE:
                language = prior_language
            else:
                language = max(allowed)[1] if allowed else self.languages[0]
            text = self._run(audio, language)[0]
        return Transcript(_drop_hallucination(text), language)


def shift_pitch(samples: np.ndarray, semitones: float) -> np.ndarray:
    """Resample: raises pitch and shortens the clip, which `length_scale` compensates for."""
    if abs(semitones) < 0.01 or len(samples) < 2:
        return samples
    factor = 2.0 ** (semitones / 12.0)
    positions = np.arange(0, len(samples) - 1, factor)
    return np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)


class PiperTTS:
    """One voice per language. Speed comes from Piper; pitch and energy are applied after."""

    def __init__(
        self,
        voice_dir: str | Path = DEFAULT_VOICE_DIR,
        voices: dict[str, str] | None = None,
    ) -> None:
        from piper import PiperVoice, SynthesisConfig

        self._config = SynthesisConfig
        self._voices = {}
        for language, name in (voices or DEFAULT_VOICES).items():
            path = Path(voice_dir) / f"{name}.onnx"
            if not path.exists():
                raise FileNotFoundError(
                    f"missing voice {path}. Download it with: "
                    f"python -m piper.download_voices --data-dir {voice_dir} {name}"
                )
            self._voices[language] = PiperVoice.load(path)
            list(self._voices[language].synthesize("Ok."))  # the first call is slow; do it now

    def synthesize(
        self, text: str, language: str, prosody: Prosody
    ) -> Iterator[tuple[np.ndarray, int]]:
        voice = self._voices.get(language) or next(iter(self._voices.values()))
        factor = 2.0 ** (prosody.pitch_semitones / 12.0)
        config = self._config(length_scale=factor / max(0.1, prosody.rate))
        for chunk in voice.synthesize(text, syn_config=config):
            audio = shift_pitch(chunk.audio_float_array, prosody.pitch_semitones)
            yield np.clip(audio * prosody.energy, -1.0, 1.0).astype(np.float32), chunk.sample_rate


class NullSink:
    """Plays nothing. With `realtime`, takes as long as the audio would."""

    def __init__(self, realtime: bool = False) -> None:
        self.realtime = realtime
        self.played: list[tuple[np.ndarray, int]] = []
        self._stop = asyncio.Event()

    async def play(self, samples: np.ndarray, sample_rate: int) -> None:
        self.played.append((samples, sample_rate))
        if self.realtime:
            self._stop.clear()
            try:
                await asyncio.wait_for(self._stop.wait(), len(samples) / sample_rate)
            except TimeoutError:
                pass

    def stop(self) -> None:
        self._stop.set()


class SoundDeviceSink:
    """The computer's speaker. `stop` cuts speech off within one block, for barge-in.

    The stream stays open between clips. Opening one takes about 500 ms on Windows,
    which would land on the first word and between every two sentences.
    """

    _BLOCK_S = 0.03

    def __init__(self, device: int | str | None = None, prepare_rate: int | None = 22050) -> None:
        import sounddevice

        self._sd = sounddevice
        self.device = device
        self._stop = threading.Event()
        self._streams: dict[int, Any] = {}
        self._lock = threading.Lock()
        if prepare_rate is not None:
            self._stream(prepare_rate)

    def _stream(self, rate: int):
        stream = self._streams.get(rate)
        if stream is None:
            stream = self._sd.OutputStream(
                samplerate=rate, channels=1, dtype="float32", device=self.device, latency="low"
            )
            stream.start()
            self._streams[rate] = stream
        return stream

    @property
    def latency_s(self) -> float:
        """What the sound hardware adds after a write, which latency reports must include."""
        return max((float(s.latency) for s in self._streams.values()), default=0.0)

    def _play_blocking(self, samples: np.ndarray, sample_rate: int, stop: threading.Event) -> None:
        block = max(1, int(sample_rate * self._BLOCK_S))
        for start in range(0, len(samples), block):
            if stop.is_set():
                return
            with self._lock:
                self._stream(sample_rate).write(samples[start : start + block].reshape(-1, 1))

    async def play(self, samples: np.ndarray, sample_rate: int) -> None:
        # Each clip has its own flag. Cancelling a task does not stop its thread, so a
        # shared flag cleared by the next clip would leave the old one playing on top.
        self._stop.set()
        self._stop = stop = threading.Event()
        await asyncio.to_thread(self._play_blocking, samples, sample_rate, stop)

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self.stop()
        with self._lock:
            for stream in self._streams.values():
                stream.stop()
                stream.close()
            self._streams.clear()


class Microphone:
    """Sends 32 ms frames to a callback on the event loop, stamped with capture time."""

    def __init__(self, on_frame, clock, device: int | str | None = None) -> None:
        import sounddevice

        from .vad import FRAME_SAMPLES, SAMPLE_RATE

        self._loop = asyncio.get_running_loop()
        self._on_frame = on_frame
        self._clock = clock
        self._stream = sounddevice.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SAMPLES,
            device=device,
            callback=self._callback,
        )

    def _callback(self, data, frames, time_info, status) -> None:
        frame = data[:, 0].copy()
        self._loop.call_soon_threadsafe(self._on_frame, frame, self._clock())

    def start(self) -> None:
        self._stream.start()

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()
