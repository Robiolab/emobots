"""Turn-taking: audio or text in; appraisals, cues, words and timing out.

The order is the point of this file. The robot reacts first, with no model involved.
The appraisal then reaches the emotion core while the reply is still generating, and
speech starts at the first finished sentence. A person talking over the robot cancels
all of it.
"""

from __future__ import annotations

import asyncio
import re
import statistics
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from affect_core import AffectEngine, AppraisalEvent
from affect_express import ExpressionMapper, load_profile

from .llm import LLM, LLMError, fallback_appraisal
from .prompt import (
    DEFAULT_PERSONA,
    ConversationMemory,
    clean_heard,
    describe_affect,
    system_prompt,
)
from .protocol import ParsedAppraisal, SentenceSplitter, TurnParser, soften_hostility
from .speech import STT, TTS, AudioSink, Prosody
from .vad import EchoGate, Endpointer, SpeechEnded, SpeechStarted

REACTION_CUE = "nod"
LISTEN_CUE = "listen"
THINK_CUE = "think"
BARGE_IN_S = 0.3

_TROUBLE = {
    "es": "Perdona, ahora mismo no puedo pensar bien.",
    "en": "Sorry, I'm having trouble thinking right now.",
}

Emit = Callable[[str, dict[str, Any]], None]
_WORD = re.compile(r"\w+")


def is_own_words(heard: str, recent: Iterable[str], threshold: float = 0.7) -> bool:
    """Whether a transcript is mostly the robot's own recent speech, picked up by the mic."""
    words = _WORD.findall(heard.lower())
    if not words:
        return False
    known = {w for sentence in recent for w in _WORD.findall(sentence.lower())}
    return sum(w in known for w in words) / len(words) >= threshold


@dataclass
class TurnTiming:
    """Clock times of one turn. Every latency is measured from true end of speech."""

    language: str = ""
    t_speech_end: float | None = None
    t_reaction: float | None = None
    t_transcript: float | None = None
    t_first_token: float | None = None
    t_appraisal: float | None = None
    t_first_sentence: float | None = None
    t_first_audio: float | None = None
    t_done: float | None = None
    interrupted: bool = False
    typed: bool = False

    def _ms(self, t: float | None) -> float | None:
        if t is None or self.t_speech_end is None:
            return None
        return round((t - self.t_speech_end) * 1000, 1)

    def report(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "interrupted": self.interrupted,
            "typed": self.typed,
            "reaction_ms": self._ms(self.t_reaction),
            "transcript_ms": self._ms(self.t_transcript),
            "first_token_ms": self._ms(self.t_first_token),
            "appraisal_ms": self._ms(self.t_appraisal),
            "first_sentence_ms": self._ms(self.t_first_sentence),
            "first_word_ms": self._ms(self.t_first_audio),
        }


class Conversation:
    def __init__(
        self,
        engine: AffectEngine,
        llm: LLM,
        *,
        mapper: ExpressionMapper | None = None,
        stt: STT | None = None,
        tts: TTS | None = None,
        sink: AudioSink | None = None,
        endpointer: Endpointer | None = None,
        persona: str = DEFAULT_PERSONA,
        emit: Emit | None = None,
        clock: Callable[[], float] = time.monotonic,
        memory_turns: int = 8,
        default_language: str = "es",
        llm_timeout_s: float = 25.0,
        barge_in_s: float = BARGE_IN_S,
    ) -> None:
        self.engine = engine
        self.llm = llm
        self.mapper = mapper
        self.stt = stt
        self.tts = tts
        self.sink = sink
        self.endpointer = endpointer
        self.memory = ConversationMemory(memory_turns)
        self.default_language = default_language
        self.llm_timeout_s = llm_timeout_s
        self.barge_in_s = barge_in_s
        self.timings: list[TurnTiming] = []
        self._emit_fn = emit
        self._clock = clock
        self._voice = ExpressionMapper(load_profile("voice"))
        cues = list(mapper.profile.cues) if mapper else []
        self._system = system_prompt(persona, cues)
        self._task: asyncio.Task | None = None
        self._speaking = False
        self._current: TurnTiming | None = None
        self._previous_state = None
        self._interruptions = 0
        self._language = default_language
        self._unanswered: str | None = None
        self._pending_audio: np.ndarray | None = None
        self._echo_gate = EchoGate()
        self._started_over_robot = False
        self._recent_said: deque[str] = deque(maxlen=6)

    # ---- inputs ----

    def feed_audio(self, frame: np.ndarray, t: float | None = None) -> None:
        """One microphone frame. Call it on the event loop."""
        if self.endpointer is None:
            raise RuntimeError("feed_audio needs an endpointer")
        now = self._clock() if t is None else t
        detect = self._echo_gate.process(frame, now)
        for event in self.endpointer.feed(frame, now, detect):
            if isinstance(event, SpeechStarted):
                self._started_over_robot = self._speaking
                self._cue(LISTEN_CUE)
                self._emit("speech_started", {})
            elif isinstance(event, SpeechEnded):
                self._on_speech_end(event)
        # Only sustained speech counts, so a cough or a door does not stop the robot.
        if self.endpointer.speech_duration(now) >= self.barge_in_s:
            if self._speaking:
                self.interrupt()
            else:
                # The person kept talking while the robot was still thinking. That is
                # not rude: drop the answer in progress and listen to the whole thought.
                self.silence()

    def say(self, text: str, language: str | None = None) -> asyncio.Task:
        """A typed turn, with no speech recognition. End of typing counts as end of speech."""
        timing = TurnTiming(t_speech_end=self._clock(), typed=True)
        self._react(timing)
        return self._start_turn(self._run_text(text, language or self.default_language, timing))

    def silence(self) -> bool:
        """Stop the current turn with no emotional effect. Returns whether one was running."""
        if self._task is None or self._task.done():
            return False
        if self._current is not None:
            self._current.interrupted = True
        self._task.cancel()
        if self.sink is not None:
            self.sink.stop()
        self._set_speaking(False)
        return True

    def interrupt(self) -> None:
        """The person talked over the robot: stop speaking and let it count, mildly."""
        if not self.silence():
            return
        self._interruptions += 1
        cause = "the person interrupted" if self._interruptions < 2 else (
            f"the person interrupted {self._interruptions} times"
        )
        self.engine.submit(
            AppraisalEvent(
                source="dialogue", cause=cause, agency="user", relevance=0.4, congruence=-0.2,
                novelty=0.1, control=0.1, norm=-0.2,
            )
        )
        self._emit("interrupted", {"cause": cause})

    async def wait(self) -> None:
        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def aclose(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        await self.wait()

    # ---- turn ----

    def _on_speech_end(self, event: SpeechEnded) -> None:
        timing = TurnTiming(t_speech_end=event.t_last_voice)
        self._react(timing)
        self._start_turn(self._run_audio(event.audio, timing))
        self._current = timing

    def _start_turn(self, coroutine) -> asyncio.Task:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = asyncio.create_task(coroutine)
        return self._task

    def _react(self, timing: TurnTiming) -> None:
        """The first reaction: immediate, from the affect state alone, with no model."""
        self._cue(REACTION_CUE)
        self._cue(THINK_CUE)
        timing.t_reaction = self._clock()
        self._emit("reaction", {"label": self.engine.state.label})

    async def _run_audio(self, audio: np.ndarray, timing: TurnTiming) -> None:
        if self.stt is None:
            raise RuntimeError("audio turns need an STT backend")
        if self._pending_audio is not None:
            audio = np.concatenate([self._pending_audio, audio])
            self._pending_audio = None
        try:
            transcript = await asyncio.to_thread(self.stt.transcribe, audio, self._language)
        except asyncio.CancelledError:
            # The person carried on before we finished: keep their words for the next turn.
            self._pending_audio = audio
            raise
        timing.t_transcript = self._clock()
        if not transcript.text.strip():
            self._emit("ignored", {"reason": "no words in the audio"})
            return
        if self._started_over_robot and is_own_words(transcript.text, self._recent_said):
            self._emit("ignored", {"reason": "the robot heard itself"})
            return
        await self._run_text(transcript.text, transcript.language, timing)

    async def _run_text(self, text: str, language: str, timing: TurnTiming) -> None:
        self._current = timing
        timing.language = language
        self._language = language
        heard = clean_heard(text)
        if self._unanswered:
            # A pause split one thought in two, and the first part never got an answer.
            heard = f"{self._unanswered} {heard}"
        self._unanswered = heard
        self._emit("heard", {"text": heard, "language": language})
        state = self.engine.state
        affect = describe_affect(state, self._previous_state)
        messages = self.memory.messages(heard, language, affect)
        self._previous_state = state
        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        said: list[str] = []  # what the person really heard, which memory must not overstate
        speaker = asyncio.create_task(self._speak(sentences, language, timing, said))
        spoken: list[str] = []
        splitter = SentenceSplitter()
        parser = TurnParser(fallback_appraisal(heard))

        def handle(item: ParsedAppraisal | str) -> None:
            if isinstance(item, ParsedAppraisal):
                self._apply_appraisal(item, timing)
                return
            for sentence in splitter.feed(item):
                push(sentence)

        def push(sentence: str) -> None:
            safe = soften_hostility(sentence, language)
            if timing.t_first_sentence is None:
                timing.t_first_sentence = self._clock()
            spoken.append(safe)
            sentences.put_nowait(safe)

        # Everything below runs under one finally, because a barge-in can cancel this
        # turn at any await: while the model streams, or while the robot speaks.
        try:
            try:
                async with asyncio.timeout(self.llm_timeout_s):
                    async for chunk in self.llm.stream(self._system, messages):
                        if timing.t_first_token is None:
                            timing.t_first_token = self._clock()
                        for item in parser.feed(chunk):
                            handle(item)
            except (LLMError, TimeoutError) as exc:
                self._emit("llm_error", {"error": str(exc) or type(exc).__name__})
            for item in parser.close():
                handle(item)
            for sentence in splitter.flush():
                push(sentence)
            if not spoken:
                push(_TROUBLE.get(language, _TROUBLE["en"]))
            sentences.put_nowait(None)
            await speaker
        finally:
            speaker.cancel()
            if said:
                self.memory.add(heard, language, " ".join(said))
                self._unanswered = None
            timing.t_done = self._clock()
            self.timings.append(timing)
            self._emit("turn", timing.report())

    def _apply_appraisal(self, parsed: ParsedAppraisal, timing: TurnTiming) -> None:
        timing.t_appraisal = self._clock()
        self.engine.submit(parsed.event)
        for cue in parsed.cues:
            self._cue(cue)
        self._emit(
            "appraisal",
            {"fallback": parsed.fallback, "cues": parsed.cues, "event": parsed.event.model_dump()},
        )

    async def _speak(
        self,
        sentences: asyncio.Queue[str | None],
        language: str,
        timing: TurnTiming,
        said: list[str],
    ) -> None:
        """Speak each sentence, reading the voice from the affect state as it is by then.

        A sentence only counts as said, in memory and on the dashboard, once it starts.
        """
        try:
            while (sentence := await sentences.get()) is not None:
                if self.tts is None:
                    said.append(sentence)
                    self._emit("reply", {"text": sentence})
                    continue
                channels = self._voice.evaluate(self.engine.state)
                prosody = Prosody.from_channels(channels)
                chunks = await asyncio.to_thread(
                    lambda s=sentence, p=prosody: list(self.tts.synthesize(s, language, p))
                )
                said.append(sentence)
                self._recent_said.append(sentence)
                self._emit("reply", {"text": sentence})
                for samples, rate in chunks:
                    if timing.t_first_audio is None:
                        # The person hears it only after the sound hardware's own delay.
                        delay = getattr(self.sink, "latency_s", 0.0)
                        timing.t_first_audio = self._clock() + delay
                    self._set_speaking(True)
                    self._echo_gate.reference(samples, rate, self._clock())
                    if self.sink is not None:
                        await self.sink.play(samples, rate)
                    silence = np.zeros(int(prosody.pause_s * rate), dtype=np.float32)
                    if self.sink is not None and len(silence):
                        await self.sink.play(silence, rate)
        finally:
            self._set_speaking(False)

    # ---- small helpers ----

    def _set_speaking(self, speaking: bool) -> None:
        self._speaking = speaking
        if self.mapper is not None:
            self.mapper.set_talking(speaking)

    def _cue(self, cue: str) -> None:
        if self.mapper is not None:
            self.mapper.trigger(cue, self.engine.t)

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        if self._emit_fn is not None:
            self._emit_fn(kind, {"t": self.engine.t, **data})


def latency_summary(reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Median latency per language against the plan's targets.

    Only spoken turns count. A typed turn has no end of speech to measure from, and
    its reaction time of zero would flatter the gate.
    """
    out: dict[str, dict[str, Any]] = {}
    spoken = [r for r in reports if not r.get("typed") and r["language"]]
    for language in sorted({r["language"] for r in spoken}):
        rows = [r for r in spoken if r["language"] == language and not r["interrupted"]]
        summary: dict[str, Any] = {"turns": len(rows)}
        for key in ("reaction_ms", "transcript_ms", "first_token_ms", "first_word_ms"):
            values = [r[key] for r in rows if r.get(key) is not None]
            summary[key] = round(statistics.median(values), 1) if values else None
        summary["pass_reaction"] = (
            summary["reaction_ms"] is not None and summary["reaction_ms"] < 1000
        )
        summary["pass_first_word"] = (
            summary["first_word_ms"] is not None and summary["first_word_ms"] < 1500
        )
        out[language] = summary
    return out


__all__ = ["Conversation", "TurnTiming", "is_own_words", "latency_summary"]
