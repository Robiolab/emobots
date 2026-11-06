"""The real speech models, end to end. Skipped unless they are installed.

    uv sync --extra speech --extra gpu
    uv run --extra speech --extra gpu pytest -m speech
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("faster_whisper")
pytest.importorskip("piper")

VOICES = Path(__file__).parent.parent / "models" / "piper"
pytestmark = [
    pytest.mark.speech,
    pytest.mark.skipif(not VOICES.exists(), reason="voices not downloaded"),
]

CLIPS = [
    ("en", "Hello robot, you did a really great job today.", {"robot", "great", "job"}),
    ("es", "Hola robot, hoy hiciste un trabajo muy bueno.", {"trabajo", "bueno"}),
]


@pytest.fixture(scope="module")
def models():
    from affect_dialogue.speech import PiperTTS, WhisperSTT

    return WhisperSTT(), PiperTTS(VOICES)


def to_16k(chunks):
    audio = np.concatenate([s for s, _ in chunks])
    rate = chunks[0][1]
    positions = np.arange(0, len(audio) - 1, rate / 16000)
    return np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)


@pytest.mark.parametrize("language,text,words", CLIPS)
def test_robot_understands_its_own_voice_in_both_languages(models, language, text, words):
    from affect_dialogue.speech import Prosody

    stt, tts = models
    result = stt.transcribe(to_16k(list(tts.synthesize(text, language, Prosody()))))
    assert result.language == language
    assert words <= set(result.text.lower().replace(",", " ").replace(".", " ").split())


@pytest.mark.parametrize("language,text,words", CLIPS)
def test_silero_finds_real_speech_and_not_silence(models, language, text, words):
    from affect_dialogue.speech import Prosody
    from affect_dialogue.vad import FRAME_SAMPLES, Endpointer, SileroVad, SpeechEnded, SpeechStarted

    _, tts = models
    speech = to_16k(list(tts.synthesize(text, language, Prosody())))
    silence = np.zeros(16000, dtype=np.float32)
    clip = np.concatenate([silence, speech, silence])
    endpointer = Endpointer(SileroVad())
    events = []
    for i in range(len(clip) // FRAME_SAMPLES):
        frame = clip[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
        events += endpointer.feed(frame, (i + 1) * FRAME_SAMPLES / 16000)
    assert isinstance(events[0], SpeechStarted) and isinstance(events[-1], SpeechEnded)
    speech_end = 1.0 + len(speech) / 16000
    assert abs(events[-1].t_last_voice - speech_end) < 0.4


def test_prosody_really_changes_the_audio(models):
    from affect_dialogue.speech import Prosody

    _, tts = models
    text = "This is a test of the voice."

    def say(prosody):
        return np.concatenate([s for s, _ in tts.synthesize(text, "en", prosody)])

    def mean_length(prosody):  # Piper is slightly random, so one clip proves nothing
        return np.mean([len(say(prosody)) for _ in range(6)])

    normal = mean_length(Prosody())
    assert mean_length(Prosody(rate=1.3)) < 0.92 * normal
    assert mean_length(Prosody(rate=0.8)) > 1.08 * normal
    assert np.abs(say(Prosody(energy=0.6))).max() < np.abs(say(Prosody())).max()
    # Pitch must not change the length much, or high-arousal speech would rush.
    assert abs(mean_length(Prosody(pitch_semitones=3)) - normal) < 0.2 * normal


async def test_full_pipeline_with_real_models_end_to_end(models):
    import asyncio

    from affect_core import PRESETS, AffectEngine
    from affect_dialogue.conversation import Conversation
    from affect_dialogue.llm import ScriptedLLM
    from affect_dialogue.speech import NullSink, Prosody
    from affect_dialogue.vad import FRAME_SAMPLES, Endpointer, SileroVad

    stt, tts = models
    engine = AffectEngine(PRESETS["pet"])
    sink = NullSink()
    events = []
    conversation = Conversation(
        engine, ScriptedLLM(), stt=stt, tts=tts, sink=sink, endpointer=Endpointer(SileroVad()),
        emit=lambda kind, data: events.append((kind, data)),
    )
    speech = to_16k(list(tts.synthesize("Muchas gracias, eres genial.", "es", Prosody())))
    clip = np.concatenate([np.zeros(8000, np.float32), speech, np.zeros(16000, np.float32)])
    for i in range(len(clip) // FRAME_SAMPLES):
        conversation.feed_audio(clip[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES], i * 0.032)
    await asyncio.wait_for(conversation.wait(), 20)
    engine.run_until(0.1)
    assert engine.state.label == "joy"
    assert [d["language"] for k, d in events if k == "heard"] == ["es"]
    assert sink.played
