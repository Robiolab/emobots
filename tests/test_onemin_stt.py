"""The 1min.ai speech-to-text backend, against a fake client: no network, no API key."""

import io
import wave
from importlib.metadata import entry_points
from types import SimpleNamespace

import numpy as np
from affect_dialogue.runtime import load_backend
from affect_dialogue.speech import OneMinSTT


class FakeClient:
    def __init__(self, text="hola, ¿cómo estás?", error=None):
        self.calls = []
        self._text, self._error = text, error
        self.audio = SimpleNamespace(transcribe=self._transcribe)

    def _transcribe(self, audio, **kwargs):
        if self._error:
            raise self._error
        self.calls.append((audio, kwargs))
        return SimpleNamespace(content=f"  {self._text} ")


def test_registered_as_a_plugin():
    assert "onemin" in {e.name for e in entry_points(group="affect.stt")}
    assert load_backend("stt", "onemin", client=FakeClient()).device == "1min.ai"


def test_sends_16khz_mono_wav_and_returns_text():
    client = FakeClient()
    stt = OneMinSTT(client=client)
    transcript = stt.transcribe(np.full(32000, 0.5, dtype=np.float32))
    (name, wav_bytes), options = client.calls[0]
    assert name.endswith(".wav") and options == {}
    with wave.open(io.BytesIO(wav_bytes)) as wav:
        assert (wav.getnchannels(), wav.getframerate(), wav.getnframes()) == (1, 16000, 32000)
    assert transcript.text == "hola, ¿cómo estás?"
    assert transcript.language == "es"


def test_short_clip_uses_the_conversation_language_as_a_hint():
    client = FakeClient("yes")
    OneMinSTT(client=client).transcribe(np.zeros(8000, dtype=np.float32), prior_language="en")
    assert client.calls[0][1] == {"language": "en"}


def test_detects_english_and_passes_the_model():
    client = FakeClient("what is your name and how are you")
    stt = OneMinSTT(model="gpt-4o-transcribe", client=client)
    assert stt.transcribe(np.zeros(32000, dtype=np.float32), prior_language="es").language == "en"
    assert client.calls[0][1] == {"model": "gpt-4o-transcribe"}


def test_api_failure_drops_the_turn_instead_of_raising():
    stt = OneMinSTT(client=FakeClient(error=ConnectionError("down")))
    assert stt.transcribe(np.zeros(32000, dtype=np.float32), prior_language="en").text == ""
