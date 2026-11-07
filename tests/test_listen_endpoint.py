"""The dashboard's record button posts PCM to /api/listen; it runs like a typed turn."""

import time

import numpy as np
import pytest
from affect_dialogue.speech import Transcript
from affect_server import create_app
from fastapi.testclient import TestClient


class FakeSTT:
    def __init__(self, text="eres genial, gracias", language="es"):
        self.calls = []
        self._transcript = Transcript(text, language)

    def transcribe(self, audio, prior_language=None):
        self.calls.append((audio, prior_language))
        return self._transcript


def pcm(seconds, rate=16000):
    return (np.full(int(seconds * rate), 1000, dtype="<i2")).tobytes()


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(log_dir=tmp_path)) as c:
        yield c


def with_stt(client, stt):
    client.app.state.host.conversation.stt = stt
    return stt


def test_needs_a_speech_backend(client):
    response = client.post("/api/listen", content=pcm(1))
    assert response.status_code == 409


def test_transcribes_and_runs_the_turn(client):
    stt = with_stt(client, FakeSTT())
    response = client.post("/api/listen?language=es", content=pcm(1))
    assert response.json() == {"text": "eres genial, gracias", "language": "es", "queued": True}
    audio, prior = stt.calls[0]
    assert prior == "es" and audio.dtype == np.float32 and len(audio) == 16000
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        body = client.get("/api/state").json()
        if body["state"]["label"] == "joy":
            break
        time.sleep(0.02)
    assert body["state"]["label"] == "joy"
    assert body["talk"][0]["type"] == "heard"


def test_no_words_queues_nothing(client):
    with_stt(client, FakeSTT(text=""))
    assert client.post("/api/listen", content=pcm(1)).json() == {"text": "", "queued": False}
    assert client.get("/api/state").json()["talk"] == []


def test_rejects_bad_recordings(client):
    with_stt(client, FakeSTT())
    assert client.post("/api/listen", content=pcm(0.05)).status_code == 422
    assert client.post("/api/listen", content=pcm(61)).status_code == 413
    assert client.post("/api/listen?language=fr", content=pcm(1)).status_code == 422


def test_hello_says_whether_the_server_has_a_live_microphone(client):
    hello = client.get("/api/state").json()
    assert hello["voice"] is False and hello["mic"] is False


class FakeRecorder:
    def __init__(self, seconds=1.0):
        self.recording = False
        self._seconds = seconds

    def start(self):
        self.recording = True

    def stop(self):
        self.recording = False
        return np.full(int(self._seconds * 16000), 0.1, dtype=np.float32)


def test_server_recorder_button_records_then_runs_the_turn(client):
    with_stt(client, FakeSTT())
    client.app.state.host.recorder = FakeRecorder()
    assert client.get("/api/state").json()["recorder"] is True
    assert client.post("/api/record/start").json() == {"recording": True}
    assert client.post("/api/record/start").status_code == 409  # already recording
    stopped = client.post("/api/record/stop?language=es").json()
    assert stopped["text"] == "eres genial, gracias" and stopped["queued"] is True
    assert client.post("/api/record/stop").status_code == 409  # nothing recording


def test_server_recorder_needs_push_to_talk_and_rejects_short_audio(client):
    assert client.post("/api/record/start").status_code == 409
    with_stt(client, FakeSTT())
    client.app.state.host.recorder = FakeRecorder(seconds=0.05)
    client.post("/api/record/start")
    assert client.post("/api/record/stop").status_code == 422
