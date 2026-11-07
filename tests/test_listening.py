"""The dashboard's Escuchar button switches the microphone on and off."""

import pytest
from affect_server import create_app
from fastapi.testclient import TestClient


class FakeMic:
    instances = []

    def __init__(self):
        self.started = self.closed = False
        FakeMic.instances.append(self)

    def start(self):
        self.started = True

    def close(self):
        self.closed = True


class FakeEndpointer:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


@pytest.fixture
def client(tmp_path):
    FakeMic.instances.clear()
    with TestClient(create_app(log_dir=tmp_path)) as c:
        host = c.app.state.host
        host.microphone_factory = lambda conversation: FakeMic()
        host.conversation.endpointer = FakeEndpointer()
        yield c


def test_no_speech_means_no_button(tmp_path):
    with TestClient(create_app(log_dir=tmp_path)) as c:
        assert c.get("/api/state").json()["can_listen"] is False
        assert c.post("/api/listening", json={"on": True}).status_code == 409


def test_starts_muted_then_the_button_turns_listening_on_and_off(client):
    hello = client.get("/api/state").json()
    assert hello["can_listen"] is True and hello["mic"] is False
    assert client.post("/api/listening", json={"on": True}).json() == {"listening": True}
    assert client.get("/api/state").json()["mic"] is True
    assert FakeMic.instances[0].started
    assert client.post("/api/listening", json={"on": True}).json() == {"listening": True}
    assert len(FakeMic.instances) == 1  # already on: no second microphone
    assert client.post("/api/listening", json={"on": False}).json() == {"listening": False}
    assert FakeMic.instances[0].closed
    assert client.app.state.host.conversation.endpointer.resets == 1  # no half utterance kept
    assert client.get("/api/state").json()["mic"] is False


def test_everyone_watching_is_told_when_it_changes(client):
    with client.websocket_connect("/ws") as socket:
        socket.receive_json()  # hello
        client.post("/api/listening", json={"on": True})
        seen = []
        for _ in range(400):
            message = socket.receive_json()
            if message["type"] == "listening":
                seen.append(message["on"])
                break
        assert seen == [True]


def test_rejects_a_bad_body(client):
    assert client.post("/api/listening", json={"on": "maybe"}).status_code == 422
    assert client.post("/api/listening", json={}).status_code == 422


def test_the_dashboard_is_revalidated_on_every_reload(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
