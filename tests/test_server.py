"""The server drives a real engine, logs it, and its live sessions replay exactly."""

import time

import pytest
from affect_core import PRESETS, read_log, replay
from affect_server import create_app
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(log_dir=tmp_path)) as c:
        yield c


def wait_for(client, predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get("/api/state").json()
        if predicate(body):
            return body
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


def test_dashboard_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Affect" in response.text


def test_catalog_lists_events_and_presets(client):
    body = client.get("/api/catalog").json()
    assert "praise" in body["events"]
    assert set(body["presets"]) == set(PRESETS)


def test_named_event_changes_state_with_cause(client):
    assert client.post("/api/event/praise").status_code == 200
    body = wait_for(client, lambda b: b["state"]["label"] == "joy")
    assert body["state"]["cause"] == "you praised it"
    assert body["recent"][-1]["event"]["cause"] == "you praised it"


def test_unknown_event_is_404(client):
    assert client.post("/api/event/nope").status_code == 404


def test_custom_event_with_bad_values_is_rejected(client):
    bad = {"cause": "x", "relevance": 9, "congruence": 0, "novelty": 0, "control": 0, "norm": 0}
    assert client.post("/api/event", json=bad).status_code == 422


def test_custom_event_ignores_client_clock(client):
    event = {"t": 1e9, "cause": "custom", "relevance": 1, "congruence": 1, "novelty": 0,
             "control": 0, "norm": 0}
    assert client.post("/api/event", json=event).status_code == 200
    wait_for(client, lambda b: b["state"]["cause"] == "custom")


def test_personality_preset_and_adjustment(client):
    assert client.post("/api/personality", json={"preset": "shy"}).status_code == 200
    wait_for(client, lambda b: b["personality"]["name"] == "shy")
    body = client.post("/api/personality", json={"tau_mood": 60.0}).json()
    assert body["tau_mood"] == 60.0
    assert body["alpha"] == pytest.approx(0.04)


def test_personality_rejects_bad_values(client):
    assert client.post("/api/personality", json={"preset": "nope"}).status_code == 404
    assert client.post("/api/personality", json={"gain": -1}).status_code == 422
    assert client.post("/api/personality", json={"tau_mood": 0.5}).status_code == 422


def test_reset_returns_to_baseline(client):
    client.post("/api/event/threat")
    wait_for(client, lambda b: b["state"]["label"] == "fear")
    client.post("/api/reset")
    body = wait_for(client, lambda b: b["state"]["cause"] is None)
    assert body["state"]["label"] == "neutral"


def test_websocket_sends_hello_then_state_and_events(client):
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "hello"
        client.post("/api/event/pet")
        seen = set()
        for _ in range(100):
            seen.add(ws.receive_json()["type"])
            if {"state", "event"} <= seen:
                break
        assert {"state", "event"} <= seen


def test_live_session_replays_exactly(tmp_path):
    # The gate holds for wall-clock sessions too, not only scripted ones.
    with TestClient(create_app(log_dir=tmp_path)) as c:
        c.post("/api/event/praise")
        time.sleep(0.2)
        c.post("/api/personality", json={"preset": "grumpy", "tau_mood": 60.0})
        c.post("/api/event/insult")
        time.sleep(0.2)
        c.post("/api/reset")
        time.sleep(0.2)
        log = c.get("/api/state").json()["log"]
    kinds = [r["kind"] for r in read_log(log)]
    assert kinds.count("event") == 2 and "reset" in kinds and "personality" in kinds
    result = replay(log)
    assert result.exact, f"first mismatch at tick {result.first_mismatch}"
