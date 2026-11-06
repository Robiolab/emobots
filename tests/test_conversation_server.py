"""The whole stack in one process: typed turns, expression, logs, and replay."""

import time

import pytest
from affect_core import read_log, replay
from affect_server import create_app
from fastapi.testclient import TestClient


def wait_for(client, predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get("/api/state").json()
        if predicate(body):
            return body
        time.sleep(0.02)
    raise AssertionError(f"condition not met; last state {body['state']}")


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(log_dir=tmp_path)) as c:
        yield c


def test_typed_turn_changes_emotion_and_the_face(client):
    rest = client.get("/api/state").json()["expression"]["face.mouth_curve"]
    said = client.post("/api/say", json={"text": "eres genial, gracias", "language": "es"})
    assert said.is_success
    body = wait_for(client, lambda b: b["state"]["label"] == "joy")
    assert body["state"]["cause"] == "the person was kind"
    assert [m["type"] for m in body["talk"]] == ["heard", "reply"]
    assert body["talk"][1]["text"] == "Qué amable, muchas gracias."
    wait_for(client, lambda b: b["expression"]["face.mouth_curve"] > rest + 0.2)


def test_say_rejects_bad_input(client):
    assert client.post("/api/say", json={"text": ""}).status_code == 422
    assert client.post("/api/say", json={"text": "x", "language": "fr"}).status_code == 422
    assert client.post("/api/say", json={"text": "x" * 3000}).status_code == 422


def test_session_with_conversation_still_replays_exactly(tmp_path):
    with TestClient(create_app(log_dir=tmp_path)) as c:
        c.post("/api/say", json={"text": "eres genial", "language": "es"})
        wait_for(c, lambda b: b["state"]["label"] == "joy")
        c.post("/api/say", json={"text": "I will unplug you", "language": "en"})
        wait_for(c, lambda b: b["state"]["cause"] == "the person made a threat")
        log = c.get("/api/state").json()["log"]
    kinds = {r["kind"] for r in read_log(log)}
    assert {"heard", "reply", "appraisal", "turn", "command", "event", "state"} <= kinds
    assert replay(log).exact


def test_every_emotional_change_traces_to_what_was_said(tmp_path):
    with TestClient(create_app(log_dir=tmp_path)) as c:
        c.post("/api/say", json={"text": "gracias", "language": "es"})
        wait_for(c, lambda b: b["state"]["cause"] is not None)
        log = c.get("/api/state").json()["log"]
    records = list(read_log(log))
    order = [r["kind"] for r in records]
    assert order.index("heard") < order.index("appraisal") < order.index("event")
    event = next(r for r in records if r["kind"] == "event")
    assert event["data"]["event"]["source"] == "dialogue"


def test_software_stop_rests_the_robot_and_refuses_commands(tmp_path):
    with TestClient(create_app(log_dir=tmp_path, profile="quadruped")) as c:
        c.post("/api/event/threat")
        wait_for(c, lambda b: b["expression"]["body.height"] < 0.94)
        c.post("/api/stop")
        body = wait_for(c, lambda b: b["expression"]["body.height"] == 0.95)
        time.sleep(0.15)
        assert c.get("/api/state").json()["expression"]["body.height"] == 0.95
        c.post("/api/resume")
        wait_for(c, lambda b: b["expression"]["body.height"] != 0.95)
        assert body["profile"] == "quadruped"


def test_same_server_drives_either_robot_by_config(tmp_path):
    for profile, channel in (("avatar", "face.eye_open"), ("quadruped", "gait.freq")):
        with TestClient(create_app(log_dir=tmp_path, profile=profile)) as c:
            c.post("/api/say", json={"text": "eres genial", "language": "es"})
            body = wait_for(c, lambda b: b["state"]["label"] == "joy")
            assert channel in body["expression"]


def test_typed_turns_are_logged_but_never_count_toward_the_latency_gate(tmp_path):
    with TestClient(create_app(log_dir=tmp_path)) as c:
        c.post("/api/say", json={"text": "hola", "language": "es"})
        body = wait_for(c, lambda b: any(m["type"] == "reply" for m in b["talk"]))
        time.sleep(0.05)
        assert c.get("/api/state").json()["latency"] == {}
        log = body["log"]
    turns = [r for r in read_log(log) if r["kind"] == "turn"]
    assert len(turns) == 1 and turns[0]["data"]["typed"] is True


def test_unknown_llm_backend_fails_at_startup(tmp_path):
    with pytest.raises(ValueError, match="scripted"):
        with TestClient(create_app(log_dir=tmp_path, llm="nope")):
            pass


def test_operator_stop_is_not_appraised_as_the_person_being_rude(tmp_path):
    with TestClient(create_app(log_dir=tmp_path)) as c:
        c.post("/api/say", json={"text": "hola", "language": "es"})
        c.post("/api/stop")
        c.post("/api/resume")
        time.sleep(0.1)
        log = c.get("/api/state").json()["log"]
    causes = [r["data"]["event"]["cause"] for r in read_log(log) if r["kind"] == "event"]
    assert not any("interrupted" in cause for cause in causes)
