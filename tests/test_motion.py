"""One recorded arm movement per emotion, with the emotion held while it plays."""

import json
import random
import time

import pytest
from affect_server import create_app
from affect_server.motion import MotionLibrary, MotionPlayer, parse_motion_map
from fastapi.testclient import TestClient


class FakeArm:
    def __init__(self):
        self.sent = []

    def send_thetas(self, thetas, speed=50):
        self.sent.append((tuple(thetas), speed))


def make_recordings(root, kinds=("happy", "sad", "angry", "disgust"), per_kind=3, seconds=1.0):
    for kind in kinds:
        for i in range(per_kind):
            folder = root / f"{kind}_wave_2026{i:04d}"
            folder.mkdir(parents=True)
            (folder / "meta.json").write_text(
                json.dumps({"name": "wave", "emotions": [kind, "other"], "duration": seconds})
            )
            rows = ["t,theta1,theta2,theta3,theta4"]
            n = int(seconds * 20)
            rows += [f"{k / 20:.4f},{i},{k},{k},{k}" for k in range(n + 1)]
            (folder / "thetas.csv").write_text("\n".join(rows) + "\n")


@pytest.fixture
def recordings(tmp_path):
    make_recordings(tmp_path / "rec")
    return tmp_path / "rec"


def test_library_groups_by_first_emotion_and_picks_one_of_that_type(recordings):
    library = MotionLibrary(recordings, random.Random(1))
    assert library.counts() == {"angry": 3, "disgust": 3, "happy": 3, "sad": 3}
    picks = {library.pick("happy").folder.name for _ in range(30)}
    assert len(picks) > 1 and all(name.startswith("happy_") for name in picks)
    assert library.pick("surprise") is None


def test_player_sends_every_sample_to_the_arm_and_only_for_mapped_emotions(recordings):
    arm = FakeArm()
    player = MotionPlayer(MotionLibrary(recordings), arm, speed=20)
    assert player.choose("surprise", 0.9) is None  # no recording type mapped
    assert player.choose("joy", 0.1) is None  # too faint
    movement = player.choose("joy", 0.9)
    assert movement.kind == "happy"
    player.start(movement)
    deadline = time.monotonic() + 5
    while player.playing and time.monotonic() < deadline:
        time.sleep(0.02)
    assert len(arm.sent) == len(movement.samples) + 1  # +1: the move to the first pose
    assert all(speed == 20 for _, speed in arm.sent)
    assert arm.sent[0][0] == movement.samples[0][1]


def test_player_stops_when_asked(recordings):
    arm = FakeArm()
    player = MotionPlayer(MotionLibrary(recordings), arm)
    player.start(player.choose("joy", 0.9))
    time.sleep(0.1)
    player.stop()
    time.sleep(0.2)
    assert not player.playing and len(arm.sent) < 20


def test_motion_map_parsing():
    assert parse_motion_map("joy=happy, fear=disgust") == {"joy": "happy", "fear": "disgust"}
    with pytest.raises(ValueError):
        parse_motion_map("joy")


def wait_for(client, predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get("/api/state").json()
        if predicate(body):
            return body
        time.sleep(0.02)
    raise AssertionError(f"condition not met; last state {body['state']}")


def test_joy_plays_a_happy_movement_and_holds_the_emotion(tmp_path, recordings):
    arm = FakeArm()
    app = create_app(log_dir=tmp_path / "logs", motions=recordings, arm=arm)
    with TestClient(app) as client:
        host = app.state.host
        base = host.base_personality
        client.post("/api/say", json={"text": "eres genial, gracias", "language": "es"})
        wait_for(client, lambda b: b["state"]["label"] == "joy")
        wait_for(client, lambda b: host.motion.playing)
        # held: the engine runs a slow emotion for now, but the dashboard sees the real one
        wait_for(client, lambda b: host.engine.personality.tau_emotion > base.tau_emotion * 10)
        assert client.get("/api/state").json()["personality"]["tau_emotion"] == base.tau_emotion
        wait_for(client, lambda b: host.engine.personality.tau_emotion == base.tau_emotion)
        assert not host.motion.playing and arm.sent
        log = client.get("/api/state").json()["log"]
    from affect_core import read_log, replay

    kinds = [r["kind"] for r in read_log(log)]
    assert kinds.count("motion") == 1
    assert replay(log).exact


def test_one_movement_per_emotion_not_one_per_tick(tmp_path, recordings):
    app = create_app(log_dir=tmp_path / "logs", motions=recordings, arm=FakeArm())
    with TestClient(app) as client:
        host = app.state.host
        client.post("/api/say", json={"text": "gracias", "language": "es"})
        wait_for(client, lambda b: host.motion.playing)
        wait_for(client, lambda b: not host.motion.playing and host._held_from is None)
        time.sleep(0.5)
        log = client.get("/api/state").json()["log"]
    from affect_core import read_log

    assert [r["kind"] for r in read_log(log)].count("motion") == 1


def test_stop_abandons_the_movement_and_releases_the_emotion(tmp_path, recordings):
    app = create_app(log_dir=tmp_path / "logs", motions=recordings, arm=FakeArm())
    with TestClient(app) as client:
        host = app.state.host
        client.post("/api/say", json={"text": "gracias", "language": "es"})
        wait_for(client, lambda b: host.motion.playing)
        client.post("/api/stop")
        wait_for(client, lambda b: not host.motion.playing and host._held_from is None)


def test_personality_change_while_holding_takes_effect_after(tmp_path, recordings):
    app = create_app(log_dir=tmp_path / "logs", motions=recordings, arm=FakeArm())
    with TestClient(app) as client:
        host = app.state.host
        client.post("/api/say", json={"text": "gracias", "language": "es"})
        wait_for(client, lambda b: host.motion.playing)
        client.post("/api/personality", json={"gain": 1.5})
        assert client.get("/api/state").json()["personality"]["gain"] == 1.5
        wait_for(client, lambda b: host.engine.personality.tau_emotion > 100)  # holding
        wait_for(client, lambda b: host.engine.personality.tau_emotion < 100)  # released
        assert host.engine.personality.gain == 1.5


def test_fear_plays_a_disgust_movement(recordings):
    player = MotionPlayer(MotionLibrary(recordings), FakeArm())
    assert player.choose("fear", 0.7).kind == "disgust"


def test_dashboard_is_told_when_the_arm_starts_and_stops(tmp_path, recordings):
    app = create_app(log_dir=tmp_path / "logs", motions=recordings, arm=FakeArm())
    with TestClient(app) as client:
        host = app.state.host
        assert client.get("/api/state").json()["acting"] is None
        client.post("/api/say", json={"text": "gracias", "language": "es"})
        body = wait_for(client, lambda b: b["acting"] is not None)
        assert body["acting"]["movement"] == "happy" and body["acting"]["remaining"] > 0
        wait_for(client, lambda b: b["acting"] is None)
        assert not host.motion.playing


def test_motion_message_keeps_its_type_on_the_websocket(tmp_path, recordings):
    app = create_app(log_dir=tmp_path / "logs", motions=recordings, arm=FakeArm())
    with TestClient(app) as client, client.websocket_connect("/ws") as socket:
        socket.receive_json()  # hello
        client.post("/api/say", json={"text": "gracias", "language": "es"})
        types = []
        for _ in range(400):
            message = socket.receive_json()
            types.append(message["type"])
            if message["type"] == "motion":
                assert message["movement"] == "happy" and message["seconds"] > 0
                break
        assert "motion" in types
