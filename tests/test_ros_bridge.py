"""The ROS bridge's logic, with no ROS. tests/ros/verify.sh covers the real thing."""

import json

import pytest
from affect_ros import AffectBridge


@pytest.fixture
def bridge():
    states, commands, warnings = [], [], []
    b = AffectBridge(states.append, commands.append, warn=warnings.append)
    b.states, b.commands, b.warnings = states, commands, warnings
    return b


GOOD = json.dumps(
    {"cause": "a threat", "relevance": 1, "congruence": -0.8, "novelty": 0.8, "agency": "user",
     "control": -0.9, "norm": -0.3}
)


def test_tick_publishes_state_and_limit_checked_commands(bridge):
    bridge.tick(1.0)
    assert json.loads(bridge.states[-1])["label"] == "neutral"
    command = json.loads(bridge.commands[-1])
    limits = bridge.adapter.capabilities()
    for name, value in command["channels"].items():
        assert limits[name].min <= value <= limits[name].max


def test_appraisal_moves_the_robot(bridge):
    bridge.tick(1.0)
    rest = json.loads(bridge.commands[-1])["channels"]["body.height"]
    assert bridge.on_appraisal(GOOD)
    bridge.tick(3.0)
    assert json.loads(bridge.states[-1])["cause"] == "a threat"
    assert json.loads(bridge.commands[-1])["channels"]["body.height"] < rest


@pytest.mark.parametrize(
    "text", ["not json", "[]", "{}", '{"cause":"x","relevance":99,"congruence":0,"novelty":0,'
    '"control":0,"norm":0}']
)
def test_bad_appraisals_never_reach_the_engine(bridge, text):
    before = bridge.engine.state
    assert not bridge.on_appraisal(text)
    assert bridge.rejected == 1 and bridge.warnings
    assert bridge.engine.state == before


def test_another_nodes_clock_cannot_reorder_time(bridge):
    bridge.tick(1.0)
    stale = json.loads(GOOD) | {"t": 99999.0}
    assert bridge.on_appraisal(json.dumps(stale))
    bridge.tick(1.1)
    assert json.loads(bridge.states[-1])["cause"] == "a threat"


def test_stop_sends_rest_once_then_refuses(bridge):
    bridge.tick(1.0)
    bridge.stop()
    rest = json.loads(bridge.commands[-1])["channels"]
    assert rest["body.height"] == bridge.adapter.capabilities()["body.height"].rest
    count = len(bridge.commands)
    bridge.tick(2.0)
    assert len(bridge.commands) == count
    bridge.resume()
    bridge.tick(3.0)
    assert len(bridge.commands) > count


def test_unknown_cue_is_reported(bridge):
    assert bridge.on_cue("nod") and not bridge.on_cue("explode")
