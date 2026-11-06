"""Embodiment: profiles, the mapper, adapter safety, and one core on many robots."""

import numpy as np
import pytest
from affect_adapters import AvatarAdapter, SdkAdapter, SimulatorAdapter, load_limits
from affect_core import CATALOG, PRESETS, AffectEngine, AffectState, ExpressionCommand
from affect_express import (
    BUILTIN_DIR,
    EmbodimentProfile,
    ExpressionLoop,
    ExpressionMapper,
    load_profile,
    remap,
)
from pydantic import ValidationError


def state_for(label, emotion, intensity=0.8, t=0.0):
    return AffectState(
        t=t, emotion=emotion, mood=(0, 0, 0), baseline=(0, 0, 0), delta=(0, 0, 0),
        label=label, intensity=intensity,
    )


JOY = state_for("joy", (0.76, 0.48, 0.35))
SAD = state_for("sadness", (-0.6, -0.35, -0.35))
FEAR = state_for("fear", (-0.5, 0.5, -0.4))
REST = state_for("neutral", (0.0, 0.0, 0.0), intensity=0.0)


def values(profile, state):
    return ExpressionMapper(load_profile(profile)).evaluate(state)


# ---- the plan's example mappings table ----


def test_quadruped_matches_plan_table():
    rest, joy, sad, fear = (values("quadruped", s) for s in (REST, JOY, SAD, FEAR))
    assert joy["body.height"] > rest["body.height"]
    assert joy["gait.freq"] > rest["gait.freq"] and joy["body.sway"] > rest["body.sway"]
    assert sad["body.height"] < rest["body.height"] and sad["gait.freq"] < rest["gait.freq"]
    assert sad["body.pitch"] > rest["body.pitch"]  # front down
    assert fear["body.height"] < sad["body.height"]  # crouch
    assert fear["impedance.stiffness"] > rest["impedance.stiffness"]
    assert fear["body.x"] < rest["body.x"]  # step back


def test_hand_matches_plan_table():
    rest, joy, sad, fear = (values("hand", s) for s in (REST, JOY, SAD, FEAR))
    assert joy["finger.curl"] < rest["finger.curl"] and joy["gesture.speed"] > rest["gesture.speed"]
    assert sad["finger.curl"] > rest["finger.curl"] and sad["gesture.speed"] < rest["gesture.speed"]
    assert fear["grip.compliance"] < rest["grip.compliance"]


def test_avatar_matches_plan_table():
    rest, joy, sad, fear = (values("avatar", s) for s in (REST, JOY, SAD, FEAR))
    assert joy["face.mouth_curve"] > rest["face.mouth_curve"]
    assert joy["face.blink_rate"] > rest["face.blink_rate"]
    assert sad["face.mouth_curve"] < rest["face.mouth_curve"]
    assert fear["face.eye_open"] > rest["face.eye_open"]
    assert fear["face.brow_raise"] > rest["face.brow_raise"]


def test_voice_matches_plan_table():
    rest, joy, sad, fear = (values("voice", s) for s in (REST, JOY, SAD, FEAR))
    assert joy["voice.rate"] > rest["voice.rate"] and joy["voice.pitch"] > rest["voice.pitch"]
    assert sad["voice.rate"] < rest["voice.rate"] and sad["voice.pause"] > rest["voice.pause"]
    assert fear["voice.rate"] > rest["voice.rate"] and fear["voice.pitch"] > sad["voice.pitch"]


# ---- profiles ----


@pytest.mark.parametrize("path", sorted(BUILTIN_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def test_builtin_profiles_load_and_rest_inside_limits(path):
    profile = load_profile(path)
    for value, (name, channel) in zip(
        ExpressionMapper(profile).evaluate(REST).values(), profile.channels.items(), strict=True
    ):
        assert channel.limits[0] <= value <= channel.limits[1], name


def test_bad_profiles_are_rejected():
    channel = {"limits": [0, 1], "rest": 0.5, "rate_limit": 1}
    with pytest.raises(ValidationError):
        EmbodimentProfile(name="x", channels={"a": {**channel, "rest": 2}})
    with pytest.raises(ValidationError):
        EmbodimentProfile(
            name="x", channels={"a": channel},
            cues={"nod": [{"channel": "missing", "amplitude": 1, "duration_s": 1}]},
        )
    with pytest.raises(FileNotFoundError):
        load_profile("no-such-robot")


# ---- mapper ----


def run_mapper(profile, state, seconds=2.0, hz=30):
    mapper = ExpressionMapper(load_profile(profile))
    out = []
    for i in range(int(seconds * hz)):
        out.append(mapper.step(state.model_copy(update={"t": i / hz})))
    return mapper, out


def test_mapper_respects_limits_and_rate_limits():
    profile = load_profile("quadruped")
    mapper = ExpressionMapper(profile)
    commands = [mapper.step(REST.model_copy(update={"t": 0.0}))]
    for i in range(1, 60):
        commands.append(mapper.step(FEAR.model_copy(update={"t": i / 30})))
    for name, channel in profile.channels.items():
        series = [c.channels[name] for c in commands]
        assert min(series) >= channel.limits[0] and max(series) <= channel.limits[1]
        assert max(abs(np.diff(series))) <= channel.rate_limit / 30 + 1e-9


def test_idle_keeps_a_resting_robot_moving_and_arousal_quickens_it():
    def movement(state):
        _, out = run_mapper("quadruped", state, seconds=10)
        series = [c.channels["body.height"] for c in out]
        return np.ptp(series), np.abs(np.diff(series)).sum()

    span, _ = movement(REST)
    assert span > 0.001  # never frozen
    excited = state_for("neutral", (0.0, 0.9, 0.0), intensity=0.0)
    assert movement(excited)[1] > movement(REST)[1]


def test_cue_moves_then_ends_and_is_reported_once():
    mapper = ExpressionMapper(load_profile("avatar"))
    base = [mapper.step(REST.model_copy(update={"t": i / 30})) for i in range(5)]
    assert mapper.trigger("nod", 5 / 30)
    out = [mapper.step(REST.model_copy(update={"t": (5 + i) / 30})) for i in range(60)]
    assert sum("nod" in c.cues for c in out) == 1
    series = [c.channels["head.pitch"] for c in out]
    assert max(np.abs(np.array(series))) > 0.2
    assert abs(series[-1]) < 0.1 and abs(base[-1].channels["head.pitch"]) < 0.1


def test_unknown_cue_is_ignored():
    mapper = ExpressionMapper(load_profile("voice"))
    assert mapper.trigger("nod", 0.0) is False
    assert mapper.step(REST).cues == []


def test_talking_moves_the_mouth_only_while_talking():
    mapper = ExpressionMapper(load_profile("avatar"))
    silent = [mapper.step(REST.model_copy(update={"t": i / 30})) for i in range(30)]
    assert max(c.channels["face.mouth_open"] for c in silent) == 0.0
    mapper.set_talking(True)
    talking = [mapper.step(REST.model_copy(update={"t": (30 + i) / 30})) for i in range(30)]
    assert max(c.channels["face.mouth_open"] for c in talking) > 0.3


def test_mapping_is_deterministic_for_replay():
    engine = AffectEngine(PRESETS["pet"])
    engine.submit(CATALOG["praise"].model_copy(update={"t": 0.5}))
    states = engine.run_until(5.0)
    assert remap(states, load_profile("quadruped")) == remap(states, load_profile("quadruped"))


# ---- adapter safety ----


def command(t=0.0, **channels):
    return ExpressionCommand(t=t, channels=channels, ttl_s=0.2)


def test_adapter_clamps_a_profile_that_asks_too_much():
    adapter = SimulatorAdapter.for_robot("quadruped")
    limit = adapter.capabilities()["body.height"]
    t = 0.0
    for _ in range(200):
        t += 1 / 30
        result = adapter.apply(ExpressionCommand(t=t, channels={"body.height": 99.0}, ttl_s=0.2))
    assert adapter.state()["body.height"] == pytest.approx(limit.max)
    assert "body.height" in result.clamped


def test_adapter_limits_are_tighter_than_the_profile():
    # A profile cannot widen a limit: the adapter's own numbers win.
    profile = load_profile("quadruped").channels
    limits = load_limits("quadruped")
    for name, limit in limits.items():
        assert limit.min >= profile[name].limits[0] and limit.max <= profile[name].limits[1]


def test_adapter_enforces_rate_limit_even_on_first_command():
    adapter = SimulatorAdapter.for_robot("quadruped")
    limit = adapter.capabilities()["body.height"]
    adapter.apply(command(t=100.0, **{"body.height": limit.max}))
    moved = abs(adapter.state()["body.height"] - limit.rest)
    assert moved <= limit.rate_limit * 0.2 + 1e-9


def test_stale_command_time_cannot_move_the_robot():
    adapter = SimulatorAdapter.for_robot("quadruped")
    adapter.apply(command(t=1.0, **{"body.height": 0.95}))
    before = adapter.state()["body.height"]
    adapter.apply(command(t=0.5, **{"body.height": 0.8}))  # older than the last one
    assert adapter.state()["body.height"] == before


def test_adapter_drops_unknown_channels():
    adapter = SimulatorAdapter.for_robot("quadruped")
    result = adapter.apply(command(**{"face.eye_open": 1.0, "body.height": 0.95}))
    assert result.dropped == ["face.eye_open"] and "body.height" in result.applied


def test_stop_refuses_commands_until_resume():
    adapter = SimulatorAdapter.for_robot("quadruped")
    limit = adapter.capabilities()["body.height"]
    adapter.apply(command(t=0.0, **{"body.height": 1.0}))
    adapter.stop()
    assert adapter.state()["body.height"] == limit.rest
    assert adapter.apply(command(t=1.0, **{"body.height": 1.0})).rejected == "stopped"
    assert adapter.state()["body.height"] == limit.rest
    adapter.resume()
    assert adapter.apply(command(t=2.0, **{"body.height": 1.0})).rejected is None


def test_expired_command_reads_as_rest():
    adapter = SimulatorAdapter.for_robot("quadruped")
    limit = adapter.capabilities()["body.height"]
    adapter.apply(command(t=0.0, **{"body.height": 1.0}))
    assert adapter.state(now=0.1)["body.height"] != limit.rest
    assert adapter.state(now=5.0)["body.height"] == limit.rest


def test_invalid_numbers_never_reach_an_adapter():
    with pytest.raises(ValidationError):
        command(**{"body.height": float("nan")})


def test_sdk_adapter_sends_only_safe_values():
    sent = []
    limits = load_limits("quadruped")
    adapter = SdkAdapter(limits, lambda v, c: sent.append(v))
    t = 0.0
    for _ in range(100):
        t += 1 / 30
        adapter.apply(command(t=t, **{"body.x": 5.0}))
    assert max(s["body.x"] for s in sent) <= limits["body.x"].max


# ---- week 16 gate: one core, two robots, config changes only ----


def drive(profile_name, adapter):
    engine = AffectEngine(PRESETS["pet"])
    mapper = ExpressionMapper(load_profile(profile_name))
    ExpressionLoop(engine, mapper, adapter.apply)
    engine.submit(CATALOG["praise"].model_copy(update={"t": 1.0}))
    engine.submit(CATALOG["threat"].model_copy(update={"t": 10.0}))
    states = engine.run_until(15.0)
    return states


def test_one_core_drives_avatar_and_quadruped_with_config_only():
    avatar = AvatarAdapter.from_profile(load_profile("avatar"))
    quadruped = SimulatorAdapter.for_robot("quadruped")
    a_states, q_states = drive("avatar", avatar), drive("quadruped", quadruped)
    assert a_states == q_states  # the emotion does not depend on the body
    assert set(avatar.state()) & set(quadruped.state()) == set()
    assert len(quadruped.history) == len(q_states)
    assert any(v["body.height"] < 0.95 for v in quadruped.history)  # the threat crouches it


def test_every_quadruped_value_stayed_inside_its_limits_under_a_barrage():
    adapter = SimulatorAdapter.for_robot("quadruped")
    engine = AffectEngine(PRESETS["pet"].model_copy(update={"gain": 4.0}))
    mapper = ExpressionMapper(load_profile("quadruped"))
    ExpressionLoop(engine, mapper, adapter.apply)
    names = list(CATALOG)
    for i in range(100):
        engine.submit(CATALOG[names[i % len(names)]].model_copy(update={"t": i * 0.1}))
        mapper.trigger("step_back", i * 0.1)
    engine.run_until(12.0)
    limits = adapter.capabilities()
    for values_at_tick in adapter.history:
        for name, value in values_at_tick.items():
            assert limits[name].min <= value <= limits[name].max


def test_a_logged_session_replays_on_another_robot_with_its_cues(tmp_path):
    from affect_core import SessionLogger
    from affect_express import remap_log

    log = tmp_path / "s.jsonl"
    engine = AffectEngine(PRESETS["pet"])
    logger = SessionLogger(log, engine)
    mapper = ExpressionMapper(load_profile("avatar"))
    avatar = AvatarAdapter.from_profile(load_profile("avatar"))
    ExpressionLoop(
        engine, mapper, avatar.apply,
        on_command=lambda c: logger.write("command", c.model_dump(mode="json")),
    )
    engine.submit(CATALOG["praise"].model_copy(update={"t": 0.5}))
    engine.run_until(1.0)
    mapper.trigger("nod", engine.t)
    engine.run_until(3.0)
    logger.close()
    commands = remap_log(log, load_profile("quadruped"))
    assert sum("nod" in c.cues for c in commands) == 1
    assert "body.height" in commands[0].channels and "face.eye_open" not in commands[0].channels
    assert remap_log(log, load_profile("quadruped")) == commands
