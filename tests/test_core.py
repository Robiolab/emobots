"""Dynamics, personality, labels and safety caps, tested with scripted events."""

import math

import numpy as np
import pytest
from affect_core import (
    CATALOG,
    PRESETS,
    AffectEngine,
    AppraisalEvent,
    AppraisalMap,
    EngineConfig,
    ExpressionCommand,
    Personality,
    label_flips,
)
from affect_core.labels import INTENSITY_NORM
from pydantic import ValidationError

FLAT = Personality(baseline=(0.0, 0.0, 0.0), gain=1.0, tau_emotion=10.0, tau_mood=600.0, alpha=0.0)


def at(event: str, t: float) -> AppraisalEvent:
    return CATALOG[event].model_copy(update={"t": t})


# ---- messages ----


def test_plan_example_messages_validate():
    AppraisalEvent(
        t=12.40, source="dialogue", cause="user praised the robot", relevance=0.7,
        congruence=0.8, novelty=0.3, agency="user", control=0.5, norm=0.6,
    )
    ExpressionCommand(
        t=12.45, channels={"voice.rate": 1.12, "body.height": 0.8}, cues=["nod"], ttl_s=0.2
    )


@pytest.mark.parametrize(
    "bad", [{"relevance": 1.5}, {"congruence": -2}, {"norm": float("nan")}, {"agency": "robot"},
            {"mystery": 1}, {"v": 2}]
)
def test_invalid_events_are_rejected(bad):
    good = dict(cause="x", relevance=0.5, congruence=0, novelty=0, control=0, norm=0)
    with pytest.raises(ValidationError):
        AppraisalEvent(**{**good, **bad})


# ---- appraisal ----


def test_default_map_matches_test_bench_formula():
    a = CATALOG["threat"]
    expected = [
        a.relevance * (0.7 * a.congruence + 0.3 * a.norm),
        a.relevance * (0.8 * a.novelty + 0.3 * abs(a.congruence)),
        a.relevance * (0.7 * a.control + 0.3 * a.congruence),
    ]
    assert AppraisalMap().impulse(a) == pytest.approx(expected)


def test_irrelevant_event_has_no_effect():
    event = CATALOG["threat"].model_copy(update={"relevance": 0.0})
    assert np.all(AppraisalMap().impulse(event) == 0.0)


# ---- dynamics ----


def test_emotion_decays_exponentially_toward_fixed_mood():
    # With alpha = 0 and mood on baseline, E(t) = E0 exp(-t / tau) in closed form.
    engine = AffectEngine(FLAT, EngineConfig(tick_hz=50))
    engine.submit(at("praise", 0.0))
    engine.step()
    e0 = np.array(engine.state.emotion)
    t0 = engine.t
    engine.run_until(t0 + 10.0)
    expected = e0 * math.exp(-(engine.t - t0) / FLAT.tau_emotion)
    assert np.array(engine.state.emotion) == pytest.approx(expected, rel=1e-9)
    assert np.array(engine.state.mood) == pytest.approx(np.zeros(3))


def test_exact_integration_matches_fine_euler():
    p = PRESETS["pet"].with_tau_mood(60.0)
    engine = AffectEngine(p, EngineConfig(tick_hz=20))
    engine.submit(at("pet", 0.0))
    engine.step()
    base = np.array(p.baseline)
    e, m = np.array(engine.state.emotion) - base, np.array(engine.state.mood) - base
    start = engine.t
    engine.run_until(start + 5.0)
    h = 1e-4
    for _ in range(round((engine.t - start) / h)):
        e, m = e + h * (-p.k_e * (e - m)), m + h * (-p.k_m * m + p.alpha * (e - m))
    assert np.array(engine.state.emotion) - base == pytest.approx(e, abs=1e-3)
    assert np.array(engine.state.mood) - base == pytest.approx(m, abs=1e-3)


def test_trajectory_does_not_depend_on_tick_rate():
    def final(hz):
        engine = AffectEngine(PRESETS["pet"], EngineConfig(tick_hz=hz))
        engine.submit(at("praise", 1.0))
        engine.submit(at("pet", 2.5))
        engine.run_until(20.0)
        return engine.state

    a, b = final(20), final(50)
    assert a.t == pytest.approx(b.t)
    assert a.emotion == pytest.approx(b.emotion, abs=1e-9)
    assert a.mood == pytest.approx(b.mood, abs=1e-9)


def test_event_applies_at_its_own_time_inside_a_tick():
    a = AffectEngine(FLAT, EngineConfig(tick_hz=20))
    a.submit(at("praise", 0.01))
    a.step()
    b = AffectEngine(FLAT, EngineConfig(tick_hz=20))
    b.submit(at("praise", 0.04))
    b.step()
    assert np.linalg.norm(b.state.emotion) > np.linalg.norm(a.state.emotion)


def test_events_in_same_tick_apply_in_time_order_not_arrival_order():
    def run(order):
        engine = AffectEngine(PRESETS["pet"], EngineConfig(tick_hz=20))
        for name, t in order:
            engine.submit(at(name, t))
        engine.step()
        return engine.state

    forward = run([("praise", 0.01), ("threat", 0.02)])
    backward = run([("threat", 0.02), ("praise", 0.01)])
    assert forward == backward


def test_past_event_is_applied_now_and_logged_with_real_time():
    engine = AffectEngine(FLAT)
    engine.run_until(1.0)
    seen = []
    engine.add_sink(lambda kind, data: seen.append(data) if kind == "event" else None)
    engine.submit(at("praise", 0.0))
    engine.step()
    assert seen[0]["event"]["t"] == pytest.approx(1.0)


def test_mood_follows_sustained_emotion_then_returns_to_baseline():
    p = PRESETS["assistant"].with_tau_mood(60.0)
    engine = AffectEngine(p)
    for t in range(0, 30, 3):
        engine.submit(at("praise", float(t)))
    engine.run_until(30.0)
    raised = engine.state.mood[0]
    assert raised > p.baseline[0] + 0.1
    engine.run_until(600.0)
    assert engine.state.mood == pytest.approx(p.baseline, abs=1e-2)
    assert engine.state.emotion == pytest.approx(p.baseline, abs=1e-2)


def test_state_stays_in_the_cube_under_a_barrage():
    engine = AffectEngine(PRESETS["pet"].model_copy(update={"gain": 5.0}))
    for i in range(200):
        engine.submit(at(list(CATALOG)[i % len(CATALOG)], i * 0.01))
    for state in engine.run_until(5.0):
        assert all(-1.0 <= x <= 1.0 for x in (*state.emotion, *state.mood))


def test_delta_is_change_since_last_tick():
    engine = AffectEngine(FLAT)
    engine.submit(at("praise", 0.0))
    first = engine.step()
    second = engine.step()
    expected = np.array(second.emotion) - np.array(first.emotion)
    assert np.array(second.delta) == pytest.approx(expected)


def test_tick_rate_must_be_in_plan_range():
    with pytest.raises(ValidationError):
        EngineConfig(tick_hz=100)


# ---- personality ----


def test_mood_must_be_slower_than_emotion():
    with pytest.raises(ValidationError):
        Personality(tau_emotion=30.0, tau_mood=10.0)


def test_low_neuroticism_is_more_pleasant_and_calmer():
    # Mehrabian's factor is emotional stability, so the sign must flip for neuroticism.
    p = Personality.from_big_five(neuroticism=-0.8).baseline
    assert p[0] > 0 and p[1] < 0


def test_big_five_directions():
    assert Personality.from_big_five(agreeableness=0.8).baseline[0] > 0
    assert Personality.from_big_five(extraversion=0.8).baseline[2] > 0
    assert Personality.from_big_five(neuroticism=0.8).baseline[1] > 0
    assert Personality.from_big_five().baseline == (0.0, 0.0, 0.0)


def test_big_five_time_constants_stay_in_plan_range():
    for n in (-1.0, 0.0, 1.0):
        p = Personality.from_big_five(neuroticism=n)
        assert 5.0 <= p.tau_emotion <= 30.0
        assert 300.0 <= p.tau_mood <= 1800.0


def test_big_five_rejects_out_of_range():
    with pytest.raises(ValueError):
        Personality.from_big_five(openness=2.0)


def test_presets_start_neutral_for_themselves():
    for preset in PRESETS.values():
        assert AffectEngine(preset).state.label == "neutral"


def test_with_tau_mood_keeps_coupling_ratio():
    p = PRESETS["pet"]
    q = p.with_tau_mood(60.0)
    assert q.alpha * q.tau_mood == pytest.approx(p.alpha * p.tau_mood)


# ---- labels and causes ----


@pytest.mark.parametrize(
    "preset,event,label",
    [("pet", "praise", "joy"), ("pet", "threat", "fear"), ("grumpy", "insult", "anger"),
     ("shy", "scold", "sadness")],
)
def test_events_read_out_as_expected_label(preset, event, label):
    engine = AffectEngine(PRESETS[preset])
    engine.submit(at(event, 0.0))
    assert engine.step().label == label


def test_cause_is_strongest_remaining_event_and_expires():
    engine = AffectEngine(PRESETS["pet"])
    engine.submit(at("threat", 0.0))
    engine.submit(at("pet", 0.01))
    assert engine.step().cause == "a threat"
    engine.run_until(300.0)
    assert engine.state.cause is None


def test_no_cause_before_any_event():
    assert AffectEngine(PRESETS["pet"]).state.cause is None


# ---- safety caps ----


def test_fear_and_anger_intensity_are_capped():
    for preset, event in (("pet", "threat"), ("grumpy", "insult")):
        engine = AffectEngine(PRESETS[preset].model_copy(update={"gain": 3.0}))
        limit = engine.config.caps.max_intensity
        for i in range(5):
            engine.submit(at(event, i * 0.1))
        for state in engine.run_until(5.0):
            if state.label in ("fear", "anger"):
                assert state.intensity <= limit + 1e-9
                assert np.linalg.norm(state.emotion) <= limit * INTENSITY_NORM + 1e-9


def test_joy_is_not_capped():
    engine = AffectEngine(PRESETS["pet"])
    engine.submit(at("praise", 0.0))
    assert engine.step().intensity > engine.config.caps.max_intensity


def test_fear_decays_faster_than_uncapped_emotion():
    def remaining(boost):
        cfg = EngineConfig(caps={"decay_boost": boost, "max_intensity": 1.0})
        engine = AffectEngine(FLAT, cfg)
        engine.submit(at("threat", 0.0))
        engine.step()
        start = np.linalg.norm(engine.state.emotion)
        engine.run_until(3.0)
        return np.linalg.norm(engine.state.emotion) / start

    assert remaining(2.0) < remaining(1.0)


# ---- stability metric ----


def test_quiet_engine_never_flips():
    engine = AffectEngine(PRESETS["pet"])
    assert label_flips(engine.run_until(120.0)) == []


def test_flip_detected_and_explained_by_cause():
    engine = AffectEngine(PRESETS["pet"])
    engine.submit(at("praise", 1.0))
    flips = label_flips(engine.run_until(30.0))
    assert flips and all(f.explained for f in flips)
    assert flips[0].after == "joy" and flips[0].cause == "you praised it"


def test_big_five_baseline_leaves_room_to_react():
    # The plan's own example must rest well inside the cube, like the presets do.
    p = Personality.from_big_five(extraversion=0.6, agreeableness=0.7, neuroticism=-0.5)
    assert p.baseline[0] > 0
    assert max(abs(x) for x in p.baseline) < 0.6
    engine = AffectEngine(p)
    engine.submit(at("praise", 0.0))
    assert engine.step().label == "joy"


def test_a_threat_reads_as_fear_for_a_while_not_an_instant():
    for preset in PRESETS.values():
        engine = AffectEngine(preset)
        engine.submit(at("threat", 0.0))
        engine.run_until(1.5)
        assert engine.state.label == "fear", preset.name


def test_capping_keeps_the_emotion_it_caps():
    engine = AffectEngine(PRESETS["pet"].model_copy(update={"gain": 5.0}))
    engine.submit(at("threat", 0.0))
    state = engine.step()
    assert state.label == "fear" and state.intensity <= engine.config.caps.max_intensity + 1e-9


def test_repeated_friendliness_leaves_room_before_the_edge():
    engine = AffectEngine(PRESETS["pet"])
    for i in range(3):
        engine.submit(at("pet", float(i)))
    engine.run_until(3.0)
    assert engine.state.emotion[0] < 0.95  # hard clipping would pin this near 1


def test_a_negative_event_counts_in_full_even_at_high_joy():
    high = AffectEngine(PRESETS["pet"])
    for i in range(4):
        high.submit(at("praise", i * 0.5))
    high.run_until(2.0)
    before = high.state.emotion[0]
    high.submit(at("insult", 2.0))
    high.step()
    impulse = PRESETS["pet"].gain * AppraisalMap().impulse(CATALOG["insult"])
    assert before - high.state.emotion[0] == pytest.approx(-impulse[0], abs=0.05)


def test_saturation_can_be_turned_off():
    def after(soft):
        engine = AffectEngine(PRESETS["pet"], EngineConfig(soft_saturation=soft))
        engine.submit(at("praise", 0.0))
        return engine.step().emotion[0]

    assert after(True) < after(False)
