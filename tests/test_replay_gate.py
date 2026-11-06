"""Phase 0 gate: replay reproduces scripted scenarios exactly."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from affect_core import (
    CATALOG,
    PRESETS,
    AffectEngine,
    Scenario,
    SessionLogger,
    check_expectations,
    label_flips,
    read_log,
    replay,
    run_scenario,
)

SCENARIOS = sorted((Path(__file__).parent.parent / "scenarios").glob("*.json"))


def test_scenarios_exist():
    assert len(SCENARIOS) >= 4


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_scenario_meets_expectations(path):
    scenario = Scenario.load(path)
    states = run_scenario(scenario)
    assert check_expectations(scenario, states) == []


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_no_label_flip_without_a_logged_cause(path):
    states = run_scenario(Scenario.load(path))
    assert [f for f in label_flips(states) if not f.explained] == []


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_gate_replay_is_exact(path, tmp_path):
    log = tmp_path / "session.jsonl"
    states = run_scenario(Scenario.load(path), log)
    result = replay(log)
    assert result.exact, f"first mismatch at tick {result.first_mismatch}"
    assert result.compared == len(states)
    assert result.states == states


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_gate_replay_is_exact_across_processes(path, tmp_path):
    log = tmp_path / "session.jsonl"
    cli = [sys.executable, "-m", "affect_core"]
    command = [*cli, "run", str(path), "--log", str(log)]
    run = subprocess.run(command, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    rep = subprocess.run([*cli, "replay", str(log)], capture_output=True, text=True)
    assert rep.returncode == 0, rep.stdout + rep.stderr
    assert rep.stdout.startswith("exact:")


def test_replay_detects_a_tampered_log(tmp_path):
    log = tmp_path / "session.jsonl"
    run_scenario(Scenario.load(SCENARIOS[0]), log)
    lines = log.read_text().splitlines()
    record = json.loads(lines[100])
    assert record["kind"] == "state"
    record["data"]["emotion"][0] += 1e-12
    lines[100] = json.dumps(record)
    log.write_text("\n".join(lines) + "\n")
    result = replay(log)
    assert not result.exact and result.mismatches == 1


def test_replay_covers_live_personality_changes_and_resets(tmp_path):
    log = tmp_path / "session.jsonl"
    engine = AffectEngine(PRESETS["pet"])
    with SessionLogger(log, engine):
        engine.submit(CATALOG["praise"].model_copy(update={"t": 0.5}))
        engine.run_until(2.0)
        engine.set_personality(PRESETS["shy"])
        engine.submit(CATALOG["threat"].model_copy(update={"t": 0.0}))  # stale stamp
        engine.run_until(4.0)
        engine.reset()
        engine.run_until(5.0)
    assert replay(log).exact
    kinds = {r["kind"] for r in read_log(log)}
    assert {"session", "event", "personality", "reset", "state"} <= kinds


def test_counterfactual_replay_with_another_personality_differs(tmp_path):
    log = tmp_path / "session.jsonl"
    states = run_scenario(Scenario.load(SCENARIOS[0]), log)
    result = replay(log, personality=PRESETS["shy"])  # the first scenario is not shy
    assert not result.exact
    assert len(result.states) == len(states)
    assert result.states[-1].baseline == PRESETS["shy"].baseline


def test_read_log_tolerates_only_a_cut_off_last_line(tmp_path):
    log = tmp_path / "session.jsonl"
    log.write_text('{"a": 1}\n{"b"', encoding="utf-8")
    assert list(read_log(log)) == [{"a": 1}]
    log.write_text('{"a"\n{"b": 1}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        list(read_log(log))
