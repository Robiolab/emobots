"""Command line: run a scripted scenario to a log, or replay a log."""

from __future__ import annotations

import argparse
import sys

from .metrics import label_flips
from .personality import PRESETS
from .scenario import Scenario, check_expectations, run_scenario
from .session import replay


def _run(args: argparse.Namespace) -> int:
    scenario = Scenario.load(args.scenario)
    states = run_scenario(scenario, args.log)
    failures = check_expectations(scenario, states)
    unexplained = [f for f in label_flips(states) if not f.explained]
    print(f"{scenario.name}: {len(states)} ticks, final label {states[-1].label}")
    for failure in failures:
        print(f"  FAIL {failure}")
    for flip in unexplained:
        print(f"  FAIL t={flip.t:.2f}: unexplained flip {flip.before} -> {flip.after}")
    return 1 if failures or unexplained else 0


def _replay(args: argparse.Namespace) -> int:
    if args.preset:
        result = replay(args.log, personality=PRESETS[args.preset])
        print(f"counterfactual as {args.preset}: {result.compared} ticks")
        for flip in label_flips(result.states):
            print(f"  t={flip.t:8.2f}  {flip.before} -> {flip.after}  ({flip.cause})")
        return 0
    result = replay(args.log)
    if result.exact:
        print(f"exact: {result.compared} of {result.compared} states identical")
        return 0
    print(
        f"DIVERGED: {result.mismatches} of {result.compared} states differ, "
        f"first at tick {result.first_mismatch}"
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="affect-core")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a scenario and check its expectations")
    run.add_argument("scenario")
    run.add_argument("--log", help="write the session log here")
    run.set_defaults(func=_run)
    rep = commands.add_parser("replay", help="replay a session log and verify it exactly")
    rep.add_argument("log")
    rep.add_argument("--preset", choices=sorted(PRESETS), help="replay as another personality")
    rep.set_defaults(func=_replay)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
