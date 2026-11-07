"""Run the affect server and dashboard. Binds to this machine only by default."""

from __future__ import annotations

import argparse

import uvicorn
from affect_core import PRESETS

from .app import create_app
from .motion import DEFAULT_PORT, DEFAULT_SPEED, REAL_SETTLE_S, create_arm, parse_motion_map


def main() -> None:
    parser = argparse.ArgumentParser(prog="affect-server")
    parser.add_argument("--personality", default="pet", choices=sorted(PRESETS))
    parser.add_argument("--profile", default="avatar", choices=["avatar", "quadruped"])
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--tick-hz", type=float, default=30.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--llm", default="scripted", help="scripted (default), claude or ollama")
    parser.add_argument("--llm-model", help="model name for the chosen backend")
    parser.add_argument("--effort", help="Claude effort level, default low")
    parser.add_argument("--speech", action="store_true", help="use the microphone and speaker")
    parser.add_argument("--stt", default="whisper", help="whisper (local, default) or onemin (API)")
    parser.add_argument(
        "--push-to-talk",
        action="store_true",
        help="with --speech: no always-on microphone; use the dashboard's record button",
    )
    parser.add_argument(
        "--motions", help="folder of arm recordings: play one for each emotion, holding it"
    )
    parser.add_argument(
        "--arm", default="mock", choices=["mock", "real"], help="mock (default) or the real arm"
    )
    parser.add_argument("--arm-port", default=DEFAULT_PORT, help="serial port of the real arm")
    parser.add_argument("--arm-speed", type=int, default=DEFAULT_SPEED)
    parser.add_argument(
        "--motion-map",
        default="joy=happy,sadness=sad,anger=angry,fear=disgust",
        help="engine label=recording type, comma separated",
    )
    parser.add_argument("--vad", default="silero", help="silero (default) or energy")
    parser.add_argument("--voices", default="models/piper", help="folder of Piper voices")
    parser.add_argument("--audio-device", help="sound device name or number")
    parser.add_argument("--persona", help="who the robot is, in a sentence")
    args = parser.parse_args()
    options = {}
    if args.llm_model:
        options["model"] = args.llm_model
    if args.effort:
        options["effort"] = args.effort
    device = args.audio_device
    if device is not None and device.isdigit():
        device = int(device)
    arm = create_arm(args.arm, args.arm_port) if args.motions else None
    app = create_app(
        args.personality,
        args.log_dir,
        args.tick_hz,
        profile=args.profile,
        llm=args.llm,
        llm_options=options,
        speech=args.speech,
        stt=args.stt,
        push_to_talk=args.push_to_talk,
        motions=args.motions,
        arm=arm,
        motion_map=parse_motion_map(args.motion_map),
        arm_speed=args.arm_speed,
        arm_settle_s=REAL_SETTLE_S if args.arm == "real" else 0.0,
        vad=args.vad,
        voices=args.voices,
        audio_device=device,
        persona=args.persona,
    )
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        if arm is not None:
            arm.disconnect()


if __name__ == "__main__":
    main()
