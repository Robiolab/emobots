"""Run the affect server and dashboard. Binds to this machine only by default."""

from __future__ import annotations

import argparse

import uvicorn
from affect_core import PRESETS

from .app import create_app


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
    app = create_app(
        args.personality,
        args.log_dir,
        args.tick_hz,
        profile=args.profile,
        llm=args.llm,
        llm_options=options,
        speech=args.speech,
        vad=args.vad,
        voices=args.voices,
        audio_device=device,
        persona=args.persona,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
