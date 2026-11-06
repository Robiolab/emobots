"""Latency bench, and a typed chat that needs no audio devices."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import numpy as np
from affect_core import PRESETS, AffectEngine
from affect_express import ExpressionMapper, load_profile

from .conversation import Conversation, latency_summary
from .runtime import load_backend, run_engine
from .speech import NullSink, PiperTTS, Prosody, WhisperSTT
from .vad import FRAME_SAMPLES, SAMPLE_RATE, Endpointer

UTTERANCES = {
    "en": [
        "Hello robot, you did a really great job today.",
        "What is your favorite thing about this lab?",
        "I don't think that was fair at all.",
        "Could you tell me how your day has been so far?",
        "Thank you for helping me with the experiment.",
    ],
    "es": [
        "Hola robot, hoy hiciste un trabajo muy bueno.",
        "¿Qué es lo que más te gusta de este laboratorio?",
        "No creo que eso haya sido justo para nada.",
        "¿Me cuentas cómo ha estado tu día hasta ahora?",
        "Gracias por ayudarme con el experimento de hoy.",
    ],
}


def _llm_kwargs(args: argparse.Namespace) -> dict:
    if args.llm == "scripted":
        return {"first_token_s": args.simulated_first_token}
    return {"model": args.model} if args.model else {}


def _resample(samples: np.ndarray, rate: int) -> np.ndarray:
    if rate == SAMPLE_RATE:
        return samples
    positions = np.arange(0, len(samples) - 1, rate / SAMPLE_RATE)
    return np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)


def add_noise(clip: np.ndarray, snr_db: float | None, rng: np.random.Generator) -> np.ndarray:
    """White noise at a signal-to-noise ratio measured against the loud parts of the clip."""
    if snr_db is None:
        return clip
    loud = clip[np.abs(clip) > 0.01]
    power = float(np.mean(np.square(loud))) if len(loud) else 1e-4
    noise = rng.normal(0.0, np.sqrt(power / 10 ** (snr_db / 10)), len(clip))
    return np.clip(clip + noise, -1.0, 1.0).astype(np.float32)


async def feed_realtime(conversation: Conversation, clip: np.ndarray) -> None:
    """Feed audio at real speed, as a microphone would."""
    start = time.monotonic()
    for i in range(len(clip) // FRAME_SAMPLES):
        due = start + (i + 1) * FRAME_SAMPLES / SAMPLE_RATE
        await asyncio.sleep(max(0.0, due - time.monotonic()))
        frame = clip[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
        conversation.feed_audio(frame, time.monotonic())


def speak_clip(tts, line: str, language: str, pad_s: float = 0.6) -> np.ndarray:
    silence = np.zeros(int(pad_s * SAMPLE_RATE), dtype=np.float32)
    pieces = [_resample(s, r) for s, r in tts.synthesize(line, language, Prosody())]
    return np.concatenate([silence, *pieces, silence])


async def run_bench(
    llm,
    stt,
    tts,
    vad,
    turns: int = 5,
    snr_db: float | None = None,
    emit=None,
) -> list[dict]:
    """Speak the test lines through the whole pipeline. Returns one report per turn."""
    engine = AffectEngine(PRESETS["pet"])
    mapper = ExpressionMapper(load_profile("avatar"))
    engine.add_sink(lambda kind, _: mapper.step(engine.state) if kind == "state" else None)
    reports: list[dict] = []

    def on_event(kind: str, data: dict) -> None:
        if kind == "turn":
            reports.append(data)
        if emit is not None:
            emit(kind, data)

    conversation = Conversation(
        engine,
        llm,
        mapper=mapper,
        stt=stt,
        tts=tts,
        sink=NullSink(),
        endpointer=Endpointer(vad),
        emit=on_event,
    )
    rng = np.random.default_rng(7)
    ticker = asyncio.create_task(run_engine(engine))
    try:
        for language, lines in UTTERANCES.items():
            for line in lines[:turns]:
                clip = add_noise(speak_clip(tts, line, language), snr_db, rng)
                await feed_realtime(conversation, clip)
                await conversation.wait()
    finally:
        ticker.cancel()
    return reports


async def _bench(args: argparse.Namespace) -> int:
    print("loading speech models...", flush=True)
    stt = WhisperSTT()
    tts = PiperTTS(args.voices)
    vad = load_backend("vad", args.vad)
    print(f"stt device: {stt.device}; vad: {args.vad}; llm: {args.llm}", flush=True)
    llm = load_backend("llm", args.llm, **_llm_kwargs(args))
    reports = await run_bench(llm, stt, tts, vad, args.turns, args.snr_db)
    summary = latency_summary(reports)
    if args.json:
        print(json.dumps({"turns": reports, "summary": summary}, indent=2))
    else:
        for language, row in summary.items():
            print(json.dumps({"language": language, **row}))
    passed = bool(summary) and all(
        r["pass_reaction"] and r["pass_first_word"] for r in summary.values()
    )
    print("gate targets met" if passed else "gate targets NOT met")
    return 0 if passed else 1


async def _eval(args: argparse.Namespace) -> int:
    from .evaluation import run_eval

    state = AffectEngine(PRESETS["pet"]).state
    cues = list(load_profile("avatar").cues)
    llm = load_backend("llm", args.llm, **_llm_kwargs(args))
    results = await run_eval(llm, args.cases, state, cues)
    for result in results:
        mark = "FAIL" if result.failures else "ok  "
        print(f"{mark} {result.id:14} {result.first_sentence_ms} ms | {result.reply[:100]}")
        for failure in result.failures:
            print(f"       - {failure}")
    passed = sum(not r.failures for r in results)
    print(f"{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


async def _chat(args: argparse.Namespace) -> int:
    engine = AffectEngine(PRESETS[args.personality])
    conversation = Conversation(
        engine,
        load_backend("llm", args.llm, **_llm_kwargs(args)),
        mapper=ExpressionMapper(load_profile("avatar")),
        emit=lambda kind, data: print(f"  robot: {data['text']}") if kind == "reply" else None,
        default_language=args.language,
    )
    ticker = asyncio.create_task(run_engine(engine))
    try:
        while True:
            text = await asyncio.to_thread(input, "you: ")
            if not text.strip():
                continue
            await conversation.say(text)
            state = engine.state
            print(f"  [{state.label} {state.intensity:.0%}; cause: {state.cause}]")
    except (EOFError, KeyboardInterrupt):
        return 0
    finally:
        ticker.cancel()


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="affect-dialogue")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, func in (("bench", _bench), ("chat", _chat), ("eval", _eval)):
        sub = commands.add_parser(name)
        sub.add_argument("--llm", default="scripted")
        sub.add_argument("--model")
        sub.add_argument("--simulated-first-token", type=float, default=0.0)
        sub.set_defaults(func=func)
    bench = commands.choices["bench"]
    bench.add_argument("--turns", type=int, default=5)
    bench.add_argument("--snr-db", type=float, help="add white noise at this level")
    bench.add_argument("--vad", default="silero")
    bench.add_argument("--voices", default="models/piper")
    bench.add_argument("--json", action="store_true")
    commands.choices["eval"].add_argument("--cases", default="evals/dialogue_cases.json")
    chat = commands.choices["chat"]
    chat.add_argument("--personality", default="pet", choices=sorted(PRESETS))
    chat.add_argument("--language", default="es", choices=["es", "en"])
    args = parser.parse_args(argv)
    return asyncio.run(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
