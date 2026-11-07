"""FastAPI app: runs one engine in real time, logs it, streams it to the dashboard."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from affect_adapters import AvatarAdapter, SimulatorAdapter
from affect_core import (
    CATALOG,
    PRESETS,
    AffectEngine,
    AppraisalEvent,
    AppraisalMap,
    EngineConfig,
    ExpressionCommand,
    Personality,
    SessionLogger,
)
from affect_dash import STATIC_DIR
from affect_dialogue.conversation import Conversation, latency_summary
from affect_dialogue.runtime import load_backend
from affect_express import ExpressionLoop, ExpressionMapper, load_profile
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .motion import DEFAULT_SPEED, MotionLibrary, MotionPlayer, Movement

BROADCAST_HZ = 20.0
RECENT_EVENTS = 50
RECENT_TALK = 40
CLIENT_QUEUE = 256
TALK_KINDS = ("heard", "reply", "llm_error", "interrupted")
# While an arm movement plays, the emotion's time constant is this fraction of the mood's,
# which is slow enough that the emotion stays put for the length of the movement.
HOLD_TAU_FRACTION = 0.9


class ListeningRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    on: bool


class SayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=2000)
    language: str = Field(default="es", pattern="^(es|en)$")


class PersonalityUpdate(BaseModel):
    """Choose a preset, adjust the current personality, or both."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    preset: str | None = None
    gain: float | None = Field(default=None, gt=0.0)
    tau_emotion: float | None = Field(default=None, gt=0.0)
    tau_mood: float | None = Field(default=None, gt=0.0)


class EngineHost:
    """Owns the engine. All access happens on the event loop, so no locks are needed."""

    def __init__(
        self,
        personality: Personality,
        log_dir: str | Path,
        tick_hz: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        profile: str = "avatar",
    ) -> None:
        self.engine = AffectEngine(personality, EngineConfig(tick_hz=tick_hz))
        self.session = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.log_path = Path(log_dir) / f"{self.session}.jsonl"
        self.logger = SessionLogger(self.log_path, self.engine, session=self.session)
        self.recent: deque[dict[str, Any]] = deque(maxlen=RECENT_EVENTS)
        self.talk: deque[dict[str, Any]] = deque(maxlen=RECENT_TALK)
        self.turns: list[dict[str, Any]] = []
        self.conversation: Conversation | None = None
        self.microphone: Any = None
        self.microphone_factory: Callable[[Conversation], Any] | None = None
        self.motion: MotionPlayer | None = None
        self._armed = True  # a new emotion, or a new event, may start one movement
        self._held_from: Personality | None = None  # the personality to go back to
        self._quiet_personality = 0  # holds are internal: the dashboard need not see them
        self._release_at = 0.0
        self._acting: dict[str, Any] | None = None
        self._clients: set[asyncio.Queue[dict[str, Any]]] = set()
        self._clock = clock
        self._start = clock()
        self.engine.add_sink(self._on_engine)
        embodiment = load_profile(profile)
        self.profile = profile
        self.mapper = ExpressionMapper(embodiment)
        self.adapter = (
            AvatarAdapter.from_profile(embodiment)
            if profile == "avatar"
            else SimulatorAdapter.for_robot(profile)
        )
        self.loop = ExpressionLoop(
            self.engine, self.mapper, self.adapter.apply, on_command=self._on_command
        )

    def _on_command(self, command: ExpressionCommand) -> None:
        self.logger.write("command", command.model_dump(mode="json"))

    def on_conversation(self, kind: str, data: dict[str, Any]) -> None:
        """Log every conversation event, and show the dashboard the ones people read."""
        self.logger.write(kind, data)
        self.logger.flush()
        message = {"type": kind, **data}
        if kind in TALK_KINDS:
            self.talk.append(message)
        if kind == "turn":
            self.turns.append(data)
            message["summary"] = latency_summary(self.turns)
        self._broadcast(message)

    def _on_engine(self, kind: str, data: dict[str, Any]) -> None:
        if kind == "state":
            return  # states go out at the broadcast rate, from pump()
        if kind == "personality" and self._quiet_personality:
            self._quiet_personality -= 1
            self.logger.flush()
            return
        message = {"type": kind, **data}
        if kind == "event":
            self.recent.append(message)
            self._armed = True
        self.logger.flush()
        self._broadcast(message)
        if kind == "personality":
            self._broadcast(self.hello())

    def _broadcast(self, message: dict[str, Any]) -> None:
        for queue in self._clients:
            if queue.full():  # a slow client loses its oldest message, never the engine's time
                queue.get_nowait()
            queue.put_nowait(message)

    def hello(self) -> dict[str, Any]:
        return {
            "type": "hello",
            "session": self.session,
            "log": str(self.log_path),
            "tick_hz": self.engine.config.tick_hz,
            "personality": self.base_personality.model_dump(mode="json"),
            "state": self.engine.state.model_dump(mode="json"),
            "recent": list(self.recent),
            "talk": list(self.talk),
            "profile": self.profile,
            "expression": self.adapter.state(),
            "latency": latency_summary(self.turns),
            "voice": self.conversation is not None and self.conversation.stt is not None,
            "mic": self.microphone is not None,
            "can_listen": self.microphone_factory is not None,
            "motion": self.motion is not None,
            "acting": self._acting_now(),
        }

    def _acting_now(self) -> dict[str, Any] | None:
        if self._acting is None:
            return None
        remaining = max(0.0, self._acting["ends_at"] - self._clock())
        return {**{k: v for k, v in self._acting.items() if k != "ends_at"}, "remaining": remaining}

    def set_listening(self, on: bool) -> bool:
        """Switch the always-on microphone. Returns whether it is on afterwards."""
        if self.microphone_factory is None or self.conversation is None:
            raise RuntimeError("no speech: start the server with --speech")
        if on and self.microphone is None:
            microphone = self.microphone_factory(self.conversation)
            microphone.start()
            self.microphone = microphone
        elif not on and self.microphone is not None:
            self.microphone.close()
            self.microphone = None
            endpointer = self.conversation.endpointer
            if endpointer is not None:
                endpointer.reset()
        self._broadcast({"type": "listening", "on": self.microphone is not None})
        return self.microphone is not None

    @property
    def base_personality(self) -> Personality:
        """The personality the person chose, even while an emotion is being held."""
        return self._held_from or self.engine.personality

    def set_personality(self, personality: Personality) -> None:
        if self._held_from is None:
            self.engine.set_personality(personality)
        else:  # keep holding; the change takes effect when the hold ends
            self._held_from = personality

    def pump(self) -> None:
        self.engine.run_until(self._clock() - self._start)
        if self.motion is not None:
            self._drive_motion()

    def _drive_motion(self) -> None:
        state = self.engine.state
        if self._held_from is not None:
            if not self.motion.playing and self._clock() >= self._release_at:
                self._release_emotion()
            return
        if not self._armed:
            return
        movement = self.motion.choose(state.label, state.intensity)
        if movement is not None:
            self._armed = False
            self._start_motion(state.label, movement)

    def _start_motion(self, label: str, movement: Movement) -> None:
        window = self.motion.window_s(movement)
        self._hold_emotion()
        self._release_at = self._clock() + window
        self.motion.start(movement)
        info = {
            "emotion": label,
            "movement": movement.kind,  # not "type": that names the message
            "name": movement.name,
            "folder": movement.folder.name,
            "seconds": round(window, 1),
        }
        self._acting = {**info, "ends_at": self._release_at}
        self.on_conversation("motion", info)

    def _hold_emotion(self) -> None:
        base = self.engine.personality
        self._held_from = base
        self._quiet_personality += 1
        self.engine.set_personality(
            base.model_copy(update={"tau_emotion": base.tau_mood * HOLD_TAU_FRACTION})
        )

    def _release_emotion(self) -> None:
        base, self._held_from = self._held_from, None
        self._quiet_personality += 1
        self.engine.set_personality(base)
        self._acting = None
        self.on_conversation(
            "motion_end",
            {
                "error": self.motion.error,
                "arm_errors": dict(getattr(self.motion.arm, "errors", {})),
                "clamped": getattr(self.motion.arm, "clamped", 0),
            },
        )

    def stop_motion(self) -> None:
        """Abandon the movement and let the emotion decay again."""
        if self.motion is not None:
            self.motion.stop()
            self._release_at = 0.0

    async def run(self) -> None:
        interval = 1.0 / BROADCAST_HZ
        while True:
            self.pump()
            self.logger.flush()
            self._broadcast(
                {
                    "type": "state",
                    **self.engine.state.model_dump(mode="json"),
                    "expression": self.adapter.state(),
                }
            )
            await asyncio.sleep(interval)

    def connect(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(CLIENT_QUEUE)
        self._clients.add(queue)
        return queue

    def disconnect(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._clients.discard(queue)


async def _attach_speech(
    host: EngineHost, vad: str, voices: str, device: Any, stt: str = "whisper"
) -> dict[str, Any]:
    """Load the speech models and open the microphone and speaker."""
    from affect_dialogue.speech import Microphone, PiperTTS, SoundDeviceSink
    from affect_dialogue.vad import Endpointer

    stt = await asyncio.to_thread(load_backend, "stt", stt)
    tts = await asyncio.to_thread(PiperTTS, voices)
    parts = {
        "stt": stt,
        "tts": tts,
        "sink": SoundDeviceSink(device),
        "endpointer": Endpointer(load_backend("vad", vad)),
    }
    parts["microphone_factory"] = lambda c: Microphone(c.feed_audio, time.monotonic, device)
    return parts


def create_app(
    personality: str = "pet",
    log_dir: str | Path = "logs",
    tick_hz: float = 30.0,
    profile: str = "avatar",
    llm: str = "scripted",
    llm_options: dict[str, Any] | None = None,
    speech: bool = False,
    vad: str = "silero",
    voices: str = "models/piper",
    audio_device: Any = None,
    persona: str | None = None,
    stt: str = "whisper",
    listen: bool = False,
    motions: str | Path | None = None,
    arm: Any = None,
    motion_map: dict[str, str] | None = None,
    arm_speed: int = DEFAULT_SPEED,
    arm_settle_s: float = 0.0,
) -> FastAPI:
    if personality not in PRESETS:
        raise ValueError(f"unknown personality {personality!r}")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        host = EngineHost(PRESETS[personality], log_dir, tick_hz, profile=profile)
        app.state.host = host
        parts = await _attach_speech(host, vad, voices, audio_device, stt) if speech else {}
        microphone_factory = parts.pop("microphone_factory", None)
        options = {"persona": persona} if persona else {}
        host.conversation = Conversation(
            host.engine,
            load_backend("llm", llm, **(llm_options or {})),
            mapper=host.mapper,
            emit=host.on_conversation,
            **parts,
            **options,
        )
        host.microphone_factory = microphone_factory
        if microphone_factory is not None and listen:
            host.set_listening(True)
        if motions is not None:
            library = MotionLibrary(motions)
            if not library.counts():
                raise RuntimeError(f"no recordings found in {motions}")
            host.motion = MotionPlayer(
                library, arm, motion_map, arm_speed, settle_s=arm_settle_s
            )
        task = asyncio.create_task(host.run())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            host.stop_motion()
            if host.microphone is not None:
                host.microphone.close()
                host.microphone = None
            await host.conversation.aclose()
            if hasattr(host.conversation.sink, "close"):
                host.conversation.sink.close()
            host.logger.close()

    app = FastAPI(title="Affect server", lifespan=lifespan)

    def host_of() -> EngineHost:
        return app.state.host

    @app.get("/api/catalog")
    async def catalog() -> dict[str, Any]:
        weights = AppraisalMap()
        return {
            "events": {
                name: {
                    "cause": event.cause,
                    "appraisal": event.model_dump(mode="json"),
                    "impulse": [float(x) for x in weights.impulse(event)],
                }
                for name, event in CATALOG.items()
            },
            "presets": {name: p.model_dump(mode="json") for name, p in PRESETS.items()},
            "prototypes": host_of().engine.config.prototypes,
        }

    @app.get("/api/state")
    async def state() -> dict[str, Any]:
        return host_of().hello()

    @app.post("/api/event/{name}")
    async def fire_named(name: str) -> dict[str, Any]:
        if name not in CATALOG:
            raise HTTPException(404, f"unknown event {name!r}")
        host_of().engine.submit(CATALOG[name])
        return {"queued": name}

    @app.post("/api/event")
    async def fire_custom(event: AppraisalEvent) -> dict[str, Any]:
        # Stamped by the engine on arrival, so a client clock can never reorder a session.
        host_of().engine.submit(event.model_copy(update={"t": 0.0}))
        return {"queued": event.cause}

    @app.post("/api/personality")
    async def set_personality(update: PersonalityUpdate) -> dict[str, Any]:
        if update.preset is not None and update.preset not in PRESETS:
            raise HTTPException(404, f"unknown preset {update.preset!r}")
        base = PRESETS[update.preset] if update.preset else host_of().base_personality
        fields = {}
        if update.gain is not None:
            fields["gain"] = update.gain
        if update.tau_emotion is not None:
            fields["tau_emotion"] = update.tau_emotion
        try:
            data = base.model_dump()
            data.update(fields)
            changed = Personality.model_validate(data)
            if update.tau_mood is not None:
                changed = Personality.model_validate(
                    changed.with_tau_mood(update.tau_mood).model_dump()
                )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        host_of().set_personality(changed)
        return changed.model_dump(mode="json")

    @app.post("/api/say")
    async def say(request: SayRequest) -> dict[str, Any]:
        """A typed turn. It runs through the same pipeline as speech."""
        conversation = host_of().conversation
        conversation.say(request.text, request.language)
        return {"queued": True}

    @app.post("/api/listening")
    async def listening(request: ListeningRequest) -> dict[str, Any]:
        """Switch the microphone on or off, from the dashboard's Escuchar button."""
        try:
            return {"listening": host_of().set_listening(request.on)}
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/stop")
    async def stop() -> dict[str, Any]:
        """Software stop: the robot goes to rest and refuses commands until resumed.

        This is no substitute for the hardware e-stop, which bypasses the engine.
        """
        host_of().adapter.stop()
        host_of().stop_motion()
        host_of().conversation.silence()  # an operator's stop is not the person being rude
        return {"stopped": True}

    @app.post("/api/resume")
    async def resume() -> dict[str, Any]:
        host_of().adapter.resume()
        return {"stopped": False}

    @app.post("/api/reset")
    async def reset() -> dict[str, Any]:
        host_of().engine.reset()
        return {"reset": True}

    @app.websocket("/ws")
    async def stream(socket: WebSocket) -> None:
        await socket.accept()
        host = host_of()
        queue = host.connect()
        try:
            await socket.send_json(host.hello())
            while True:
                await socket.send_json(await queue.get())
        except WebSocketDisconnect:
            pass
        finally:
            host.disconnect(queue)

    @app.get("/")
    async def index() -> FileResponse:
        # The dashboard changes with the code: make the browser ask again on every reload.
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
