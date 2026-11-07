# Robot Emotions Engine

A robot-agnostic affect engine. The engine holds an explicit Pleasure–Arousal–Dominance state; an LLM appraises events and talks, but does not own the emotion. Each robot joins through a YAML profile and a small adapter. See the [development plan](https://claude.ai/artifact/PWATJS7sjY3jHGofB8TC8w).

## Status

| Phase | State | What is missing |
| --- | --- | --- |
| 0 Foundations | **Gate passes.** Replay reproduces scenarios exactly | |
| 1 Conversation loop | **Built, and meets its targets on synthetic speech.** The gate itself needs real people speaking | A run with human speakers and a real microphone |
| 2 Embodiment | **Software built.** One core drives the avatar and a simulated quadruped with config changes only. The ROS 2 node is verified on ROS 2 Jazzy | A real robot; its real limits; the ethics application |
| 3 Perception, memory | Not started. Only the short conversation window exists | |
| 4 Study and release | Not started | |

## Run it

Needs [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest                       # no models, GPU or network needed
uv run python -m affect_server      # dashboard at http://127.0.0.1:8765, typed chat
```

That default uses `scripted`, a keyword stand-in for the LLM, so everything runs offline.

If you have ROS 2 sourced in your shell, `pytest` can fail at startup with `No module named 'lark'`: ROS puts its own pytest plugins on `PYTHONPATH`. Run it without that variable:

```sh
env -u PYTHONPATH uv run pytest
```

To use a real model:

```sh
uv run python -m affect_server --llm ollama --llm-model qwen2.5:7b   # local, the tested one
uv sync --extra claude                                               # then set ANTHROPIC_API_KEY
uv run --extra claude python -m affect_server --llm claude           # cloud, text only
```

To use the 1min.ai speech-to-text API as the voice input instead of local Whisper (the audio leaves this machine, and each utterance is a batch request, so it is slower):

```sh
export ONEMIN_API_KEY=...                                            # your 1min.ai key
uv run --extra speech --with-editable ../1minai/python python -m affect_server --speech --stt onemin
```

With `--speech` the microphone starts off. The dashboard's **Escuchar** button switches it on and off (`--listen` starts it on). While it is on, a voice activity detector finds each utterance, sends it to the chosen STT backend, and the transcript runs like a typed turn. Anyone talking in the room counts as the person talking, so switch it off when you are not addressing the robot.

For a fast reply, `--llm onemin --llm-model gpt-5.4-nano` uses the 1min.ai chat API (about 1 to 2 s; the text of the conversation leaves this machine). It needs `ONEMIN_API_KEY` and `--with-editable ../1minai/python`.

To move an arm with the robot's emotion, point `--motions` at the `movements` project's recordings (one folder per recording, typed by its first emotion). When the robot enters `joy`, `sadness`, `anger` or `fear` it plays one random `happy`, `sad`, `angry` or `disgust` recording (the engine has no disgust label, so fear plays those) and holds the emotion for the length of the movement (about 8 s); it decays normally afterwards. `interest` plays one random recording of any type, and no emotion repeats the recording it played last time. Only a new event starts a movement, so a fading emotion does not. Change the pairing with `--motion-map` (`any` means every type). The dashboard shows an "Ejecutando acción" window while the arm moves.

```sh
uv run --extra speech --with-editable ../movements python -m affect_server --motions ../movements/recordings            # simulated arm
uv run --extra speech --with-editable ../movements python -m affect_server --motions ../movements/recordings --arm real --arm-port /dev/ttyACM0   # the real arm
```

The real arm is opt-in and moves for real: it goes to the recording's first pose, then plays at `--arm-speed 20`. Nothing checks joint limits, and `/api/stop` abandons the movement but cannot brake the arm, so keep the hardware e-stop within reach. Only one process may own each serial port. To drive several arms together, list their ports separated by commas (`--arm-port /dev/ttyACM0,/dev/ttyACM1,/dev/ttyACM2`): each gets the same command at the same time, and one failing does not stop the others. `/dev/serial/by-id/...` paths keep each arm's name when the `ttyACM` numbers change.

To talk out loud, install the speech models once, then start with `--speech`:

```sh
uv sync --extra speech --extra gpu
uv run --extra speech python -m piper.download_voices --data-dir models/piper en_US-lessac-medium es_MX-claude-high
uv run --extra speech --extra gpu python -m affect_server --speech --llm ollama --llm-model qwen2.5:7b
```

Use a headset if you can. An echo gate keeps the robot from interrupting itself on a loudspeaker, but it has only been tested on simulated echo, not in a real room.

### Phase 1 latency gate

```sh
uv run --extra speech --extra gpu python -m affect_dialogue bench                # stand-in LLM, shows the speech stack alone
uv run --extra speech --extra gpu python -m affect_dialogue bench --llm ollama --model qwen2.5:7b
uv run --extra speech --extra gpu pytest -m speech                               # real models, end to end
```

The bench speaks test lines with Piper and feeds them at microphone speed. Measured on an RTX 5070 with `qwen2.5:7b`, over 30 turns, with nothing else running:

| Median | English | Spanish | Target |
| --- | --- | --- | --- |
| First reaction | 313 ms | 328 ms | under 1,000 ms |
| First word, before the sound hardware | 1,063 ms | 1,188 ms | under 1,500 ms |

The sound hardware adds about 91 ms, which puts the slowest turn at about 1,404 ms. Turns reached 5 to 11 s when another heavy job shared the machine, so do not run one alongside a session.

**This is not the gate result.** It uses clean synthetic speech, because no person has spoken to it yet. The model ran through the `ollama` command rather than the HTTP backend. On CPU, Whisper takes 0.6 s (`base`) to 1.7 s (`small`), so the first-word target needs the GPU.

### Grading a model

```sh
uv run python -m affect_dialogue eval --llm ollama --model qwen2.5:7b
```

This runs 24 cases, half in each language: appraisal direction, format, reply language, spoken style, hostility and honesty about feelings. `qwen2.5:7b` passes 23. The miss is an honest answer that does not use the word "model".

### ROS 2

```sh
docker run --rm -v "$PWD:/ws" -w /ws ros:jazzy-ros-base bash tests/ros/verify.sh
```

This starts the real node and checks it with the real ROS tools. See [node.py](packages/affect-ros/src/affect_ros/node.py) for the topics and services.

### Scripted scenarios and replay

```sh
uv run python -m affect_core run scenarios/threat_recovery.json --log logs/threat.jsonl
uv run python -m affect_core replay logs/threat.jsonl                # verifies it exactly
uv run python -m affect_core replay logs/threat.jsonl --preset shy   # how would a shy robot have reacted?
```

## Add a robot

No core changes are needed:

1. Write `profile.yaml`: its channels, their ranges and rate limits, and how PAD and labels map onto them. See [quadruped.yaml](packages/affect-express/src/affect_express/profiles/quadruped.yaml).
2. Write the robot's limits separately. A profile must never be able to widen them. See [limits/quadruped.yaml](packages/affect-adapters/src/affect_adapters/limits/quadruped.yaml), whose numbers are **placeholders to replace before connecting hardware**.
3. Subclass `BaseAdapter` and implement `_send`, or pass a function to `SdkAdapter`. Clamping, rate limiting and stop are inherited.

## Layout

| Package | Holds | Depends on |
| --- | --- | --- |
| `affect-core` | messages, appraisal, dynamics, personality, labels, logging, replay | numpy, pydantic |
| `affect-express` | embodiment profiles, mapper, idle motion, cues | core |
| `affect-adapters` | adapter interface and limits; avatar, simulator and SDK adapters | express |
| `affect-ros` | ROS 2 node and its limit-checked adapter | adapters; rclpy from the ROS install |
| `affect-dialogue` | VAD, STT, LLM, TTS, memory, turn-taking, latency bench, model grading | core, express; speech and cloud are optional |
| `affect-server` | HTTP and WebSocket API; runs everything in real time | all of the above, FastAPI |
| `affect-dash` | dashboard and avatar face, static files with no build step | none |

Plugins register through the `affect.llm`, `affect.stt`, `affect.tts`, `affect.vad` and `affect.adapters` entry points, so a lab can add a backend without forking.

## How a turn works

1. The VAD finds the end of speech. All latency is measured from the last voiced frame, not from detection.
2. **The robot reacts at once**, from the affect state alone, with no model involved.
3. Whisper transcribes the utterance, in Spanish or English.
4. The LLM streams one JSON appraisal, then the reply. The appraisal goes straight to the emotion core, so the body changes before the words are finished.
5. Each sentence is spoken as soon as it is complete, with rate, pitch and energy read from the affect state at that moment.
6. Sustained speech from the person cancels all of this. The interruption is itself a mild appraisal, unless the robot had not started speaking: then it just listens on, and answers the whole thought.

Memory holds only sentences the robot actually began to say. If the model fails or times out, the robot says so and carries on. LLM output is untrusted: every value is clamped, and if the appraisal is missing or malformed, a keyword appraiser stands in, so an insult still registers.

## How it stays honest and safe

- **Replay.** The engine never reads a clock, so a session re-runs bit for bit from its log. Live sessions replay exactly too. Logs also hold commands, transcripts and turn timings.
- **Safety limits.** The adapter holds the limits, clamps every value, limits its rate, drops channels it does not know, and refuses everything after `stop()`. Software stop is at `POST /api/stop`. **The hardware e-stop must bypass all of this.**
- **Hostility.** The prompt forbids it, anger and fear are capped, and a word list softens what is said. The list is crude and should be checked in the pilot.
- **Honesty.** The prompt tells the robot to say it runs a model of emotion and does not claim to experience anything.
- **Privacy.** Audio stays on this machine and is not logged. Cloud calls send text only. Transcripts are logged, so `logs/` is ignored by git.
- **Network.** The server has no authentication and binds to `127.0.0.1` by default.

## Decisions

Open questions from the plan, settled here:

| Question | Decision | Why |
| --- | --- | --- |
| Cloud or local | Local STT and TTS; LLM selectable | The RTX 5070 runs speech in about 100 ms, and audio stays on the machine as the plan requires |
| Two robots | Avatar and quadruped | The plan's recommendation; tests a face and a whole body |
| VLA conditioning | Deferred | `AffectState` is already a plain vector, so nothing blocks it later |
| Licence | Apache-2.0 | Permissive, with a patent grant suited to robotics |
| Publish before or after the paper | **Still open** | Nothing is published; decide at Phase 4 |
| Student owners | **Still open** | A decision about people |

Hand-set starting points, for the Phase 3 pilot to test:

| Choice | Decision | Where |
| --- | --- | --- |
| Appraisal weights | From the Affect Pet test bench. Relevance gates the impulse, and `abs(congruence)` feeds arousal | `appraisal.py` |
| Agency | Logged, but has no weight yet | `appraisal.py` |
| Big Five baseline | Least-squares inverse of Mehrabian's trait = f(PAD) equations, neuroticism as negative stability, then halved so ordinary traits leave room to react | `personality.py` |
| Big Five gains | Mehrabian gives none. Higher neuroticism reacts harder and recovers more slowly | `personality.py` |
| Neutral label | Near the personality's baseline as well as near the origin | `labels.py` |
| "Interest" prototype | (0.35, 0.30, 0.10) | `labels.py` |
| Negative caps | Anger and fear capped at 0.8 intensity, landing on the emotion's prototype; they decay twice as fast | `engine.py` |
| Soft saturation | A push counts less the nearer the state is to that edge of the cube; pushes the other way count in full. The plan only clamps, but clamping alone pinned the robot at full joy within two friendly remarks | `engine.py` |
| Echo gate | Passes frames at least twice as loud as the echo expected from what is being played | `vad.py` |
| Dominant cause | Largest decayed impulse. A heuristic, not an exact decomposition | `engine.py` |
| Expression mappings | All profile values | `profiles/*.yaml` |
| LLM model | `claude-opus-5-5` at low effort, with server-side refusal fallback. Its latency is unmeasured; `--llm-model` changes it | `llm.py` |
| Whisper | `small` on GPU, `base` on CPU, whole utterance at the endpoint rather than incremental | `speech.py` |

Known behaviour to tune:

- A threat straight after strong praise reads as "surprise", not "fear", because leftover joy keeps pleasure positive.
- Praise makes the shy and grumpy presets read as "interest", never "joy". This fits their personalities, but raters may disagree.
- Fear after a threat lasts 2 to 5 s depending on personality, which is short. It follows from the plan's fast-decay rule.

The Mehrabian coefficients come from a secondary source, [arXiv:0809.4784](https://arxiv.org/abs/0809.4784). Check them against the 1996 paper before publishing.

## Not tested

- A person speaking. All speech so far is Piper's clean voice, plus added white noise at 20, 10 and 5 dB in some runs.
- A real microphone, and audible playback. The speaker path was exercised on the real device with silence only.
- Echo in a real room. The gate passed on simulated echo: a delayed, attenuated copy of the robot's voice.
- The Claude backend against the live API, since no key was available. It was tested through the real SDK with a mock transport, so its latency is unknown.
- The Ollama HTTP backend against a running server, tested with a mock transport. The model itself was exercised through the `ollama` command.
- The running web server over a real socket. The routes and WebSocket are tested in process, and the dashboard's clicks in headless Chrome against stubbed connections.
- Any real robot. Quadruped limits are placeholders, and commands are JSON on `std_msgs/String`, with no robot controller listening yet.
