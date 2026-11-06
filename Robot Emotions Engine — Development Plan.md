# Robot Emotions Engine — Development Plan

Oct 31, 2026 · @Jonathan

Build the engine as a robot-agnostic Python core with an explicit affect state, an LLM that appraises and talks but does not own the emotion, and per-robot adapters driven by configuration. Plan for about 30 weeks to a first published, evaluated release.

## Read this first: risks and constraints

These shape the architecture, so they come before the design.

- **Latency is the hardest constraint.** A turn feels natural under about 1 s from end of speech to first robot sound or motion. A naive STT → LLM → TTS chain lands at 2–4 s. Everything must stream, and the robot must react (gaze, nod, breath) before the words arrive.
- **Letting the LLM "feel" does not work on its own.** Asking the LLM each turn how it feels gives emotions that jump, contradict themselves and drift with the prompt. The engine needs its own state with dynamics; the LLM only supplies appraisals and words.
- **"Emotion" here means a computational model of affect.** Its validity is judged by people perceiving the robot, not by its internals. Budget for user studies from the start.
- **Generality lives in the mapping, not the model.** Joy on a quadruped (posture, gait) shares nothing with joy on a screen face. Expect each new robot to need a profile and tuning, not zero work.
- **Expression never overrides safety.** The engine sends expression requests; the robot's own controller clamps and can refuse them. This matters for the quadruped and the hand, which carry real torque.
- **Cloud vs local.** Cloud LLM/STT/TTS give the best quality but add cost, network latency and data-privacy questions. Local models (a GPU workstation or Jetson Orin) cut latency but lower quality, mainly in Spanish speech.
- **Human studies need ethics approval.** Recording participants' voice and face requires the Uniandes ethics committee; start that paperwork in Phase 2, not at the end.

## Goals and scope

One engine, many robots: connect any robot through an adapter, talk to it naturally in Spanish or English, and see a consistent, believable emotional state in its voice and body.

**Goals**

1. A reusable core library with no robot or ROS dependency, usable from Python, ROS 2 or a network API.
2. Natural spoken conversation: streaming, interruptible, bilingual (es/en), with memory of the person.
3. An explicit, inspectable affect state with personality, mood and short-lived emotions that evolve over time.
4. Expression through whatever channels a robot has: voice prosody, face, gaze, posture, gesture, gait, sound, lights.
5. New robot = new embodiment profile + adapter, with no change to the core.
6. Full logging and replay, so every emotional change can be traced to its cause for research.

**Non-goals for the first release**

- Claims that the robot experiences emotion.
- Learning personality from data (fixed, configurable personalities first).
- Multi-party conversation with several people at once.
- Task planning or manipulation; the engine sits beside the robot's task stack, not inside it.

## Architecture

&#91;embedded content: engine architecture · 6 layers\]

Events flow down from speech and robot sensors; the LLM turns them into appraisals, the emotion core updates the state, and the mapper turns that state into commands each adapter can execute. The affect state also flows back to the LLM, so its words match the robot's mood.

## Emotion model

The state is continuous: emotion, mood and personality are each a point in Pleasure–Arousal–Dominance (PAD) space, the cube \[−1, 1\]³. That makes 9 continuous state variables, evolving on three timescales and driven by appraisals of events. Discrete emotions (joy, fear…) are never state. They are a read-out: the PAD point is projected onto labelled prototype regions to get a name and an intensity.

**Why this combination.** PAD is continuous, so it maps smoothly onto motor and voice parameters on any robot. Appraisal theory (OCC, Scherer's component process model) explains *why* an emotion arose, which the LLM can supply and which makes behaviour explainable. Discrete labels (joy, sadness, anger, fear, surprise, interest…) are derived from PAD for the LLM and for logging, never stored as the state itself.

**Appraisal.** Every event (an utterance, a touch, a task result, a face appearing) becomes an appraisal vector:

- relevance to the robot's goals
- goal congruence (good or bad for it), the main driver of pleasure
- novelty / unexpectedness, a driver of arousal
- agency (who caused it: self, user, world)
- coping potential / control, the driver of dominance
- norm compatibility (was it polite, fair)

An appraisal-to-PAD map turns each vector into an impulse. Start with hand-set linear weights, then fit them to user-study ratings.

**Dynamics.** Emotion E, mood M and personality baseline P are PAD vectors. Emotions jump on impulses and decay toward mood; mood drifts slowly toward recent emotion and back to personality.

```latex
\dot{E} = -k_e\,(E - M) + \sum_i \delta(t - t_i)\,W\,a_i
```

```latex
\dot{M} = -k_m\,(M - P) + \alpha\,(E - M), \qquad k_m \ll k_e
```

Here a\_i is the appraisal of event i and W the appraisal-to-PAD matrix. Clamp all states to \[−1, 1\]. Typical time constants: emotions 5–30 s, mood 5–30 min; both are personality parameters.

**Personality.** Define P and the gains (k\_e, k\_m, α, gain of W) from Big Five traits using Mehrabian's published trait-to-PAD mapping, so a user configures "extraverted, agreeable, low neuroticism" instead of raw numbers. Ship 4–5 presets (calm assistant, playful pet, shy, grumpy-but-kind) for demos.

**Output.** Each tick (20–50 Hz) the engine publishes the PAD state, the nearest discrete label with its intensity, the dominant cause, and the change since last tick.

## Conversation layer

The LLM gets two jobs per turn, done in one structured call: appraise what just happened, and reply in a way that fits the current affect state it is given.

**Turn pipeline (all streaming)**

1. Microphone → voice activity detection and endpointing (Silero VAD or similar), with barge-in so the person can interrupt.
2. Streaming speech-to-text, es/en auto-detect (faster-whisper locally, or a cloud streaming STT).
3. Optional user-affect perception: speech emotion from prosody, facial expression from a camera, text sentiment. These are appraisal inputs, never copied directly into the robot's emotion (no mirroring by default).
4. LLM call with the transcript, conversation memory, persona, and the current affect state as text ("you feel mildly irritated, rising; cause: interrupted twice"). It returns structured output: the appraisal vector, the reply text, and optional expression cues (a gesture, a pause, a laugh).
5. The appraisal goes to the emotion engine; the updated state reaches the expression layer before the reply finishes generating.
6. Expressive text-to-speech: speaking rate, pitch and energy set from arousal and pleasure; emotion style tags where the TTS supports them.

**Hiding latency.** While the LLM thinks, the robot shows listening and thinking behaviour from the affect state alone (gaze, small nods, a breath, a "mm"). Stream TTS sentence by sentence. Target under 1 s to first reaction and under 1.5 s to first word.

**Memory.** A short window of recent turns, plus a per-person long-term store (name, preferences, past topics) keyed by voice or face ID. Memory also feeds appraisal: the robot can be glad to see someone it knows.

**Model choice.** Keep the LLM behind one interface with backends for a cloud model and a local model, so labs without budget or connectivity can still run it.

## Expression layer and robot abstraction

A robot joins by declaring what it can express in an embodiment profile and implementing a small adapter; the core never knows which robot it drives.

**Embodiment profile (YAML).** Lists the robot's expression channels, each with its range, rate limit and a mapping from PAD (and discrete label) to channel parameters. Mappings are small functions or lookup curves, for example arousal → motion speed and amplitude, pleasure → posture openness, dominance → body height and gaze directness.

**Expression mapper.** Each tick it reads the affect state, evaluates every channel's mapping, blends in one-shot cues from the LLM (a nod, a laugh), and emits an `ExpressionCommand`. Idle motion (breathing, gaze shifts) is modulated by arousal so the robot never freezes.

**Adapter interface.** Four calls: `capabilities()`, `apply(command)`, `state()` and `stop()`. The adapter, not the engine, enforces joint, torque and velocity limits and may reject a command.

**Backends.** A ROS 2 node (topics for state and commands), a direct Python SDK adapter, and a simulator or on-screen avatar adapter so development needs no hardware.

**Example mappings**

| Robot | Channels | Joy | Sadness | Fear |
| --- | --- | --- | --- | --- |
| Quadruped (12 MyActuator joints) | body height, pitch, sway, gait frequency, step height, impedance stiffness | tall, bouncy gait, quick sway | low body, slow gait, head (front) down | crouch, stiff impedance, step back |
| Robotic hand | finger posture, gesture speed, grip compliance | open relaxed palm, quick fluid moves | slow, half-closed | fast withdrawal, tense fingers |
| Screen face / avatar | eyes, brows, mouth, blink rate, colour | wide smile, fast blinks | drooping brows and mouth | wide eyes, raised brows |
| Voice only | rate, pitch, energy, pauses | faster, higher, brighter | slower, lower, more pauses | faster, higher, tense |

Start with the avatar and the quadruped: one tests faces and voice, the other tests whole-body expression with no face at all.

## Software stack and interfaces

A pure-Python core package, thin wrappers for ROS 2 and the network, and three versioned message types that every part speaks.

| Package | Responsibility | Depends on |
| --- | --- | --- |
| `affect-core` | appraisal → PAD, dynamics, personality, labels | numpy, pydantic only |
| `affect-dialogue` | VAD, STT, LLM, TTS, memory, turn-taking | core + pluggable model backends |
| `affect-express` | embodiment profiles, mapper, idle motion | core |
| `affect-adapters` | ROS 2, Python SDK, avatar, simulator adapters | express |
| `affect-ros` | ROS 2 nodes, msgs, launch files | rclpy |
| `affect-server` | WebSocket/gRPC API for non-ROS robots and the dashboard | FastAPI |
| `affect-dash` | live view of PAD, causes, transcript; manual event injection | web |

Plugins (STT, LLM, TTS, perception, adapters) register through Python entry points, so a lab adds a backend without forking.

**Core messages**

```json
{
  "AppraisalEvent": {"t": 12.40, "source": "dialogue", "cause": "user praised the robot",
    "relevance": 0.7, "congruence": 0.8, "novelty": 0.3, "agency": "user", "control": 0.5, "norm": 0.6},
  "AffectState": {"t": 12.45, "emotion": [0.62, 0.40, 0.15], "mood": [0.20, 0.05, 0.10],
    "label": "joy", "intensity": 0.58, "cause": "user praised the robot"},
  "ExpressionCommand": {"t": 12.45, "channels": {"voice.rate": 1.12, "body.height": 0.8, "gait.freq": 1.15},
    "cues": ["nod"], "ttl_s": 0.2}
}
```

**Engineering basics.** Every message is logged to JSONL (and rosbag on ROS) with a session ID, so any session can be replayed offline with a different personality or mapping. Unit tests cover dynamics with scripted event sequences; CI runs the avatar adapter headless.

## Roadmap

&#91;embedded content: roadmap · 5 phases, 30 weeks\]

Each phase ends in a usable demo and a gate; the week 16 gate is the one that tests the "reusable and general" claim. Durations assume one or two graduate students working most of their time on it.

**Phase 0 prototype:** [Affect Pet test bench](https://claude.ai/artifact/MCXZBuZzSNEguWigdtNFvo) is a browser version of the dashboard. It runs the PAD dynamics from the Emotion model section and drives a pixel face, with event buttons, personality presets, time-constant sliders, a pleasure–arousal trail and an event log. Typed messages use a keyword stand-in for the LLM appraiser, and mood runs on seconds instead of minutes so changes are visible while testing.

## Evaluation plan

Success means people recognise the intended emotion, find the robot more natural with the engine than without it, and the same engine does this on two different robots.

| Measure | How | Target |
| --- | --- | --- |
| Turn latency | end of speech → first motion / first word, logged | < 1.0 s / < 1.5 s median |
| Affect stability | label flips per minute in scripted dialogues | no flip without a logged cause |
| Expression recognition | raters label emotion from silent robot video clips | clearly above chance (1/6), per robot |
| Perceived naturalness | Godspeed and RoSAS questionnaires after a conversation | engine > baselines |
| Portability | config and code lines changed to add a robot | profile + adapter only, no core edits |

**User study design.** Within-subject, three conditions in counterbalanced order: (A) no emotion, (B) LLM-only emotion (the LLM picks a label each turn), (C) the full engine. Run on the avatar and on the quadruped. Condition B is the important baseline, because it is what most current systems do.

## Safety, ethics and open questions

**Built-in rules**

- Affect can shape *how* the robot moves, never *whether* a safety limit applies; adapters clamp everything and a hardware e-stop bypasses the engine.
- Caps on negative states: anger and fear intensity are bounded and decay fast, and the robot never expresses hostility toward a person.
- The robot answers honestly if asked whether it has feelings: it has a model of emotion, not experiences.
- Audio and video stay on the lab machine by default; cloud calls send text only unless configured otherwise. Long-term memory is opt-in and deletable per person.

**Open questions to settle**

- [ ] Cloud or local LLM/STT/TTS for the first demo, given budget and Spanish quality?
- [ ] Which two robots for Phase 2: avatar + quadruped, or the hand instead of the quadruped?
- [ ] Should the engine also feed the VLA work, e.g. affect as a conditioning input for motion style?
- [ ] Open-source licence (MIT/Apache-2.0) and whether to publish before or after the first paper?
- [ ] Which graduate student owns the core, and which owns the robot adapters?
