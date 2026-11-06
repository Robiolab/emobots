# Reflexion on the Robot Emotions Engine

Oct 31, 2026

**The core design is right, but I would change the conversation layer and the build order.** This review also found mistakes of mine, listed first. It comes from reviewing the design and what I remember of the code, with no new tests, so the items below are not yet confirmed by running anything.

## Mistakes to fix first

- **The 23/24 model score is inflated.** When I fixed the prompt, I added examples nearly identical to the graded cases. "You are a useless pile of junk" appears word for word in both. The score needs a held-out test set.
- **The prompt leaks a name.** One example says "Te llamas Ana", so the model could tell a stranger that is their name. The later memory run can't count either, because the name was already in the prompt. The first run, before I added that example, is still valid.
- **Any website can reach the local server.** `/ws` has no origin check, so a page you visit could read transcripts. `/api/resume` takes a bodyless POST, so a page could also un-stop the robot. Binding to `127.0.0.1` does not prevent this.
- **The command expiry stops nothing.** `ttl_s` only changes what `state()` reports. If the engine dies, the adapter does nothing, so the robot's controller must enforce the timeout.
- **Fallback keywords match inside words.** "hate" matches "whatever", and "odio" matches "episodio". This only fires when the model skips its appraisal.

## What I would keep

- The explicit emotion state, with the LLM only appraising and talking
- Exact replay
- Profiles and robot limits in separate files
- Local speech recognition

## What I would change, most important first

1. **Test with people now.** The plan's claim is that people recognise the emotions, and nobody has judged them yet. Nobody has even listened to the voice, because my prosody test only checks clip length and loudness. I would run a small recognition pilot on silent avatar clips, then on a rendered simulated quadruped, before writing more features.
2. **Add the study baselines.** Conditions A (no emotion) and B (the LLM picks a label) don't exist, so the main comparison can't run yet.
3. **Replace my hand-built turn-taking.** My endpointing, barge-in and level-based echo gate are the weakest code. I would trial Pipecat or LiveKit Agents, plus real echo cancellation and a smarter end-of-turn detector, and keep my code only if they lose.
4. **Replace the voice.** Piper with resampled pitch is unlikely to sound emotional, and voice carries much of the effect. I would compare emotion-capable TTS options by ear, in Spanish.
5. **Shrink the appraisal header.** About 580 ms passes between the model's first token and the first word, and the roughly 60-token JSON must finish before speech can start. I measured that gap but not how it splits. A compact array, with the cause sent after the reply, could save a few hundred ms.
6. **Make agency count.** Blame is what separates anger from sadness in appraisal theory, and it currently has no weight.
7. **Remove some patches.** Soft saturation, prototype capping, faster decay and the neutral-at-baseline rule each fixed one symptom. Together they make the model harder to fit and to explain in a paper. I would keep only what the pilot shows is needed.
8. **Use typed ROS messages.** JSON on `std_msgs/String` was a shortcut, and a real-time controller should not have to parse it.

## What I would not do

I would not switch to a cloud speech-to-speech API. It might feel faster, but it sends audio off the machine, which goes against the plan's default. It also takes appraisal and prosody out of the engine's control, and those are the research.

## Build order

I built Phases 1 and 2 before anyone tested Phase 0's face and emotions. The better order was the core, then the avatar with typed chat, then a recognition pilot, and only then speech and ROS.

The code does not need throwing away. The next step should be fixing the mistakes above and running the pilot, not starting Phase 3.
