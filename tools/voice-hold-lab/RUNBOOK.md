# voice-hold-lab — RUNBOOK

The ADR-0041 runtime harness: raw AVAudioEngine/VPIO exercised with the voice
hold's exact call discipline, on real hardware. **Committed, not gitignored**
— v1 of this lab lived in `research/` and evaporated with its evidence; the
ADR now requires the harness and its results to travel with the code.

## Running

```sh
swift run --package-path tools/voice-hold-lab voice-hold-lab <command>
```

| command | what it does |
|---|---|
| `all` | E1 E2 E3 E5 E6 E7 (the non-interactive set, ~1 min of test tones) |
| `e1`…`e7` | one scenario (`e3 --unsafe` opts into the crash-class repro — expect SIGABRT) |
| `e4` | device-change watch, interactive: switch devices while it runs |
| `emit-fixture` | E2 + E5, then writes `tesseractTests/VoiceLabFixtures.swift` (`--with-owner-barge` adds the interactive owner-speech trace; `--out <path>` overrides) |
| `record-barge` | the owner-speech-over-reply trace alone |

Conditions for a valid `emit-fixture` run: **quiet room, normal listening
volume, Tesseract Agent quit** (its always-armed VPIO engine interferes),
terminal granted microphone access. The tones are audible — that is the
measurement.

After regenerating fixtures, run the replay lock:

```sh
xcodebuild test -project tesseract.xcodeproj -scheme tesseract \
  -destination 'platform=macOS' -only-testing:tesseractTests/VoiceBargeReplayTests
```

Red replay tests mean the shipped `EchoResidualFloor.Config.standard()`
constants don't hold against the new traces — recalibrate (margin,
echoPathLoss, attack) until zero clean-trace onsets AND the owner splice
fires ≤ 600 ms, then commit constants + fixtures + this file together.

## Results — 2026-07-17, MacBook Pro (Apple Silicon), macOS 26.5.2, built-in mic + speakers

### E1 — hold wiring + capture gate + dynamic attach

- Wired vp=true render=true; 20 gate open/close cycles on the running engine: ok.
- Player attach/connect/detach on the RUNNING engine fired **0**
  `.AVAudioEngineConfigurationChange` notifications (the D3 rebuild-echo risk
  does not materialize on this hardware; the persistent node stays the design
  anyway).

### E2 — echo residual at the VP mic (single held rig, one continuous trace)

Signal: 2 s log chirp (100 Hz–8 kHz) + 6 s speech-shaped bursts, amplitude 0.5.
Levels are the app meter's domain: `(dBFS+60)/60`, 20 Hz bins.

| segment | peak | mean | n |
|---|---|---|---|
| noise floor (no playback) | 0.187 | 0.087 | 60 |
| hosted (VP engine's own player, undipped) | 0.708 | 0.132 | 160 |
| dedicated second engine, duck `.min` | 0.658 | 0.110 | 160 |
| dedicated second engine, duck `.default` | 0.595 | 0.112 | 160 |
| hosted, first second (from-cold AEC) | 0.121 | 0.036 | 20 |

**Readings.** (1) Residual peaks 0.6–0.7 in *every* path — the static 0.25
gate never had a chance; this is the 2026-07-17 false-barge storm, measured.
(2) Hosted ≈ dedicated in steady state (loopback reference confirmed working
for other-engine audio), so the dual path's wins are the undipped reply and
convergence, not raw AEC gain — matching PR #355's E2 claim. (3) From-cold
AEC converges fast (first-second peak 0.121). (4) The minimum observed
playback−mic gap sets `echoPathLoss`: envelope peaks ≈ 0.85–0.9 vs mic peaks
≈ 0.7 ⇒ gap ≈ 0.2 — shipped as `echoPathLoss: 0.2` (the cap is exactly the
worst case; margin 0.08 rides above it).

### E3 — running-engine discipline

Gate flips, buffer scheduling, pause/play, node volume on the running VP
engine: no throw. Crash-class repro (`--unsafe` re-tap with a different
format) left opt-in; the 02:25 crash report is the standing evidence.

### E5 — duck/fade/pause/resume transient profile (hosted)

Duck to 0.25 over 100 ms at t=2 s, fade back over 200 ms at t=3 s, hard pause
t=4 s, resume t=5 s. Windows (10 bins each): post-duck peak 0.102, post-fadeup
peak 0.107, post-resume peak 0.099 — **the ducked reply reads ≈ 0.10 at the
mic, well under the 0.22 listening threshold** (the Soft Barge confirm window
gets a clean owner signal), and the held engine shows **no resume transient**
(AEC stays converged through pause/resume). `postResumeGrace = 1.0 s` is
generous; kept as safety against the fade-up ramp.

### E6 — full hold wiring latency

Five runs, cold engine each: 2261–2296 ms, median ≈ **2.28 s** (VP arm +
aggregate construction dominates). Consequence: the hold's wiring MUST stay
detached with captures fast-failing into the session's 1 s backoff (v2's
design), and the session-enter UX must cover ~2.3 s before the mic is live.
(PR #355's "~900 ms measured" does not reproduce on this hardware/OS.)

### E7 — begin/end churn

10 rapid wire → play → discard cycles: no throw, clean teardown.

### Lab-found gotchas (now encoded in the lab itself)

- **Back-to-back VP engine create/destroy cycles wedge CoreAudio input** —
  later engines in the process (and for a while, in NEW processes) get zero
  input buffers. The same pattern the app's kept-engine design avoids. E2
  therefore uses ONE held rig with segment slicing; a too-short trace aborts
  the run rather than emitting garbage fixtures. If wedged: wait ~10 s and
  re-run; persistent wedges clear with a coreaudiod restart or device toggle.

## Pending owner scenarios

- `e4` device-change under hold (interactive watch).
- `record-barge` / `emit-fixture --with-owner-barge`: the real
  owner-speaking-over-reply fixture (the replay suite currently pins the
  fire path with a synthetic 0.75 splice; a recorded one is better).
- `e3 --unsafe` if the crash-class repro is ever needed again.
