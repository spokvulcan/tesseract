# Dictation capture: VPIO armed for the app's lifetime, idle duck lifted via AudioDeviceDuck

Status: accepted

**Voice Processing** (Apple's VPIO: AEC + AGC + NS) is the standard mode for all
microphone capture — the owner's PRD #175 experiment verdict was an unambiguous
quality win, so the toggle is removed. But VPIO imposes a triangle no public API
breaks: arming (`setVoiceProcessingEnabled(true)`) costs 170–600 ms (measured
2026-07-07, `research/vpio-lab`), an armed unit ducks all other system audio for
as long as it exists (duck engages at `AudioUnitInitialize`, engine stopped or
not), and the public ducking floor (`duckingLevel: .min`) is audible (owner-
verified by ear). Arming per capture is the press latency; disarming at idle is
the arm/disarm churn implicated in intermittent silent captures; staying armed
at `.min` is an all-day audible duck.

**Decision:** one capture engine, armed once (at prewarm/first use) and never
disarmed; at idle the duck is *reversed* with the private, weak-linked CoreAudio
SPI `AudioDeviceDuck(outputDevice, 1.0, nil, ramp)` — the same call Chromium and
WebKit ship for exactly this — plus `duckingLevel: .min` as the configured
floor. A recording sets `duckingLevel: .default` (the dip doubles as source-
level noise reduction, PRD #175); stop returns to `.min` + un-duck. Press cost
is engine start only (~50 ms measured, music playing or not).

All four legs were verified live on target hardware (2026-07-07, owner's ears +
`research/vpio-lab` runbook): `.min` idle duck clearly audible; SPI un-duck
restores full volume while armed; the duck correctly re-engages for a recording
after an un-duck; un-duck restores again after stop.

Considered and rejected:
- **Disarm after a grace at idle** (shipped 2026-07-06, `c687938c`): reintroduced
  the press latency and the arm/disarm churn — the state this ADR replaces.
- **Always armed at `.min`, public API only**: owner-rejected by ear.
- **Dual engine (warm plain + parallel VP arm + handoff)**: falsified in
  prototype — arming VP silently starves a concurrently-capturing plain engine
  in-process (1.1 s of audio from a 5 s run, no error, no recovery).
- **No VPIO (industry-standard dictation path)**: rejected by the experiment
  verdict; capture quality is the point.

Consequences:
- `AudioDeviceDuck` is private SPI. It is resolved via `dlsym` and nil-guarded;
  if a future macOS removes it, the fallback is the disarm-after-grace lifecycle
  (latency returns, correctness doesn't). Fine for a notarized, non-App-Store app.
- The un-duck targets the *default output device by ID* — it must be re-fired on
  default-output-device changes and after any engine rebuild.
- An armed-but-stopped VPIO lives in coreaudiod all day. The documented
  input-volume-pinning hazard (PRD #175) is a soak-test acceptance item, not a
  proven risk; the mitigation (cache/restore input `VolumeScalar`) is available
  if it ever materializes.
