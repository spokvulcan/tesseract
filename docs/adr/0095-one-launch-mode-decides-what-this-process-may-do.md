# ADR-0095: One Launch Mode decides what this process may do

- Status: Accepted
- Date: 2026-10-11
- Extends: ADR-0073 (a test run never reaches the owner's data; its
  process-level detection and the scratch launch)
- Relates to: ADR-0081 (memory, not the GPU, is what the models share),
  ADR-0057 and ADR-0059 (the DFlash2 bench and ruler), ADR-0047 (the sandbox
  model migration)

## Context

The app runs as four kinds of process: the owner's app, a scratch launch
(`scripts/dev.sh dev --scratch`), the unit tests' host, and a headless
harness (a bench, a parity check or a trace replay that `scripts/bench.sh`,
`scripts/dev.sh` and their siblings start with a flag). "What kind of process
is this?" was answered by four predicates in ten files:
`ProcessEnvironment.isRunningTests`, `ProcessEnvironment.usesScratchData`,
`TesseractApp.isHarnessLaunch`, and `CommandLine.arguments.contains` with a
flag written out at the call site.

- **The harness flags were written twice.** Once in `TesseractApp.harnessFlags`
  (for the single-instance check and the migrations) and again in the
  17-branch dispatch chain in `TesseractApp.init`, with each harness's label
  and log subdirectory.
- **Only three harnesses kept off the owner's background work.**
  `DependencyContainer.setup()` returned early for the SSD read bench, the
  TurboQuant bench and the DFlash2 bench, each added after it broke. The
  DFlash2 guard came after a Companion moment loaded a second copy of the
  model beside a ruler run and deadlocked both in MLX's compile. The other 14
  harnesses still started the hotkeys, the HTTP server, the Companion, the
  Whisper load and the proofreader prewarm beside the model they measure.
- **Harness launches opened the main window**, which is what ran `setup()`
  and installed the menu bar item. On a machine where the owner had not
  finished onboarding, a harness opened the Welcome window instead, whose
  first chapter starts downloading models.
- **The answers were spread out.** The single-instance check skipped
  harnesses at its call site and the test host inside itself; the sandbox
  migration and the retired-checkpoint removal skipped tests, scratch launches
  and harnesses, but the Companion's retired-data clean skipped only tests.

## Decision

1. **One module decides the Launch Mode once**, from the process's arguments
   and environment: `owner`, `scratch`, `testHost`, or `harness(flag:)`
   (`tesseract/Core/LaunchMode.swift`). The environment decides first: the test
   runner's variables make a test host, `TESSERACT_SCRATCH_DATA=1` a scratch
   launch. Only then do the arguments name a harness. Anything else is the
   owner's launch.
2. **Callers ask the answers, never the mode's name.** What a process may do:
   - keep off the owner's data (storage, settings, the Agenda, the Profile,
     day state, the browser profile): a scratch launch and the test host;
   - run the owner's background work (`setup()`): the owner's launch and a
     scratch launch;
   - the window it opens at launch: the owner's first launch opens the
     Welcome window and every later one the main window, a scratch launch
     opens the main window, and a test host and a harness open none and
     restore none;
   - post OS notifications: the owner's launch only;
   - enforce a single instance: the owner's launch and a scratch launch;
   - migrate or clean the owner's data at launch: the owner's launch only;
   - play the app's sounds: everything but the test host.
3. **One harness table** (`tesseract/App/Harness.swift`) replaces both the
   flag set and the dispatch chain. Each row holds the flag, the label, the log
   subdirectory, the exit code a failure ends with, and how it runs. Its order
   is precedence: `--paro-parity-bench`, and every harness `bench.sh` starts,
   comes before `--benchmark`, which `bench.sh` always passes. `--benchmark`
   keeps its own behaviour: it logs a failure and exits 0. The table decides
   the Mac's `LaunchMode.current`, and `TesseractApp.init` starts the harness it
   names.
4. **Every harness skips the owner's background work and opens no window.**
   None of the runners under `Features/Agent/Benchmark/` needs anything
   `setup()` starts: each builds its own engine and loads its model from the
   model folder.
5. **The container picks its adapters from the answers**, as before: the
   in-memory Settings Store Adapter, Agenda store, Profile and day state where
   the process keeps off the owner's data, and the OS notification center only
   where it posts.

The decision and its answers are pure and shared with the iPhone app, which
compiles `Core/`. The harness table is the Mac's: its rows run Mac-only code.
Shared code asks `StorageEnvironment.keepsOffOwnersData`, which decides from
the environment alone. That is the same answer, because the environment
decides first and a harness keeps the owner's data as the owner's launch does.

## Considered and rejected

- **The harness enumeration in `Core/`, its runners in an exhaustive switch
  on the Mac.** The compiler would refuse a harness without a runner, but one
  harness would live in four places (its case, its label, its log
  subdirectory, its runner), and the phone would compile the Mac's bench
  names. A row in one table is the one place a new harness is added.
- **A launch-mode parameter on `DependencyContainer`.** ADR-0073 rejected a
  storage parameter for the same reason: the test host's own container would
  still reach the owner's data. The mode stays process-level.
- **A scratch harness** (harness flags honoured in a scratch launch). No
  script starts one, and a harness measures against the owner's models. The
  environment decides first, so a scratch launch with a harness flag is a
  scratch launch.

## Consequences

- A new harness is one row; a new question about the kind of process is one
  answer, tested in `LaunchModeTests`.
- Behaviour changes, each pinned in `LaunchModeTests`:
  - every harness skips `setup()`: before, 14 of the 17 started the owner's
    background work beside the model they measured;
  - a harness opens no window and restores none, so it no longer installs the
    menu bar item, and on an un-onboarded machine it no longer opens the
    Welcome window;
  - a harness no longer runs the Companion's retired-data clean at launch
    (only tests skipped it). A scratch launch skips it too, which changes
    nothing: its storage starts empty.
- A scratch launch no longer marks onboarding done in its in-memory settings
  to keep the Welcome window shut; the Launch Mode opens the main window for
  it.
- A harness is now a windowless process. If App Nap ever throttles a long
  bench while the owner works in another app, the harness run should hold a
  user-initiated activity. Before, the window it opened was hidden behind
  other windows the same way.

## As built

- `tesseract/Core/LaunchMode.swift` (was `ProcessEnvironment.swift`): the
  mode, the decision, the answers.
- `tesseract/Core/StorageEnvironment.swift`: `keepsOffOwnersData`.
- `tesseract/App/Harness.swift`: the table, `Harness.start()`,
  `LaunchMode.current`, `LaunchMode.harness`.
- `tesseract/App/TesseractApp.swift` (the harness start, the launch windows),
  `tesseract/App/AppDelegate.swift` (single instance),
  `tesseract/App/DependencyContainer.swift` (adapters, `setup()`),
  `tesseract/App/SandboxMigration.swift`,
  `tesseract/Features/Models/RetiredCheckpoints.swift`,
  `tesseract/Features/Companion/Migration/RetiredCompanionData.swift`,
  `tesseract/Features/Browser/Core/AgentProfile.swift`,
  `tesseract/Features/Dictation/DictationCoordinator.swift` (sounds).
- Tests: `LaunchModeTests`.
