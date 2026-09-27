# ADR-0073: A test run never reaches the owner's data

- Status: Accepted
- Date: 2026-09-27
- Extends: #159 (telemetry written to a scratch folder under a test runner)
- Relates to: ADR-0035 (the memory evals read a copy of the owner's store)

## Context

The unit tests run inside the app. The test runner launches the full app as
the test host, and every suite runs in that process. The app keeps its data in
the owner's `~/Library/Application Support` and `~/Library/Caches`, and the
test host is the same app with the same bundle identifier, so it resolves the
same folders and the same defaults domain. Until now only telemetry moved
somewhere else under a test runner (#159).

So an ordinary test run reached the owner's data. During a run of
`BootstrapSequenceTests`, a suite that builds nothing, the test host held the
owner's memory database open and put its main window on screen. That window
opens on the Agent page, and building the Agent page opens the real
conversation store and seeds the agent folder. The store's initializer wipes
the conversations folder when its storage version differs from the build's,
so running the tests of a branch that bumps it would have wiped the owner's
conversations. The window's launch task also added the menu bar item and built
the dictation and speech services against the owner's files.

It also blocked a test we want. The Speech page crashed on open because it read
the model download manager from the environment and nothing injected it for
that page. The regression test for that kind of crash renders every page of the
main window with the app's own wiring. That means building a
`DependencyContainer` inside a test, and that container opened the same stores
the running app does.

## Decision

- Under a test runner, the app's storage roots (Application Support and
  Caches) resolve to folders in one scratch directory per test process. One
  per process, as in #159, because the scheme runs suites in parallel
  processes. `StorageEnvironment` is the seam.
- Every default storage location resolves through it: conversations, memory,
  the agent folder, transcription history, correction pairs, the capture dump,
  pinned voices, telemetry (the #159 diversion now builds on it) and the
  default SSD prefix-cache folder.
- The Agent Profile, the agent browser's WebKit data store, is non-persistent
  under a test runner.
- The container's settings use the in-memory Settings Store Adapter under a
  test runner, so every setting reads its catalogue default. The adapter moves
  from the test target into the app so the test host can use it.
- The test host opens no windows. The main window and the Welcome window are
  suppressed and not restored under a test runner. With in-memory settings the
  host would otherwise read onboarding as unfinished and open the Welcome
  window, whose first chapter starts downloading models. A suite that needs a
  view renders it itself.
- The model folder stays where it is. Some suites load installed models on
  purpose, and nothing under a test runner downloads one.
- Suites that read the owner's data on purpose, like the memory evals, locate
  it themselves and never write to it. The recall eval copies the memory store
  before opening it.

## Consequences

- A test run no longer reaches the owner's conversations, memory, agent
  folder, transcription history, pinned voices, correction pairs, browser
  profile or settings, whether a test builds a store itself or goes through
  the test host's container. Suites that read the owner's data on purpose
  still can.
- A suite that found the owner's data through an app default now finds the
  empty scratch copy, and a test gated on that data skips instead of running.
  The backfill's real-corpus guard did exactly that, so it now names the
  corpus the way the memory evals do.
- A `DependencyContainer` built in a test is safe to render, which is what
  `MainWindowPageTests` does for every page of the main window.
- Two things still read the owner's files. One is the model folder, as
  decided above. The other is the views' `@AppStorage` keys, which live in
  `UserDefaults.standard`, and in the test host that is the app's real
  defaults domain. Moving them would take a scratch defaults suite, which
  leaves files in `~/Library/Preferences`, for keys that rendering only reads.
- Scratch directories pile up in the temporary folder, one per test process,
  as the #159 telemetry folders already did. macOS clears them.
- Anything in the test host that depended on the owner's settings now sees
  catalogue defaults. Running the app from Xcode is unaffected, because that
  is not a test run.

## Considered and rejected

- A storage parameter on `DependencyContainer`, used only by containers that
  tests build. Narrower, but the test host's own container would still reach
  the owner's data on every run.
- Pointing the whole test process at a fake home directory from the scheme.
  It also moves the model folder, and it lives in the scheme, so any other way
  of running the tests reaches the owner's data again.
- Rendering pages through the test host's own main window. The owner's data
  stays in play, and it only works once onboarding is done.
