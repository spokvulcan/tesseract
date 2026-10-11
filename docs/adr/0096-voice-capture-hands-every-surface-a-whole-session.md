# ADR-0096: Voice capture hands every surface a whole session

- Status: Accepted
- Date: 2026-10-11
- Amends: ADR-0085 (§2, "every caller gains them", is now enforced by
  construction), ADR-0086 (§1–3: the Live Preview and the final pass read
  through one pipeline inside the session), ADR-0034 ("insert anyway" is the
  session's)
- Relates to: ADR-0081 (the menu's Models section), ADR-0082 (the voice
  session's takes)

## Context

The **Voice Capture Session** was the deep core of push-to-talk: the
capture→transcribe→commit lifecycle, the Operation Guard, the guards, the
Learned Words. But every dependency it took was optional with a nil default,
so each surface restated the whole recipe, and the compiler could not tell
when one left a stage out.

- **The Onboarding Tour built its Voice Input without the Learned Words, the
  Capture Dump or the Proofread Pass.** ADR-0085 §2 says every caller gains
  them.
- **The proofread closure and the language read were hand-built three
  times**: in dictation (with the feed narration), in Voice Input, and in the
  voice session.
- **The Live Preview re-composed the take's text pipeline.** The dictation
  coordinator ran its own regex cleanup and Learned Words for the preview,
  with its own staleness epoch, and reached past the session to the
  microphone and the recognizer. ADR-0086 §3 ("the preview reads like the
  take will") was enforced by two copies of the pipeline.
- **"Insert anyway" re-implemented the commit**: the history entry, the catch
  counting and relocation, and the paste, by hand beside the session's own.
- **"Is a take open" was not one fact.** The Models section's Dictation row
  enumerated surfaces by hand; #636 had to add Today's mic to the list.
- **The UI copied two rules**: the hold-to-talk gesture three times, and
  "spoken words join typed words" four times, with small differences.

## Decision

1. **Voice Capture is built once**, in the composition root, from the shared
   microphone and recognizer, the Capture Dump and its switch, the Learned
   Words, the Proofread Pass and the dictation language. Every one is
   required. It hands each surface its own session (`makeSession()`); the
   session's initializer is private to it. Dictation, the agent composer,
   Today, the capture and Jarvis panels, the voice session and the Onboarding
   Tour (through the environment) all take their sessions from it.
2. **The session owns the take**:
   - its one text pipeline (the regex cleanup, then the Learned Words for the
     take's app), which both the Live Preview and the final pass read
     through. `start(preview:)` runs the preview pump inside the session;
     `stop()` and `cancel()` end it;
   - the Proofread Pass, as the take's `.proofreading` phase rather than a
     wrapper each surface builds;
   - "insert anyway": `commitRejectedTake(_:)` commits the last rejected take
     through the surface's own commit, exactly like a take that landed, and
     counts its catches once it is in;
   - whether its take is open (`phase`: listening, transcribing,
     proofreading). Voice Capture answers it across every session, and the
     Models section reads that answer.
3. **Surfaces keep only what is theirs**: their commit (dictation's history,
   Correction Pair, paste or held take; a field's text), their presentation,
   their sounds and their errors. The commit receives the take's text, its
   length, its catches where they stand in the text, and its app.
4. **One hold-to-talk gesture** (`View.holdToTalk`) and **one rule for spoken
   words joining typed ones** (`AgentVoiceInputController.joining(_:to:)`:
   after the typed words, trimmed, with one space) replace the copies.

## Considered and rejected

- **A port per stage at the session's interface.** Each stage has one
  production adapter, and the tests already have peers below the module (the
  fake microphone, the recognizer peers, the in-memory stores). A required
  value at the module's construction is what stops a surface skipping a
  stage; ports would only add indirection.
- **Voice Capture hands out Voice Inputs.** Dictation and the voice session
  are not fields; a session is the unit every surface shares. A Voice Input
  takes a session.
- **Keeping the preview pump in dictation, fed by a shared pipeline
  function.** The pump's staleness rule is the session's own (its Operation
  Guard and phase), and only the session knows the take's app.

## Consequences

- No surface can build a take that skips a stage. The Onboarding Tour's takes
  gain the Learned Words, the Capture Dump and the Proofread Pass (a
  behaviour change, pinned in `AgentVoiceInputControllerTests`).
- The Models section's Dictation row reads one fact for every surface: a
  voice session's take now shows "Listening" and "Transcribing" like any
  other (before, the row said "Voice session" throughout), and the
  Proofreader row shows "Proofreading" for any surface's pass, not only
  dictation's. While a pass reads a take, the Dictation row is idle for every
  surface, as it always was for dictation (a Voice Input's pass read
  "Transcribing" there).
- "Insert anyway" commits like any take: its history entry keeps the take's
  length (it kept 0 s) and the take's own app.
- Spoken words join the agent composer's text the way they join every other
  field's: after the typed words with their trailing spaces and newlines
  trimmed (`AgentVoiceInputControllerTests`).
- A preview decode's span is no longer recorded in `DictationPerf`; it was
  cleared at every release, so the log line is unchanged.
- A test builds the module over peers (`tesseractTests/VoiceCapturePeers.swift`)
  and takes a session from it, as the app does.

## As built

- `tesseract/Core/VoiceCaptureSession.swift`: `VoiceCapture` (the stages,
  `makeSession()`, `isListening`, `isTranscribing`, `isProofreading`) and
  `VoiceCaptureSession` (`phase`, `start(preview:)`, `stop()`,
  `transcribeAndCommit(_:onTake:commit:)`, `commitRejectedTake(_:)`,
  `cancel()`).
- `tesseract/Features/Agent/AgentVoiceInputController.swift`: Voice Input over
  a session, `joining(_:to:)`; `tesseract/Features/Agent/Views/HoldToTalk.swift`:
  `View.holdToTalk`.
- `tesseract/Features/Dictation/DictationCoordinator.swift`: dictation's commit
  shared by a landed take and "insert anyway"; the feed follows the session's
  phase.
- `tesseract/Features/Companion/Voice/CompanionVoiceSessionController.swift`,
  `tesseract/Features/Onboarding/Views/DictationChapter.swift`, the composers
  and panels, `tesseract/App/DependencyContainer.swift` (`voiceCapture`), and
  the Models section's rows (then `DependencyContainer+ModelActivity.swift`;
  since ADR-0081's 2026-10-11 amendment, `tesseract/App/ModelResidency.swift`).
- Tests: `VoiceCaptureSessionTests` (through the module; the preview and the
  take read the same; insert anyway; one open take across surfaces),
  `AgentVoiceInputControllerTests` (every stage; the joining rule),
  `DictationCoordinatorTests`. The container's hand-list tests in
  `MenuModelsSectionTests` are gone.
