# ADR-0084: Learned Words replace the Proofread Pass as the corrector, and the Lens takes the keyboard

- Status: Accepted
- Date: 2026-10-03
- Amends: ADR-0034 (the Proofread Pass is off by default and no longer the
  corrector; Settings keeps it as an opt-in)
- Amended by: ADR-0085 (2026-10-03: the Lens is the one dictation overlay and
  streams the take while it is recorded, so the recording overlay is no longer
  the Overlay Variant (decision 6); it also takes the keyboard while a held
  take waits)
- Relates to: #612 (the Catch PRD, slice 1), #289 (the Correction Pair
  flywheel), #283 (the dictation overlay redesign map), ADR-0081 (dictation
  never waits for the LLM; Learned Words need no model at all), ADR-0073 (the
  new store resolves its directory through `StorageEnvironment`)

## Context

The 2026-09-28 replay of the owner's dictation
(`docs/research/2026-09-28-dictation-errors-and-learning.md`: 99 takes with
hand-checked references, 1,000 Correction Pairs) measured where the errors
come from and what the pipeline does about them:

- **Generic accuracy is high; the errors are the owner's own words.** Whisper
  large-v3-turbo gets 99.4% of words right before any cleanup (word error
  rate 0.61%), and 14 of the 16 errors left are the owner's names and
  technical terms. Across the 1,000 takes, 83 misrecognitions of about 30
  terms in 68 takes, and the same words come back: Claude as "cloud" (so
  `CLAUDE.md` becomes "cloud.md"), Tesseract as "SRACT", "SRAX" or "TSRAC",
  DFlash2 in six spellings, "a PR" as "APR".
- **The Proofread Pass is not a corrector.** Replayed on all 1,000 stored
  texts it fixed 4 misheard words and changed the meaning of 13 takes (a
  question turned into its own answer, a sentence dropped from an
  instruction); 69 more rewrote the owner's phrasing. On the 99 takes with
  audio it took committed-text errors from 0.69% to 1.39% of words. Given a
  glossary of the owner's 40 terms in its prompt, it fixed none of them.
- **The loop had no input.** 0 of the 1,000 pairs were gold: a fix meant
  opening the history window, finding the take and editing it there, and the
  owner never did. `CONTEXT.md` said "the overlay stays keyboard-free".
- **One fix per term would catch about half of the repeats.** Replaying the
  83 misrecognitions in time order as if each term was fixed the first time,
  39 of the later ones (47%) repeat an exact misheard form already fixed.
  Twelve more are a known term in a new spelling; sound-alike matching
  catches 10 of those but raises 31 false alarms on correct words.
- **The regex cleanup adds errors of its own.** It split file names in 14
  takes (`CLAUDE.md` as "cloud. Md"), turned Whisper's "..." pause into 61
  false sentence breaks in 47 takes, and dropped a word of deliberate
  repetition ("very very") in 10.
- **Silence gets text.** Seven of the 1,000 takes are exactly "Thank you.",
  at least one from a capture peaking at -78 dBFS. The app passes WhisperKit a
  no-speech threshold, but WhisperKit 1.0.0 never computes the probability it
  compares against: it is a hard-coded zero.

## Decision

1. **Learned Words are the corrector.** A Learned Word is one "heard → meant"
   replacement learned from the owner's fix: the meant spelling, every heard
   form (normalized, possibly several words, "D flash two"), the apps it is
   left alone in (bundle ids), its Catches per day, its fix count, the
   Correction Pair it came from with a before-and-after example, and a
   forgotten state. `LearnedWordStore` keeps them as JSON beside the
   Correction Pairs; a disk failure is logged and swallowed, so learning
   never breaks dictation.
2. **The Voice Capture Session applies them after the regex cleanup and
   before the Proofread Pass.** Every caller gains them: dictation, Voice
   Input, the capture and Jarvis panels, the voice session. Matching is on
   whole tokens, case-insensitive, longest heard form first, never
   overlapping; punctuation inside a multi-word form breaks the match; case
   is fitted at the start of a sentence; a word left alone in the take's app
   is skipped. The take's app is the one in front when the take started
   (`TargetApp`). Each replacement is a **Catch**, counted once the take
   commits: a rejected, failed or superseded take caught nothing.
3. **A word is learned on the first fix, behind gates.** The owner chose the
   word by typing it, so one fix is a confirmation. A fix is learned only
   when it looks like a mishearing (`LensFix.shouldLearn`): never when either
   side is made of ordinary words (a stop list of function words and
   fillers), never when it only changes a word's ending ("drops" → "drop" is
   grammar), never when what was heard is only part of the meant word
   ("flight" for TestFlight would rewrite every "flight"), at most three
   heard and four meant words, and the two sides must sound alike
   (`SoundAlike`: a consonant-skeleton key compared by edit distance) or
   differ only in case, where a fix that gives a word its capitals is learned
   and one that takes them away fixes that take only. Anything else is a rewrite and fixes
   that take only. A heard form that belonged to another word moves to the
   latest fix; a fix to a spelling already known adds the heard form to that
   word.
4. **Fixing a Learned Word back leaves it alone in that app.** A catch fixed
   back to what was heard (Claude flipping in where "cloud" was meant, in a
   note about the weather) adds the take's app to the word's exceptions and
   takes the catch back; a catch fixed to a third word adds the exception and
   fixes that take.
5. **Forget keeps the word.** A forgotten word stops applying and stays
   listed, so Forget can be undone, and only a new fix by the owner learns it
   again. Every change the Lens offers (learn, leave alone, forget) returns a
   receipt, and Undo restores the words it touched exactly.
6. **The Lens takes the keyboard, only while a take is being fixed.** This
   reverses `CONTEXT.md`'s "the overlay stays keyboard-free". The fix hotkey
   (⌃⌥Space, configurable in Settings → Hotkeys) opens the last take in the
   Lens, a glass card at the bottom center of the screen built like the
   Jarvis and capture panels (`GlassPanel`, a borderless non-activating
   panel). The owner types the word they meant: the Lens picks the span of up
   to three words that sounds most like it (`LensFix.target`), completes from
   the Learned Words and a small vocabulary, and lets ← → or a click pick by
   hand. ⇥ fixes and stays for another word, ↩ fixes and finishes, Esc clears
   the typing and then closes. After finishing, one line says what happened
   ("Learned SRACT → Tesseract · Undo", "Fixed here only", "Left alone in
   Notes"). The panel turns key only while fixing and gives focus back when
   done; every button is non-focusable and the field exists from the first
   layout (the macOS 27.0 key-view freeze, `tools/overlay-focus-hang-lab`).
   The recording overlay stays the Overlay Variant until slice 2.
7. **A fix goes back into the app only while the pasted text is still the
   last thing typed there.** No key pressed and no click since the paste
   (input made in the Lens itself excepted), the app the paste went into
   still in front, the same focused element in it; the events Tesseract
   posts itself carry a marker, so they never count as the owner's typing,
   and clicks are counted by a listen-only tap that never holds an event. The
   change runs from the first changed character to the end of the pasted
   text, so the caret ends where the owner left it. Where the field exposes
   its text through Accessibility, that part is selected back and the fix is
   pasted over it; where it does not (a terminal, an Electron app), it is
   erased with backspaces and the fix is pasted. Never in a password field,
   and never while Secure Keyboard Entry is on. Otherwise the fix still
   learns, and the Lens says the app already has the old text. If the
   backspaces went through but the paste did not, the Lens says so and never
   tries that paste again. This reads
   one field at the owner's request, which is the alternative the
   accessibility note recommends over watching edits; it never switches an
   Electron app into screen-reader mode.
8. **Every fix makes the take's Correction Pair gold.** The pair records each
   fix: heard, meant, how the take was reached (held take, after paste, from
   the page), the app's bundle id, and when. It also keeps the text after the
   Learned Words. Gold pairs are evicted last and keep their Capture Dump
   audio, so the fine-tuning corpus (#294) grows from real fixes.
9. **The Proofread Pass is off by default.** Settings keeps it as an opt-in;
   when on, it reads the text after the Learned Words and keeps every policy
   of ADR-0034.
10. **The regex cleanup stops adding errors.** File names and domains keep
    their dots, Whisper's "..." pause is no longer a sentence break with a
    capital, and deliberate repetition survives while stutters still
    collapse.
11. **A silent capture is not transcribed.** A capture whose level never
    rises above silence stops as `silent` (`CaptureLevel`): it is neither
    transcribed nor kept in the Capture Dump, and nothing pastes.

## Considered and rejected

- **Applying sound-alike matches on their own.** In the 2026-09-28 replay
  they caught 10 repeats in a new spelling and raised 31 false alarms on
  correct words. Sounds-alike only picks the word the owner is fixing and
  gates learning; it never changes text by itself, so a new spelling of a
  known term needs its own fix.
- **A vocabulary in Whisper's prompt.** At WhisperKit 1.0.0, the version the
  app pins, the prompt made the decoder return empty text on 74 of 99 takes
  (WhisperKit issue 501). At 1.1.0 it gained at most two terms, with more
  errors and slower decodes.
- **A vocabulary in the Proofread Pass's prompt.** With the 40 terms in its
  system prompt it fixed none and its error rate went to 1.42%.
- **Learning a word only after two confirmations,** as the 2026-09-28 note
  suggested for automatic rules. Here every rule comes from a word the owner
  typed; the Undo line at the moment of learning, Forget and the per-app
  exception cover a wrong one.
- **Biasing Whisper's decoder toward Learned Words** (a logits filter found 39
  to 42 of 48 terms against 35). Out of scope here: its own follow-up once
  Learned Words exist to feed it.
- **Learning from edits made in the target app through Accessibility.** Out
  of scope: blind in terminals and in Electron apps that are not in
  screen-reader mode, it reads far more than the dictated words, and many
  edits are rewrites (`docs/research/2026-09-28-dictation-accessibility-learning.md`).

## Consequences

- A first sighting still lands wrong; only a fixed word stays fixed. The
  replay puts the share of later repeats a single fix catches at 47%.
- A word with two meanings (Claude and cloud) misfires once in each app
  where the other meaning is wanted, until it is fixed back there.
- A Learned Word costs no model and no wait: dictation never waits for the
  LLM (ADR-0081), and with the Proofread Pass off a take commits without any
  LLM step at all.
- The Lens is a second glass panel that takes the keyboard over other apps.
  It is built under the same focus constraints as the Jarvis panel, and the
  design language's custom-glass inventory lists it.
- Putting a fix back synthesizes a selection and a paste (or backspaces) in
  another app. It acts only on the owner's fix, only on the field the take
  was pasted into, and only while nothing was typed there since.
- `learned_words.json` joins the owner's stores under `StorageEnvironment`,
  so a test run never reaches it (ADR-0073).
- The Correction Pair lineage gains a step (the text after the Learned
  Words) and a list of fixes, so pairs exported for #294 say what the owner
  changed and where.

## As built

- `Features/Dictation/LearnedWords/`: `LearnedWord.swift` (the word,
  `LearnedWordCatch`, `LearnedWordMatcher`), `LearnedWordStore.swift`,
  `SoundAlike.swift`, `TakeText.swift`.
- `Features/Dictation/Lens/`: `LensFix.swift` (target, decision, edit),
  `LensDictionary.swift` (the spell-checker gate), `DictatedTake.swift`,
  `LensModel.swift`, `LensController.swift` (with `LensPanelPresenter`),
  `LensView.swift`, `LensVocabulary.swift`.
- `Platform/InAppReplacer.swift`: a fix put back in the app;
  `Platform/SyntheticKeyEvents.swift`: the marker on every key event
  Tesseract posts; `Platform/HotkeyManager.swift`: `inputCount` (key presses
  and clicks, the clicks from a listen-only tap) and the Shift tap;
  `Platform/ModifierTapDetector.swift`; `Platform/GlassPanel.swift`: bottom
  placement and the key intercept.
- `Core/TargetApp.swift`; `Core/VoiceCaptureSession.swift` (Learned Words,
  `StopResult.silent`, the take's `learned`, `catches` and `app`);
  `Core/Audio/CaptureLevel.swift` (silent at or below -55 dBFS in the loudest
  20 ms window).
- `Features/Dictation/Corrections/CorrectionPair.swift` (`Fix`, `learned`),
  `CorrectionPairStore.recordFix`; `TranscriptionHistory.replaceText(forPairID:with:)`.
- `Features/Transcription/TranscriptionPostProcessor.swift`: the regex fixes.
- `Features/Settings/SettingsCatalogue.swift`: `proofreadDictation` off by
  default (a stored `true` from before is switched off once), the fix hotkey.
- Tests: `SoundAlikeTests`, `TakeTextTests`, `LearnedWordMatcherTests`,
  `LearnedWordStoreTests`, `LensFixTests`, `LensModelTests`,
  `LensControllerTests`, `LensViewRenderTests`, `CaptureLevelTests`,
  `InAppReplacerTests`, `InAppEditTests`, `ModifierTapDetectorTests`, and
  additions to `VoiceCaptureSessionTests`, `CorrectionPairStoreTests`,
  `TranscriptionPostProcessorTests`, `DictationCoordinatorTests`,
  `HotkeyMatcherTests`, `AppBindingsTests`, `SettingsCatalogueTests` and
  `SettingsManagerTests`.

Replayed offline on the owner's data (1,000 Correction Pairs and 100 Capture
Dump takes, aggregate numbers only):

| | Before | After |
|---|---|---|
| File names or domains split by the cleanup | 11 takes | 0 |
| "..." turned into a sentence break with a capital | 48 in 37 takes | 0 |
| Deliberate repetition lost | 16 in 13 takes | 0 |
| Stutters collapsed | 5 | 5 |
| Silent captures transcribed | 1 ("Thank you.") | 0; quietest speech 26 dB above the ceiling |

Replaying 79 logged misrecognitions (33 terms) in time order, with one Lens
fix the first time each spelling went wrong: every exact repeat of a fixed
form was caught (31 of 31, 39% of all 79; the rest are first sightings and
new spellings), every fix that reached the gates was learned, and the final
33 words made no catch on a take that had no logged error across the 1,000.
The auto target picked the misheard words on 40 of 46 fixes; the misses were
article-merged forms ("APR" for "the PR") and a span the owner widens by hand
(⇧← ⇧→).
