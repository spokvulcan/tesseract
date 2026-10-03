# ADR-0085: The Lens streams, and the paste stays a full pass

- Status: Accepted
- Date: 2026-10-03
- Amends: ADR-0084 (the Lens is the one dictation overlay, not only the card
  where a take is fixed; it also takes the keyboard while a held take waits)
- Relates to: #612 (the Catch PRD, slice 2), #291 (the Live Partial signal the
  Live Preview succeeds), #283 (the dictation overlay redesign map, whose
  Overlay Variant registry this deletes), ADR-0081 (models share memory, not
  turns: Whisper's encoder joins the LLM on the GPU), ADR-0034 (a take the
  Proofread Pass rejects now shows in the Lens with "Insert anyway")

## Context

After slice 1 the Lens fixed a take once it had pasted, but the recording
overlay was still the Overlay Variant the Setting selected, by default the
classic pill, and it showed no words until the text landed. A wrong word was
noticed in the app, in a terminal often after it was sent. The old Live
Partial pump (#291) re-decoded the trailing 12 s of the take and ran only for
the exploration variants that drew it.

The 2026-10-03 replay (`docs/research/2026-10-03-dictation-streaming.md`: the
owner's last 100 takes, 21 minutes, played against a simulated recording clock
with measured decode times on an M3 Max, through the app's own WhisperKit
1.0.0, large-v3-turbo and decoding options) measured what streaming costs.
Where two systems disagree, Parakeet TDT 0.6B v2 decides which is right.

- **Pasting streamed text loses accuracy.** WhisperKit's own streaming rule
  confirms early segments from decodes that had not heard what came next. Over
  20 s its text differs from the full pass in 15 of 18 takes; overall 2.6% of
  words differ, and where they do the full pass is right 24 words to 10.
- **Apple's recognizer streams natively but hears the owner worse.**
  SpeechAnalyzer's text is ready 0.08 s after release, but 11.6% of its words
  differ from Whisper's and Whisper is right 113 to 40, mostly on technical
  words. It mishears the owner's terms differently from Whisper, so a Learned
  Word taught from Whisper's text would not flip in its preview.
- **On the Neural Engine the encoder costs about 1.1 s per decode whatever the
  audio length**, because it always encodes a 30 s window. A preview runs
  2.0 s behind the voice and updates every 1.5 s, and an encoder pass cannot
  be stopped once it runs, so a preview in flight at release delays the paste
  by 0.44 s on average.
- **On the GPU the encoder takes 0.4 s.** The preview runs about 1 s behind
  and updates about every second; the full pass lands 0.65 s after release
  instead of 1.29 s, and the paste waits about 0.1 s for a preview in flight.
  The text is the same within run-to-run noise: 13 of 2,111 words differ from
  the Neural Engine's, against 14 between two Neural Engine runs.
- **A generation running at the same time pays for it.** With the owner's 27B
  model generating, Whisper on the GPU slows it from 21.8 to 14.8 tokens/s
  while Whisper decodes, about 15% over a take since a preview decodes about
  45% of the time. Whisper slows too, but its full pass still lands in 1.45 s
  on the GPU against 2.0 s on the Neural Engine.
- **Starting the final pass at a pause saves nothing.** The owner lets go a
  median 0.52 s after the last word and the audio before release is not
  silence, so a pause detector starts the final pass at release in the median
  take.

## Decision

1. **The paste is the full pass.** On release Whisper decodes the whole take
   once, exactly as before, and that text goes through the regex cleanup, the
   Learned Words and the opt-in Proofread Pass to the paste. The Live Preview
   is only ever shown.
2. **The Live Preview decodes from the end of its last confirmed segment, on
   WhisperKit's confirmation rule.** When a decode returns more than two
   segments, all but the last two are confirmed and the next decode starts
   where they end; the rest is the provisional tail, which the next decode
   rewrites. An unconfirmed stretch longer than 20 s confirms all but its last
   segment, so a decode never grows toward Whisper's 30 s window. Decodes are
   paced like the Live Partial pump they replace: the first once 0.6 s of
   unconfirmed audio exists, the next 300 ms after the last lands, so a slow
   decode stretches its own cycle. WhisperKit's own pacing (a decode once a
   second of new audio exists) showed the first words later in the replay. The
   lane skips rather than queues: no preview decode while the final pass or
   another preview runs. The preview is always on; no variant gates it. The
   Lens draws confirmed words in full ink and the tail dimmed.
3. **The preview reads like the take will.** The regex cleanup and the
   Learned Words run on every preview, for the take's app, so a learned word
   flips to the owner's spelling as it is heard and a word left alone in that
   app stays alone. Catches on the preview are shown, not counted: only a
   committed take counts its Catches (ADR-0084). The Proofread Pass never runs
   on the preview.
4. **Whisper's audio encoder runs on the GPU.** It is one setting in the
   adapter's `WhisperKitConfig` (`audioEncoderCompute: .cpuAndGPU`); the mel
   stays on the GPU and the text decoder on the Neural Engine. The shader
   compile (about 4 s, once) happens in the load's prewarm, not during a take.
   A generation running beside dictation slows while Whisper decodes, which
   follows ADR-0081: models share memory, not turns on the GPU, as the voice
   already does. If the slowdown matters in practice, the fallback is a second
   encoder on the Neural Engine used while the LLM Gate is held, for about
   1.2 GB more memory. It is not built.
5. **Release cancels the preview, so the paste waits at most one encoder
   pass.** The key coming up stops the pump and cancels the decode in flight,
   and the final pass cancels any preview at entry too. WhisperKit checks
   cancellation before the mel, before the encoder and before every decoder
   step, so what is left is the encoder pass already running. A decode that
   resolves after its take ended is dropped, never shown on the next take.
6. **When the take lands, the words the preview had wrong settle into place.**
   `LensSettle` takes a word-level longest common subsequence of the last
   preview and the final text; every final word outside it settles, in the
   accent. The landed take stays up for a moment with "Missed one?" and the
   fix hotkey as the owner set it, then fades.
7. **Tapping ⇧ while talking holds the take.** The hotkey manager reads a Shift
   tap from its event tap while the dictation key is held (Shift down and up
   with no key in between; a capital or a shifted chord is not a tap). A held
   take commits (history, Correction Pair, Catches) but does not paste: it
   waits in the Lens, which takes the keyboard. ↩ pastes it, with any fixes,
   into the app in front; Esc keeps it unpasted ("Kept, not pasted") and the
   fix hotkey brings it back; a paste that fails keeps it as well. A second ⇧
   tap in the same take lets it paste again. The Check Before Pasting setting
   (Settings → Dictation) decides which takes wait: one where ⇧ was tapped
   (the default), every take, or none, where ⇧ does nothing. With "insert
   text automatically" off nothing waits, since nothing pastes. A fix on a
   held take is recorded as made on a held take.
8. **The Lens replaces the Overlay Variant registry, its Setting and the
   pill.** The registry held the classic pill and six explorations; it goes
   with the classic pill (`GlobalOverlayHUD`), its fixed-frame `OverlayPanel`
   and `PillMetrics`. The stored `overlayVariant` key is abandoned, not
   migrated. `OverlayPlacement`, `ScreenGeometry` and `OverlayScreenLocator`
   stay for the Companion's voice overlay. The Lens follows the Overlay Feed
   itself through listening (the level, the app, the caught count, the ⇧
   hint), finishing, landed, waiting and fixing; its panel is built at launch
   so the first press shows it at once. Errors show in the Lens ("Didn't hear
   anything", "Hold the key while you talk", "The microphone is in use"), and
   a take the Proofread Pass rejected says "Didn't catch that" with the reason
   and "Insert anyway". The pill's one-click wrong flag and its link to the
   history go: fixing lives in the Lens, and the history keeps its own flag.
   *Amended 2026-10-03 by the catch record (#612, slice 3):* the history's
   flag went with its pair editor, so "Insert anyway" is the one wrong flag
   left; pairs flagged before keep their flag and stay gold.
9. **The Lens never takes focus while listening.** It shows as a
   non-activating panel that stays non-key while the owner talks, so the app
   being dictated into keeps the keyboard. It becomes key only while a take
   waits or is being fixed, and gives the keyboard back before anything
   pastes. ADR-0084 had it take the keyboard only while fixing.
10. **Accessibility.** Reduce Motion turns flips (a provisional word rewritten,
    a learned word turning into the owner's spelling) and settles into
    crossfades and stills the level dot. Reduce Transparency puts an opaque
    card behind the words. Increase Contrast outlines the target word.
    VoiceOver announces "Listening" when a take starts, where the take
    pasted when it lands, and that a held take is waiting with how to fix or
    paste it; the preview reads as one element with its text.

## Considered and rejected

- **Pasting the streamed text.** It shortens the wait after a long take, but
  the GPU encoder gets most of that without the loss: the streamed text
  changed 15 of 18 takes over 20 s, and the full pass is right 24 words to 10.
- **Apple's recognizer for the preview.** It is faster and streams natively,
  but it is wrong more often on this owner's speech (Whisper right 113 to 40),
  keeps fillers, and hears the owner's terms differently from the Whisper pass
  that pastes, so Learned Words would not flip live and the preview would
  promise text the paste does not deliver.
- **Starting the final pass at a pause.** The owner lets go about 0.5 s after
  the last word and the audio is not silent before release, so it starts at
  release in the median take and saves nothing.
- **Keeping the trailing 12 s window pump.** Every decode re-reads the same
  stretch, so nothing is ever confirmed: the Lens could not tell settled
  words from provisional ones, could show only the last 12 s, and on the
  Neural Engine 20% of words appeared only after release. Decoding from the
  confirmed end shows the whole take at a bounded cost per decode.

## Consequences

- Dictation is faster after release (about 0.65 s instead of 1.29 s) and
  shows the words about a second behind the voice; the pasted text is the
  same decode as before.
- The recognizer is busy about half of every take, on the GPU, where before
  the default overlay decoded nothing until release. A generation running at
  the same time is about 15% slower over a take; the Neural Engine fallback
  is known and costs about 1.2 GB.
- The overlay exploration is over: a change to how dictation looks is a
  change to the Lens, not a new variant. About 5,000 lines of variants, the
  pill and its panel are gone.
- The Lens takes the keyboard more often: whenever a take waits, not only
  when the owner asks to fix one. It is built under the same focus
  constraints as the Jarvis panel (the macOS 27.0 key-view freeze,
  `tools/overlay-focus-hang-lab`).
- A held take pastes through the same clipboard loan as any take, into the
  app in front when ↩ is pressed. Clicking away keeps it unpasted, as Esc
  does, and it stays the take the fix hotkey brings back only until the next
  take lands; after that it is in the history, never pasted.
- An owner's defaults may keep an `overlayVariant` value that nothing reads.

## As built

- `Features/Dictation/Lens/`: `LivePreview.swift` (`LivePreview`, and
  `LivePreviewAssembler`, the confirmation rule as a pure fold),
  `LensSettle.swift`, `LensModel.swift` (listening, finishing, landed, the
  held mode), `LensController.swift` (follows the Overlay Feed; holds, pastes
  and keeps a held take; errors and "Insert anyway"; `LensPanelPresenter`
  builds the panel ahead of the first press), `LensView.swift` (the live
  words, flips, the level dot), `DictatedTake.swift` (`held`).
- `Features/Dictation/DictationCoordinator.swift`: the preview pump
  (`startPreviewPump`), `shiftTapped`, `holdsTake`;
  `Features/Dictation/DictationFeed.swift`: `preview`, `targetApp`, `isHeld`.
- `Features/Transcription/TranscriptionEngine.swift`: the preview lane returns
  the whole result and is cancelled by its caller;
  `Features/Transcription/SpeechRecognizer.swift`: segment times are relative
  to the audio passed; `Features/Transcription/WhisperKitSpeechRecognizer.swift`:
  the encoder on the GPU.
- `Models/CheckBeforePasting.swift`; `Features/Settings/SettingsCatalogue.swift`
  and `SettingsManager.swift` (the setting, `overlayVariant` abandoned);
  `Features/Settings/Panes/DictationSettingsPane.swift` (the picker);
  `Features/Settings/Panes/GeneralSettingsPane.swift` (the Recording Overlay
  section removed).
- `App/DependencyContainer.swift` (the Lens wiring, ⇧ on the dictation
  hotkey's registration), `App/AppBindings.swift` (the variant, z-order and
  affordance rules removed), `App/AppDelegate.swift`;
  `Platform/OverlayPlacement.swift` (the pill's placement removed).
- Deleted: `Features/Dictation/OverlayVariants.swift`,
  `Features/Dictation/Views/GlobalOverlayHUD.swift`,
  `Features/Dictation/Views/Variants/` (six variants),
  `Platform/OverlayPanel.swift`, `Platform/PillMetrics.swift`, and the pill's
  `AudioBarsView` and `ProcessingDotsView`.
- `Core/Audio/AudioCaptureEngine.swift`: `captureSnapshot(from:)` and
  `SampleBuffer.snapshot(from:)`, so each preview decode copies only the
  audio after the last confirmed segment, never the whole take (tested in
  `SampleBufferTests`).
- Tests: `LivePreviewAssemblerTests`, `LensSettleTests`, with `ScriptedSpeechRecognizer` (a
  recognizer peer that plays one result per decode), and additions to
  `DictationCoordinatorTests` (the preview decodes from the last confirmed
  segment, ⇧ holds a take, the always setting), `DictationFeedTests`,
  `LensControllerTests` (listening and landing; a held take pasted, kept and
  failed; errors; "Insert anyway"), `LensViewRenderTests` (listening, landed
  with a settled word, a held take), `TranscriptionEngineTests` (segment times,
  the caller's cancel frees the lane), `SettingsCatalogueTests` and
  `SettingsManagerTests`. `AppBindingsTests` and `OverlayPlacementTests` lose
  the deleted rules and the pill.
