//
//  DictationCoordinator.swift
//  tesseract
//

import AppKit
import Foundation
import Observation

/// The global system-wide dictation driver. A thin composer over the shared
/// **Voice Capture Session**: it maps the session's `StopResult`/`Outcome` onto
/// the **Overlay Feed**'s phases and beats and keeps only what is
/// dictation-specific — its commit (history + auto-insert text injection),
/// success/error sounds, the maximum-recording-duration auto-stop,
/// `DictationError` mapping, and the error auto-reset. Distinct from
/// **Voice Input** (`AgentVoiceInputController`), the agent composer leaf,
/// which composes the same session for its own presentation.
///
/// The coordinator is the feed's sole phase/beat writer; the Lens and
/// the in-window dictation views read the feed, never the coordinator's
/// internals.
@Observable @MainActor
final class DictationCoordinator {
    let feed: DictationFeed
    private(set) var lastTranscription: String = ""

    /// The raw text of the last take the **Proofread Pass** rejected — kept
    /// so "Insert anyway" (the Lens's button, or this API directly) can
    /// still deliver the user's words.
    private(set) var lastRejectedRaw: String?
    /// What the Learned Words caught in that raw text: counted, and marked
    /// in the Lens, if it is inserted anyway.
    private var lastRejectedCatches: [LearnedWordCatch] = []

    /// The current lifecycle phase — a read-through to the feed, kept as the
    /// coordinator's public state surface for the in-window views and tests.
    var state: DictationFeed.Phase { feed.phase }

    private let session: VoiceCaptureSession
    private let textInjector: any TextInjecting
    private let history: any TranscriptionStoring
    private let settings: SettingsManager

    /// Retained for the **Live Preview** pump (PRD #612) — mid-capture
    /// snapshots and the partial decode lane. The capture/transcribe
    /// *lifecycle* still belongs to the session; the pump only reads.
    private let audioCapture: any AudioCapturing
    private let transcriptionEngine: any Transcribing

    /// The app in front now: read again at the paste, where a fix goes back.
    private let frontmostApp: @MainActor () -> TargetApp?
    /// Held for "insert raw anyway", which commits outside the session, and
    /// for the Live Preview, where Learned Words flip as they arrive.
    private let learnedWords: (any LearnedWordApplying)?
    /// The preview reads like the take will: the same regex cleanup.
    private let postProcessor = TranscriptionPostProcessor()

    /// The **Proofread Pass**, injected by the composition root; `nil` in
    /// tests that don't exercise it. The coordinator wraps it so the feed
    /// narrates the `.proofreading` phase — the session stays feed-blind.
    private let proofreadPass: ProofreadPass?

    /// The **Correction Pair** store (ticket #289); `nil` in tests that don't
    /// exercise the flywheel. Every take is recorded as a candidate; a fix in
    /// the Lens turns it gold.
    private let pairs: CorrectionPairStore?

    /// The pair of the last take that surfaced a beat — what "insert raw
    /// anyway" marks gold.
    private(set) var lastTakePairID: UUID?

    /// Every committed take, handed to the **Lens** (PRD #612) the moment it
    /// lands, so ⌃⌥Space can reopen it. Called right after the paste, before
    /// anything else can type into the app.
    var onTakeCommitted: (@MainActor (DictatedTake) -> Void)?

    /// The maximum-recording-duration auto-stop. Caller-owned: it finalizes a stuck
    /// recording, which is a dictation-presentation concern, not part of the shared
    /// capture lifecycle.
    private var recordingTask: Task<Void, Never>?

    /// The **Live Preview** pump (PRD #612) and its staleness epoch. The
    /// epoch guards a decode that resolves after its take ended against
    /// previewing the *next* take (the feed's phase guard alone can't tell
    /// two recordings apart).
    private var previewTask: Task<Void, Never>?
    private var previewEpoch: UInt64 = 0

    /// ⇧ was tapped while this take was recorded: it waits in the Lens.
    private var holdRequested = false

    /// A hotkey press that arrived while a previous capture was still
    /// `.processing`. Push-to-talk must never swallow a press: the intent is
    /// honored the moment processing resolves — unless the key was released
    /// first, which abandons it (a tap fully inside the processing window has
    /// no audio to offer).
    private var startPending = false

    init(
        audioCapture: any AudioCapturing,
        transcriptionEngine: any Transcribing,
        textInjector: any TextInjecting,
        history: any TranscriptionStoring,
        settings: SettingsManager,
        feed: DictationFeed,
        proofreadPass: ProofreadPass? = nil,
        captureDump: (any CaptureDumpStoring)? = nil,
        pairs: CorrectionPairStore? = nil,
        learnedWords: (any LearnedWordApplying)? = nil,
        frontmostApp: @escaping @MainActor () -> TargetApp? = { TargetApp.frontmost() }
    ) {
        self.frontmostApp = frontmostApp
        self.learnedWords = learnedWords
        self.session = VoiceCaptureSession(
            audioCapture: audioCapture,
            transcriptionEngine: transcriptionEngine,
            captureDump: captureDump,
            isCaptureDumpEnabled: { settings.captureDumpEnabled },
            learnedWords: learnedWords,
            frontmostApp: frontmostApp
        )
        self.textInjector = textInjector
        self.history = history
        self.settings = settings
        self.feed = feed
        self.proofreadPass = proofreadPass
        self.pairs = pairs
        self.audioCapture = audioCapture
        self.transcriptionEngine = transcriptionEngine
    }

    // MARK: - Public API

    func onHotkeyDown() {
        switch state {
        case .idle:
            DictationPerf.markPress()
            startRecording()
        case .error:
            // An error line is feedback, never a gate: the press *is* the retry,
            // so recording starts immediately instead of waiting out the
            // error auto-reset.
            DictationPerf.markPress()
            feed.setPhase(.idle)
            startRecording()
        case .processing, .proofreading:
            startPending = true
        case .recording:
            break
        }
    }

    func onHotkeyUp() {
        startPending = false
        guard state == .recording else { return }
        stopRecordingAndProcess()
    }

    func toggleRecording() {
        switch state {
        case .idle:
            startRecording()
        case .recording:
            stopRecordingAndProcess()
        case .processing, .proofreading:
            // Can't stop while resolving
            break
        case .error:
            // Reset and try again
            feed.setPhase(.idle)
            startRecording()
        }
    }

    func cancel() {
        startPending = false
        recordingTask?.cancel()
        recordingTask = nil
        stopPreviewPump()
        session.cancel()
        feed.setPhase(.idle)
        feed.emit(.cancelled)
    }

    // MARK: - Private

    private func startRecording() {
        // Note (audit #285 item 6): the `.recording` emission lands *after*
        // the synchronous engine start below, but reordering would gain
        // nothing — emission and `AVAudioEngine.start()` complete inside one
        // main-actor job, so the Lens's first frame can't precede either.
        // DictationPerf's press→visible measures the whole job.
        switch session.start() {
        case .started:
            holdRequested = false
            feed.setTargetApp(session.targetApp)
            feed.setPhase(.recording)

            // Start the maximum-duration timeout task.
            recordingTask = Task {
                let maxDuration = settings.maxRecordingDuration
                try? await Task.sleep(for: .seconds(maxDuration))

                if !Task.isCancelled && state == .recording {
                    stopRecordingAndProcess()
                }
            }

            if settings.playSounds {
                playSound(.startRecording)
            }

            startPreviewPump()
        case .micBusy:
            handleError(.microphoneBusy)
        case .captureFailed(let error):
            if let dictationError = error as? DictationError {
                handleError(dictationError)
            } else {
                handleError(.audioCaptureFailed(error.localizedDescription))
            }
        }
    }

    // MARK: - Live Preview (PRD #612)

    /// The pause between a decode landing and the next snapshot. Cadence is
    /// self-pacing: a slow decode simply stretches its own cycle.
    private static let previewInterval: Duration = .milliseconds(300)

    /// Decodes the take while it is recorded, from the end of its last
    /// confirmed segment, and publishes what Whisper heard with the Learned
    /// Words applied (shown, not counted). Release cancels the decode in
    /// flight; the paste is always the full pass.
    private func startPreviewPump() {
        previewEpoch &+= 1
        let epoch = previewEpoch
        let app = session.targetApp
        previewTask = Task { [weak self] in
            var assembler = LivePreviewAssembler()
            while !Task.isCancelled {
                guard let self, self.previewEpoch == epoch, self.state == .recording
                else { return }
                if let tail = self.audioCapture.captureSnapshot(from: assembler.confirmedEnd),
                    let window = assembler.window(ofTail: tail)
                {
                    let decodeStart = DispatchTime.now()
                    let result = await self.transcriptionEngine.transcribePartial(
                        window, language: self.settings.language)
                    // The decode awaited: this take may have ended (and another
                    // begun) meanwhile — a stale preview is worse than none.
                    guard !Task.isCancelled, self.previewEpoch == epoch,
                        self.state == .recording
                    else { return }
                    if let result {
                        DictationPerf.record(
                            span: "preview", ms: DictationPerf.msSince(decodeStart))
                        assembler.fold(result, windowDuration: window.duration)
                        self.feed.setPreview(self.preview(of: assembler, app: app))
                    }
                }
                try? await Task.sleep(for: Self.previewInterval)
            }
        }
    }

    private func stopPreviewPump() {
        previewEpoch &+= 1
        previewTask?.cancel()
        previewTask = nil
        feed.setPreview(nil)
    }

    /// The preview as the take will read: the regex cleanup, then the
    /// Learned Words (so a learned word flips the moment it is heard).
    private func preview(of assembler: LivePreviewAssembler, app: TargetApp?) -> LivePreview? {
        let cleaned = postProcessor.process(assembler.rawText)
        guard !cleaned.isEmpty else { return nil }
        let learned =
            learnedWords?.apply(to: cleaned, appBundleID: app?.bundleID) ?? .unchanged(cleaned)
        // The confirmed words are the preview's leading words that match the
        // confirmed text on its own (cleaned the same way): where the two
        // cleanups differ at the boundary, the word counts as provisional.
        let confirmed = postProcessor.process(assembler.confirmed)
        let confirmedText =
            learnedWords?.apply(to: confirmed, appBundleID: app?.bundleID).text ?? confirmed
        let confirmedWords = zip(TakeText.tokens(confirmedText), TakeText.tokens(learned.text))
            .prefix { $0.bare == $1.bare }.count
        return LivePreview(
            text: learned.text, catches: learned.catches, confirmedTokens: confirmedWords)
    }

    // MARK: - Holding a take (PRD #612)

    /// ⇧ tapped while recording: the take waits in the Lens instead of
    /// pasting; a second tap lets it paste again. Ignored when the setting
    /// says never or always.
    func shiftTapped() {
        guard state == .recording, settings.checkBeforePasting == .whenShiftTapped else { return }
        holdRequested.toggle()
        feed.setHeld(holdRequested)
    }

    /// Whether this take waits in the Lens rather than pasting.
    private var holdsTake: Bool {
        guard settings.autoInsertText else { return false }
        switch settings.checkBeforePasting {
        case .always: return true
        case .never: return false
        case .whenShiftTapped: return holdRequested
        }
    }

    private func stopRecordingAndProcess() {
        recordingTask?.cancel()
        recordingTask = nil
        stopPreviewPump()

        DictationPerf.markRelease()
        let stopStart = DispatchTime.now()
        let stopResult = session.stop()
        DictationPerf.record(span: "stop", ms: DictationPerf.msSince(stopStart))
        switch stopResult {
        case .noAudio:
            handleError(.audioCaptureFailed("No audio captured"))
            DictationPerf.markResolved("error(noAudio)")
        case .tooShort:
            handleError(.recordingTooShort)
            DictationPerf.markResolved("error(tooShort)")
        case .silent:
            // Nothing was said: no transcription, so no "Thank you." pasted
            // out of silence.
            handleError(.noSpeechDetected)
            feed.emit(.empty)
            DictationPerf.markResolved("error(silent)")
        case .audio(let audioData, let dumpFile):
            process(audioData, dumpFile: dumpFile)
        }
    }

    private func process(_ audioData: AudioData, dumpFile: String?) {
        feed.setPhase(.processing)

        // Fire-and-forget: the session owns the in-flight task and its cancellation,
        // so this outer task is untracked — it only maps the outcome back to state.
        Task {
            let sessionStart = DispatchTime.now()
            var committedDuration: TimeInterval = 0

            // The proofread wrapper narrates the phase around the pass, so
            // the Lens can show "finishing" — the session stays feed-blind.
            var proofread: (@MainActor (String) async -> ProofreadVerdict?)?
            if let pass = proofreadPass {
                proofread = { [feed] text in
                    feed.setPhase(.proofreading)
                    let proofreadStart = DispatchTime.now()
                    let verdict = await pass.proofread(text)
                    DictationPerf.record(
                        span: "proofread", ms: DictationPerf.msSince(proofreadStart))
                    if feed.phase == .proofreading {
                        feed.setPhase(.processing)
                    }
                    return verdict
                }
            }

            // Record the take as a Correction Pair candidate the moment its
            // lineage is known (before the commit, so the history entry can
            // link to it). Every take is a candidate — the flywheel collects
            // from day one; flags and edits turn candidates gold.
            var recordedPairID: UUID?
            var observedTake: VoiceCaptureSession.Take?
            let onTake: @MainActor (VoiceCaptureSession.Take) -> Void = { [settings, pairs] take in
                observedTake = take
                if let pairs {
                    let pair = CorrectionPair(
                        rawASR: take.rawASR,
                        cleaned: take.cleaned,
                        learned: take.catches.isEmpty ? nil : take.learned,
                        proofread: {
                            if case .corrected(let text, _) = take.verdict { return text }
                            return nil
                        }(),
                        verdict: Self.pairVerdict(from: take.verdict),
                        rejectReason: {
                            if case .rejected(let reason) = take.verdict { return reason }
                            return nil
                        }(),
                        committed: take.committedText,
                        conditions: CorrectionPair.Conditions(
                            duration: audioData.duration,
                            language: settings.language,
                            asrModel: ModelDefinition.withID(
                                settings.selectedSpeechToTextModelID)?.displayName
                                ?? settings.selectedSpeechToTextModelID
                        ),
                        audioFileName: dumpFile
                    )
                    pairs.record(pair)
                    recordedPairID = pair.id
                }
            }

            let outcome = await session.transcribeAndCommit(
                audioData, language: settings.language, proofread: proofread,
                onTake: onTake
            ) { [self] text, duration in
                lastTranscription = text
                committedDuration = duration
                let catches = Self.catches(of: observedTake, committed: text)

                history.add(
                    text: text,
                    duration: duration,
                    model: ModelDefinition.withID(settings.selectedSpeechToTextModelID)?.displayName
                        ?? settings.selectedSpeechToTextModelID,
                    pairID: recordedPairID,
                    catches: catches,
                    app: Self.historyApp(observedTake?.app)
                )

                // A held take waits in the Lens: it pastes from there.
                let held = holdsTake
                let pastes = settings.autoInsertText && !held
                if pastes {
                    textInjector.restoreClipboard = settings.restoreClipboard
                    let injectStart = DispatchTime.now()
                    try await textInjector.inject(text + " ")
                    DictationPerf.record(
                        span: "inject", ms: DictationPerf.msSince(injectStart))
                }
                // Straight after the paste, before another key can reach the
                // app: the Lens anchors "still the last thing typed" here.
                onTakeCommitted?(
                    DictatedTake(
                        pairID: recordedPairID, text: text, catches: catches,
                        app: observedTake?.app, pasted: pastes,
                        pastedInto: pastes ? frontmostApp() : nil, held: held))
            }
            DictationPerf.record(span: "session", ms: DictationPerf.msSince(sessionStart))

            switch outcome {
            case .committed(let edits):
                if settings.playSounds {
                    playSound(.success)
                }
                lastRejectedRaw = nil
                lastTakePairID = recordedPairID
                feed.setPhase(.idle)
                feed.emit(
                    .committed(
                        text: lastTranscription, duration: committedDuration, edits: edits))
                DictationPerf.markResolved("committed")
                drainPendingStart()
            case .rejected(let raw, let reason):
                // Passive by design (map #283): the press is the retry, so the
                // phase returns to idle — no error gate. The beat carries the
                // raw text for "insert raw anyway".
                lastRejectedRaw = raw
                lastRejectedCatches = observedTake?.catches ?? []
                lastTakePairID = recordedPairID
                feed.setPhase(.idle)
                feed.emit(.rejected(raw: raw, reason: reason))
                DictationPerf.markResolved("rejected")
                drainPendingStart()
            case .empty:
                handleError(.noSpeechDetected)
                feed.emit(.empty)
                DictationPerf.markResolved("error(noSpeech)")
                drainPendingStart()
            case .failed(let error):
                if let dictationError = error as? DictationError {
                    handleError(dictationError)
                } else {
                    handleError(.transcriptionFailed(error.localizedDescription))
                }
                DictationPerf.markResolved("error(failed)")
                drainPendingStart()
            case .cancelled:
                feed.setPhase(.idle)
                feed.emit(.cancelled)
                DictationPerf.markResolved("cancelled")
            case .superseded:
                // A cancel-and-restart superseded this operation — the newer
                // operation owns the state; commit nothing and leave it untouched.
                feed.emit(.superseded)
                DictationPerf.markResolved("superseded")
            }
        }
    }

    /// Injects the raw text of the last rejected take — the "insert raw
    /// anyway" affordance (map #283). Using it *is* "the pass was wrong":
    /// the take's pair is flagged gold, which also protects its audio.
    func insertRawAnyway() {
        guard let raw = lastRejectedRaw else { return }
        lastRejectedRaw = nil
        if let lastTakePairID {
            pairs?.flagWrong(lastTakePairID)
        }
        // The raw text is now the last take: ⌃⌥Space must reopen it, and a
        // fix must measure against what this paste typed, not the take before.
        let catches = lastRejectedCatches
        lastRejectedCatches = []
        history.add(
            text: raw,
            duration: 0,
            model: ModelDefinition.withID(settings.selectedSpeechToTextModelID)?.displayName
                ?? settings.selectedSpeechToTextModelID,
            pairID: lastTakePairID,
            catches: catches,
            app: Self.historyApp(session.targetApp)
        )
        let take = DictatedTake(
            pairID: lastTakePairID, text: raw, catches: catches, app: session.targetApp,
            pasted: settings.autoInsertText,
            pastedInto: settings.autoInsertText ? frontmostApp() : nil)
        guard settings.autoInsertText else {
            learnedWords?.recordCatches(catches)
            onTakeCommitted?(take)
            return
        }
        textInjector.restoreClipboard = settings.restoreClipboard
        Task {
            // Surface a failed injection: the loan can refuse to borrow the
            // pasteboard (unreadable, or an earlier return still unrestored),
            // and a silent no-op here reads as the button doing nothing.
            do {
                try await textInjector.inject(raw + " ")
                learnedWords?.recordCatches(catches)
                onTakeCommitted?(take)
            } catch let error as DictationError {
                handleError(error)
            } catch {
                handleError(.textInjectionFailed(error.localizedDescription))
            }
        }
    }

    /// The take's catches, positioned in the committed text (the Proofread
    /// Pass may have rewritten it after the Learned Words ran).
    private static func catches(
        of take: VoiceCaptureSession.Take?, committed text: String
    ) -> [LearnedWordCatch] {
        guard let take, !take.catches.isEmpty else { return [] }
        return take.learned == text
            ? take.catches : LearnedWordMatcher.relocate(take.catches, in: text)
    }

    /// The app as the history keeps it (no pid: it is stale by the time the
    /// Dictation page opens the take).
    private static func historyApp(_ app: TargetApp?) -> TranscriptionEntry.App? {
        app.map { TranscriptionEntry.App(bundleID: $0.bundleID, name: $0.name) }
    }

    private static func pairVerdict(
        from verdict: ProofreadVerdict?
    ) -> CorrectionPair.Verdict {
        switch verdict {
        case .corrected: return .corrected
        case .rejected: return .rejected
        case .unchanged: return .unchanged
        case nil: return .skipped
        }
    }

    /// Honors a hotkey press that arrived mid-`.processing`: the key is still
    /// held (release clears the flag), so recording starts now. An error the
    /// resolution just raised does not gate — same rule as a press on an idle
    /// error line, and the new recording replaces it.
    private func drainPendingStart() {
        guard startPending else { return }
        startPending = false
        if case .error = state { feed.setPhase(.idle) }
        guard state == .idle else { return }
        startRecording()
    }

    private func handleError(_ error: DictationError) {
        feed.setPhase(.error(error))

        if settings.playSounds {
            playSound(.error)
        }

        // Auto-reset after a delay (shared duration so dictation and agent voice
        // input don't drift on how long an error lingers).
        Task {
            try? await Task.sleep(for: ErrorAutoReset.delay)
            if case .error = state {
                feed.setPhase(.idle)
            }
        }
    }

    /// Preloaded once: `NSSound(named:)` loads from disk on first use, and the
    /// start-recording play sits on the same main-actor job that precedes the
    /// Lens's state emission — a per-press load would delay the Lens.
    private let sounds: [SystemSound: NSSound] = [
        SystemSound.startRecording: NSSound(named: "Tink"),
        SystemSound.success: NSSound(named: "Purr"),
        SystemSound.error: NSSound(named: "Funk"),
    ].compactMapValues { $0 }

    private func playSound(_ sound: SystemSound) {
        guard let nsSound = sounds[sound] else { return }
        if nsSound.isPlaying {
            nsSound.stop()
        }
        nsSound.play()
    }

    private enum SystemSound {
        case startRecording
        case success
        case error
    }
}
