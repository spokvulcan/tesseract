//
//  DictationCoordinator.swift
//  tesseract
//

import AppKit
import Foundation
import Observation

/// The global system-wide dictation driver. A thin presenter over its own
/// **Voice Capture Session** (from **Voice Capture**, so its takes run every
/// stage): it maps the session's phases, `StopResult` and `Outcome` onto the
/// **Overlay Feed**'s phases and beats, and keeps only what is
/// dictation-specific — its commit (history, the **Correction Pair**, the
/// paste or the held take, the hand-off to the Lens), success/error sounds,
/// the maximum-recording-duration auto-stop, `DictationError` mapping, and the
/// error auto-reset. Distinct from **Voice Input** (`AgentVoiceInputController`),
/// which presents a session of its own in a field.
///
/// The coordinator is the feed's sole phase/beat writer; the Lens and
/// the in-window dictation views read the feed, never the coordinator's
/// internals.
@Observable @MainActor
final class DictationCoordinator {
    let feed: DictationFeed
    private(set) var lastTranscription: String = ""

    /// The current lifecycle phase — a read-through to the feed, kept as the
    /// coordinator's public state surface for the in-window views and tests.
    var state: DictationFeed.Phase { feed.phase }

    private let session: VoiceCaptureSession
    private let textInjector: any TextInjecting
    private let history: any TranscriptionStoring
    private let settings: SettingsManager

    /// The app in front now: read again at the paste, where a fix goes back.
    private let frontmostApp: @MainActor () -> TargetApp?

    /// When the Proofread Pass began reading the take, for its span.
    private var proofreadStart: DispatchTime?

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

    /// ⇧ was tapped while this take was recorded: it waits in the Lens.
    private var holdRequested = false

    /// A hotkey press that arrived while a previous capture was still
    /// `.processing`. Push-to-talk must never swallow a press: the intent is
    /// honored the moment processing resolves — unless the key was released
    /// first, which abandons it (a tap fully inside the processing window has
    /// no audio to offer).
    private var startPending = false

    init(
        session: VoiceCaptureSession,
        textInjector: any TextInjecting,
        history: any TranscriptionStoring,
        settings: SettingsManager,
        feed: DictationFeed,
        pairs: CorrectionPairStore? = nil,
        frontmostApp: @escaping @MainActor () -> TargetApp? = { TargetApp.frontmost() }
    ) {
        self.session = session
        self.frontmostApp = frontmostApp
        self.textInjector = textInjector
        self.history = history
        self.settings = settings
        self.feed = feed
        self.pairs = pairs
        session.onPhaseChange = { [weak self] phase in self?.sessionPhaseChanged(phase) }
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
        // The Live Preview (ADR-0086): the session decodes the take while it
        // is recorded, read the way the take will read; release ends it.
        switch session.start(preview: { [feed] in feed.setPreview($0) }) {
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

    /// The session's take, narrated in the feed: the Lens shows the Proofread
    /// Pass as "finishing".
    private func sessionPhaseChanged(_ phase: VoiceCaptureSession.Phase) {
        switch phase {
        case .proofreading:
            proofreadStart = .now()
            feed.setPhase(.proofreading)
        case .transcribing:
            guard let start = proofreadStart else { return }
            proofreadStart = nil
            DictationPerf.record(span: "proofread", ms: DictationPerf.msSince(start))
            if feed.phase == .proofreading { feed.setPhase(.processing) }
        case .idle, .listening:
            proofreadStart = nil
        }
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

            // Record the take as a Correction Pair candidate the moment its
            // lineage is known (before the commit, so the history entry can
            // link to it). Every take is a candidate — the flywheel collects
            // from day one; flags and edits turn candidates gold.
            var recordedPairID: UUID?
            let onTake: @MainActor (VoiceCaptureSession.Take) -> Void = { [settings, pairs] take in
                guard let pairs else { return }
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

            let outcome = await session.transcribeAndCommit(audioData, onTake: onTake) {
                [self] take in
                lastTranscription = take.text
                committedDuration = take.duration
                // A held take waits in the Lens: it pastes from there.
                try await commit(take, pairID: recordedPairID, held: holdsTake)
            }
            DictationPerf.record(span: "session", ms: DictationPerf.msSince(sessionStart))

            switch outcome {
            case .committed(let edits):
                if settings.playSounds {
                    playSound(.success)
                }
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
                // raw text, and the session keeps the take for "insert anyway".
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

    /// Dictation's commit, for a take that lands and for one inserted
    /// anyway: the history entry linked to its pair, the paste unless the
    /// take waits in the Lens, and the hand-off to the Lens straight after the
    /// paste, before another key can reach the app (the Lens anchors "still
    /// the last thing typed" there).
    private func commit(
        _ take: VoiceCaptureSession.Commit, pairID: UUID?, held: Bool
    ) async throws {
        history.add(
            text: take.text,
            duration: take.duration,
            model: ModelDefinition.withID(settings.selectedSpeechToTextModelID)?.displayName
                ?? settings.selectedSpeechToTextModelID,
            pairID: pairID,
            catches: take.catches,
            app: Self.historyApp(take.app)
        )
        let pastes = settings.autoInsertText && !held
        if pastes {
            textInjector.restoreClipboard = settings.restoreClipboard
            let injectStart = DispatchTime.now()
            try await textInjector.inject(take.text + " ")
            DictationPerf.record(span: "inject", ms: DictationPerf.msSince(injectStart))
        }
        onTakeCommitted?(
            DictatedTake(
                pairID: pairID, text: take.text, catches: take.catches, app: take.app,
                pasted: pastes, pastedInto: pastes ? frontmostApp() : nil, held: held))
    }

    /// "Insert anyway" on the last take the Proofread Pass rejected (map
    /// #283): the session commits it like any take, through dictation's
    /// commit, and it pastes. Using it *is* "the pass was wrong": the take's
    /// pair is flagged gold, which also protects its audio.
    func insertRawAnyway() {
        guard session.hasRejectedTake else { return }
        if let lastTakePairID {
            pairs?.flagWrong(lastTakePairID)
        }
        let pairID = lastTakePairID
        Task {
            let outcome = await session.commitRejectedTake { [self] take in
                try await commit(take, pairID: pairID, held: false)
            }
            // Surface a failed injection: the loan can refuse to borrow the
            // pasteboard (unreadable, or an earlier return still unrestored),
            // and a silent no-op here reads as the button doing nothing.
            guard case .failed(let error) = outcome else { return }
            if let dictationError = error as? DictationError {
                handleError(dictationError)
            } else {
                handleError(.textInjectionFailed(error.localizedDescription))
            }
        }
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
    /// Lens's state emission — a per-press load would delay the Lens. None
    /// where the Launch Mode plays no sounds: the test host drives hundreds of
    /// takes and nobody listens.
    private let sounds: [SystemSound: NSSound] =
        !LaunchMode.current.playsSounds
        ? [:]
        : [
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
