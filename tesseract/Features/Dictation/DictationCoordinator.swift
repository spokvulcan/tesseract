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
/// The coordinator is the feed's sole phase/beat writer; overlay variants and
/// the in-window dictation views read the feed, never the coordinator's
/// internals.
@Observable @MainActor
final class DictationCoordinator {
    let feed: DictationFeed
    private(set) var lastTranscription: String = ""

    /// The raw text of the last take the **Proofread Pass** rejected — kept
    /// so "insert raw anyway" (an overlay affordance, or this API directly)
    /// can still deliver the user's words.
    private(set) var lastRejectedRaw: String?

    /// The current lifecycle phase — a read-through to the feed, kept as the
    /// coordinator's public state surface for the in-window views and tests.
    var state: DictationFeed.Phase { feed.phase }

    private let session: VoiceCaptureSession
    private let textInjector: any TextInjecting
    private let history: any TranscriptionStoring
    private let settings: SettingsManager

    /// Retained for the **Live Partial** pump (ticket #291) — mid-capture
    /// snapshots and the partial decode lane. The capture/transcribe
    /// *lifecycle* still belongs to the session; the pump only reads.
    private let audioCapture: any AudioCapturing
    private let transcriptionEngine: any Transcribing

    /// The **Proofread Pass**, injected by the composition root; `nil` in
    /// tests that don't exercise it. The coordinator wraps it so the feed
    /// narrates the `.proofreading` phase — the session stays feed-blind.
    private let proofreadPass: ProofreadPass?

    /// The **Correction Pair** store (ticket #289); `nil` in tests that don't
    /// exercise the flywheel. Every take is recorded as a candidate; the
    /// overlay affordances and the history editor turn candidates gold.
    private let pairs: CorrectionPairStore?

    /// The living memory (ADR-0035). Dictation is a memory write source by the
    /// owner's explicit call (map #314): what he dictates into other apps is
    /// often the most revealing thing he says all day, and a memory that only
    /// hears him when he is *talking to the assistant* knows a stranger.
    ///
    /// Gated twice — `memoryEnabled` and `memoryCaptureDictation`, both checked
    /// inside the engine — because this is the one source that captures speech
    /// the owner did not aim at this app. Speech he *did* aim at this app is
    /// skipped at commit: the chat door records it with the reply attached, and
    /// one utterance must not enter the store through two doors.
    private let memory: MemoryEngine?

    /// The pair of the last take that surfaced a beat — what the overlay's
    /// flag/edit affordances target while the beat lingers.
    private(set) var lastTakePairID: UUID?

    /// The overlay "edit" affordance's window hook: summon the main window
    /// onto the dictation page. Set by the app delegate (window management
    /// is its turf); the coordinator only decides *when*.
    var onOpenDictationHistory: (@MainActor () -> Void)?

    /// The maximum-recording-duration auto-stop. Caller-owned: it finalizes a stuck
    /// recording, which is a dictation-presentation concern, not part of the shared
    /// capture lifecycle.
    private var recordingTask: Task<Void, Never>?

    /// Whether the live Overlay Variant consumes the feed's partial signal.
    /// Set by the composition root (which knows the selected variant); the
    /// coordinator reads a policy, never the variant — the pipeline stays
    /// variant-blind. Default off: no pump, zero cost, baselines untouched.
    var isLivePartialsEnabled: @MainActor () -> Bool = { false }

    /// The **Live Partial** pump (ticket #291) and its staleness epoch. The
    /// epoch guards a decode that resolves after its take ended against
    /// captioning the *next* take (the feed's phase guard alone can't tell
    /// two recordings apart).
    private var partialTask: Task<Void, Never>?
    private var partialEpoch: UInt64 = 0

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
        memory: MemoryEngine? = nil
    ) {
        self.memory = memory
        self.session = VoiceCaptureSession(
            audioCapture: audioCapture,
            transcriptionEngine: transcriptionEngine,
            captureDump: captureDump,
            isCaptureDumpEnabled: { settings.captureDumpEnabled }
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
            // An error pill is feedback, never a gate: the press *is* the retry,
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
        stopPartialPump()
        session.cancel()
        feed.setPhase(.idle)
        feed.emit(.cancelled)
    }

    // MARK: - Private

    private func startRecording() {
        // Note (audit #285 item 6): the `.recording` emission lands *after*
        // the synchronous engine start below, but reordering would gain
        // nothing — emission and `AVAudioEngine.start()` complete inside one
        // main-actor job, so the pill's first frame can't precede either.
        // DictationPerf's press→visible measures the whole job.
        switch session.start() {
        case .started:
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

            startPartialPump()
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

    // MARK: - Live Partial pump (ticket #291)

    /// How much trailing audio a partial decode sees. Capping the window
    /// bounds the decode (and the worst-case delay a just-released final pays
    /// waiting out a partial's cancellation) regardless of take length; the
    /// caption shows the tail anyway.
    private static let partialWindowSeconds: TimeInterval = 12
    /// No decode before this much audio exists — sub-second snippets return
    /// noise, and the first words deserve one clean pass.
    private static let partialMinimumAudio: TimeInterval = 0.6
    /// The pause between a decode landing and the next snapshot. Cadence is
    /// self-pacing: a slow decode simply stretches its own cycle.
    private static let partialInterval: Duration = .milliseconds(300)

    private func startPartialPump() {
        guard isLivePartialsEnabled() else { return }
        partialEpoch &+= 1
        let epoch = partialEpoch
        partialTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self, self.partialEpoch == epoch, self.state == .recording
                else { return }
                if let snapshot = self.audioCapture.captureSnapshot(),
                    snapshot.duration >= Self.partialMinimumAudio
                {
                    let decodeStart = DispatchTime.now()
                    let text = await self.transcriptionEngine.transcribePartial(
                        Self.trailingWindow(of: snapshot), language: self.settings.language)
                    // The decode awaited: this take may have ended (and another
                    // begun) meanwhile — a stale caption is worse than none.
                    guard !Task.isCancelled, self.partialEpoch == epoch,
                        self.state == .recording
                    else { return }
                    if let text, !text.isEmpty {
                        DictationPerf.record(
                            span: "partial", ms: DictationPerf.msSince(decodeStart))
                        self.feed.setPartial(text)
                    }
                }
                try? await Task.sleep(for: Self.partialInterval)
            }
        }
    }

    private func stopPartialPump() {
        partialEpoch &+= 1
        partialTask?.cancel()
        partialTask = nil
        feed.setPartial(nil)
    }

    /// The trailing `partialWindowSeconds` of a snapshot (whole snapshot when
    /// shorter). `raw` stays nil — a partial is never Capture Dump evidence.
    private static func trailingWindow(of audio: AudioData) -> AudioData {
        let maxSamples = Int(audio.sampleRate * partialWindowSeconds)
        guard audio.samples.count > maxSamples else { return audio }
        return AudioData(
            samples: Array(audio.samples.suffix(maxSamples)),
            sampleRate: audio.sampleRate,
            duration: partialWindowSeconds,
            raw: nil
        )
    }

    private func stopRecordingAndProcess() {
        recordingTask?.cancel()
        recordingTask = nil
        stopPartialPump()

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
            // variants can show "polishing" — the session stays feed-blind.
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
            var onTake: (@MainActor (VoiceCaptureSession.Take) -> Void)?
            if let pairs {
                onTake = { [settings] take in
                    let pair = CorrectionPair(
                        rawASR: take.rawASR,
                        cleaned: take.cleaned,
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

                history.add(
                    text: text,
                    duration: duration,
                    model: ModelDefinition.withID(settings.selectedSpeechToTextModelID)?.displayName
                        ?? settings.selectedSpeechToTextModelID,
                    pairID: recordedPairID
                )

                // Memory rides the *committed* text, not the raw ASR: what the
                // owner actually meant to say, after proofreading, is the thing
                // worth remembering. Detached — dictation's whole promise is that
                // the words land in the frontmost app instantly, and nothing in
                // the memory system gets to stand between the take and the
                // keystroke. Words headed into Tesseract itself are the one
                // exception: the chat door captures what he *sends* with the
                // reply attached, and the same utterance must not become two
                // episodes (one door per testimony).
                let targetIsSelf =
                    NSWorkspace.shared.frontmostApplication?.bundleIdentifier
                    == Bundle.main.bundleIdentifier
                if let memory, !targetIsSelf {
                    Task { [memory] in
                        await memory.record(
                            source: .dictation, text: text,
                            meta: ["duration": String(format: "%.1f", duration)])
                    }
                }

                if settings.autoInsertText {
                    textInjector.restoreClipboard = settings.restoreClipboard
                    let injectStart = DispatchTime.now()
                    try await textInjector.inject(text + " ")
                    DictationPerf.record(
                        span: "inject", ms: DictationPerf.msSince(injectStart))
                }
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
        history.add(
            text: raw,
            duration: 0,
            model: ModelDefinition.withID(settings.selectedSpeechToTextModelID)?.displayName
                ?? settings.selectedSpeechToTextModelID,
            pairID: lastTakePairID
        )
        guard settings.autoInsertText else { return }
        textInjector.restoreClipboard = settings.restoreClipboard
        Task {
            try? await textInjector.inject(raw + " ")
        }
    }

    /// The overlay's one-click "that was wrong" on the lingering beat. No
    /// focus steal, no window — it marks the pair gold (protecting its
    /// Capture Dump audio) and nothing else.
    func flagLastTakeWrong() {
        guard let lastTakePairID else { return }
        pairs?.flagWrong(lastTakePairID)
    }

    /// The overlay's "edit" on the lingering beat: reveal the take's entry in
    /// the history (full editing lives there — the overlay stays
    /// keyboard-free) and summon the window via the delegate hook.
    func editLastTake() {
        guard let lastTakePairID else { return }
        history.requestFocus(pairID: lastTakePairID)
        onOpenDictationHistory?()
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
    /// error pill, and the new recording replaces it.
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
            try? await Task.sleep(for: VoiceCaptureSession.errorAutoResetDelay)
            if case .error = state {
                feed.setPhase(.idle)
            }
        }
    }

    /// Preloaded once: `NSSound(named:)` loads from disk on first use, and the
    /// start-recording play sits on the same main-actor job that precedes the
    /// pill's state emission — a per-press load would delay the pill.
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
