//
//  VoiceCaptureSession.swift
//  tesseract
//
//  **Voice Capture** (ADR-0096): the one module every push-to-talk surface
//  takes its takes from — dictation, the agent composer, Today's composer, the
//  capture and Jarvis panels, the voice session, the Onboarding Tour. It is
//  built once from the shared microphone and recognizer, the Capture Dump and
//  its switch, the Learned Words, the Proofread Pass and the dictation
//  language, every one required, and hands each surface its own **Voice
//  Capture Session** with every stage. No surface can build a take that skips
//  one (ADR-0085 §2).
//
//  The session owns the take: the capture→transcribe→commit lifecycle with the
//  **Operation Guard** ticket discipline re-checked after every `await`, the
//  microphone-busy, minimum-duration, silence and empty-text guards, the
//  take's one text pipeline (the regex cleanup, then the Learned Words) for
//  both the **Live Preview** and the final pass (ADR-0086 §3), the Proofread
//  Pass as a phase of the take, "insert anyway" for a take the pass rejected,
//  the in-flight `Task`, and cancellation. Each surface keeps only what is its
//  own: its commit, its presentation, its sounds and its errors.
//
//  Whether a take is open — listening, transcribing, proofreading — is one
//  fact per session (`phase`), and one across them (`VoiceCapture`), which the
//  menu bar's Models section reads.
//

import Foundation
import Observation

/// Builds every push-to-talk surface's **Voice Capture Session** from one set
/// of stages, and says whether any of their takes is open.
@Observable @MainActor
final class VoiceCapture {
    @ObservationIgnored private let stages: Stages
    /// Every session made so far, held weakly: a surface owns its session.
    private var sessions: [WeakSession] = []

    init(
        microphone: any AudioCapturing,
        recognizer: any Transcribing,
        captureDump: any CaptureDumpStoring,
        isCaptureDumpEnabled: @escaping @MainActor () -> Bool,
        learnedWords: any LearnedWordApplying,
        proofread: @escaping @MainActor (String) async -> ProofreadVerdict?,
        language: @escaping @MainActor () -> String,
        frontmostApp: @escaping @MainActor () -> TargetApp? = { TargetApp.frontmost() }
    ) {
        stages = Stages(
            microphone: microphone, recognizer: recognizer, captureDump: captureDump,
            isCaptureDumpEnabled: isCaptureDumpEnabled, learnedWords: learnedWords,
            proofread: proofread, language: language, frontmostApp: frontmostApp)
    }

    /// A surface's own session, with every stage.
    func makeSession() -> VoiceCaptureSession {
        let session = VoiceCaptureSession(stages: stages)
        sessions.removeAll { $0.session == nil }
        sessions.append(WeakSession(session: session))
        return session
    }

    /// A take is being recorded on some surface.
    var isListening: Bool { isAnySession(in: .listening) }
    /// A take is being transcribed and committed on some surface.
    var isTranscribing: Bool { isAnySession(in: .transcribing) }
    /// The Proofread Pass is reading a take on some surface.
    var isProofreading: Bool { isAnySession(in: .proofreading) }

    private func isAnySession(in phase: VoiceCaptureSession.Phase) -> Bool {
        sessions.contains { $0.session?.phase == phase }
    }

    private struct WeakSession {
        weak var session: VoiceCaptureSession?
    }
}

/// The stages every take runs through, built once by `VoiceCapture`.
@MainActor
private struct Stages {
    let microphone: any AudioCapturing
    let recognizer: any Transcribing
    let captureDump: any CaptureDumpStoring
    let isCaptureDumpEnabled: @MainActor () -> Bool
    let learnedWords: any LearnedWordApplying
    let proofread: @MainActor (String) async -> ProofreadVerdict?
    let language: @MainActor () -> String
    let frontmostApp: @MainActor () -> TargetApp?
    let postProcessor = TranscriptionPostProcessor()

    /// A take's text as it will read: the regex cleanup, then the Learned
    /// Words for the take's app. The Live Preview and the final pass both
    /// read through here, so the preview reads like the take will (ADR-0086).
    func read(_ heard: String, in app: TargetApp?) -> (
        cleaned: String, learned: LearnedWordsApplication
    ) {
        let cleaned = postProcessor.process(heard)
        guard !cleaned.isEmpty else { return (cleaned, .unchanged(cleaned)) }
        return (cleaned, learnedWords.apply(to: cleaned, appBundleID: app?.bundleID))
    }
}

/// One surface's push-to-talk takes, from `VoiceCapture.makeSession()`.
@Observable @MainActor
final class VoiceCaptureSession {
    /// A capture below this is rejected by ``stop()`` before any transcription runs
    /// — an accidental tap must not show a spurious "processing" flash.
    private static let minimumRecordingDuration: TimeInterval = 0.5

    /// The pause between a preview decode landing and the next snapshot.
    /// Cadence is self-pacing: a slow decode simply stretches its own cycle.
    private static let previewInterval: Duration = .milliseconds(300)

    /// Where the session's take is.
    enum Phase: Equatable, Sendable {
        case idle
        /// The microphone is open for the take.
        case listening
        /// The take is being transcribed and committed.
        case transcribing
        /// The Proofread Pass is reading the take.
        case proofreading
    }

    /// The outcome of ``start(preview:)``.
    enum StartResult {
        /// Capture began; the operation epoch was advanced.
        case started
        /// The shared microphone is already capturing — nothing started.
        case micBusy
        /// `startCapture()` threw; the error is surfaced for the caller to map.
        case captureFailed(any Error)
    }

    /// The outcome of ``stop()``.
    enum StopResult {
        /// Capture produced audio at or above the minimum duration.
        /// `dumpFile` is the Capture Dump recording this stop saved (nil when
        /// the dump is disabled) — the **Correction Pair**'s audio reference.
        case audio(AudioData, dumpFile: String?)
        /// The capture was shorter than the minimum duration.
        case tooShort
        /// The capture engine returned no audio at all.
        case noAudio
        /// The capture's level never rose above silence. Not transcribed:
        /// Whisper writes "Thank you." over silence, and WhisperKit 1.0.0
        /// never computes the no-speech probability that would catch it.
        case silent
    }

    /// One take's full text lineage, observed via `transcribeAndCommit`'s
    /// `onTake` just before the commit (or the rejected return) — the
    /// **Correction Pair**'s source. Only the session sees the raw ASR text,
    /// so only the session can hand it out.
    struct Take: Sendable {
        /// The recognizer's text before any cleanup.
        let rawASR: String
        /// The regex post-processor's output.
        let cleaned: String
        /// `cleaned` after the **Learned Words** — what the Proofread Pass
        /// saw. Equal to `cleaned` when nothing was caught.
        let learned: String
        /// What the Learned Words caught, positioned in `learned`.
        let catches: [LearnedWordCatch]
        /// The Proofread Pass's verdict; `nil` when the pass was skipped.
        let verdict: ProofreadVerdict?
        /// What commits; `nil` for a rejected take.
        let committedText: String?
        /// The app in front when the take started.
        let app: TargetApp?
    }

    /// What a surface's `commit` receives: the text, the take's length, what
    /// the Learned Words caught in that text (where they now stand), and the
    /// app in front when the take started.
    struct Commit: Sendable {
        let text: String
        let duration: TimeInterval
        let catches: [LearnedWordCatch]
        let app: TargetApp?
    }

    /// The outcome of ``transcribeAndCommit(_:onTake:commit:)``.
    enum Outcome {
        /// Transcription succeeded, produced non-empty text, and the injected
        /// `commit` ran to completion while this operation was still current.
        /// `edits` is the **Proofread Pass**'s word-swap diff (empty when the
        /// pass was skipped or changed nothing).
        case committed(edits: [WordEdit])
        /// Post-processing yielded empty text — no speech to commit.
        case empty
        /// The **Proofread Pass** judged the take unintelligible: nothing was
        /// committed; ``commitRejectedTake(_:)`` still can. `raw` is its text
        /// as the Learned Words left it.
        case rejected(raw: String, reason: String)
        /// Transcription (or the commit) failed; the error is surfaced to map.
        case failed(any Error)
        /// The work was cancelled while still the current operation.
        case cancelled
        /// A cancel-and-restart superseded this operation after it began — the
        /// newer operation owns the caller's state, so this result commits nothing.
        case superseded
    }

    /// Where the take is. A stop that returns audio leaves it transcribing
    /// until ``transcribeAndCommit(_:onTake:commit:)`` resolves.
    private(set) var phase: Phase = .idle

    /// Called on every phase change, as it happens: a surface narrates the
    /// take in its own presentation (dictation's Overlay Feed shows the
    /// Proofread Pass).
    @ObservationIgnored var onPhaseChange: (@MainActor (Phase) -> Void)?

    /// The app in front when the current take started.
    private(set) var targetApp: TargetApp?

    /// A take the Proofread Pass rejected, waiting for "insert anyway".
    var hasRejectedTake: Bool { rejected != nil }

    @ObservationIgnored private let stages: Stages

    /// The **Operation Guard** for this session. `invalidate()`d at *operation
    /// start* and at `cancel()` so a transcription that finishes (or races
    /// cancellation to *success*) after a cancel-and-restart recognizes it is stale
    /// via its `OperationTicket` and commits nothing.
    @ObservationIgnored private let operations = OperationGuard()

    /// The in-flight transcribe→commit `Task`. Held so `cancel()` can cancel it: the
    /// commit's `await` is itself the side effect and is cancellation-aware, so the
    /// post-`await` ticket check alone can't stop a commit already suspended in
    /// flight — cancelling the task does. Identity-guarded on clear so an overlapping
    /// successor is never clobbered.
    @ObservationIgnored private var inFlightTask: Task<Outcome, Never>?

    /// This session started the running capture. The microphone and the
    /// recognizer are shared, so ``cancel()`` stops only what it started.
    @ObservationIgnored private var ownsCapture = false

    /// The Live Preview pump and where it shows each preview.
    @ObservationIgnored private var previewTask: Task<Void, Never>?
    @ObservationIgnored private var showPreview: (@MainActor (LivePreview?) -> Void)?

    /// The last take the Proofread Pass rejected, with its length.
    @ObservationIgnored private var rejected: (take: Take, duration: TimeInterval)?

    fileprivate init(stages: Stages) {
        self.stages = stages
    }

    // MARK: - Interface

    /// Begins capture. Welds the Operation Guard epoch bump to capture start (so a
    /// new operation supersedes an overlapping prior one) and carries the single
    /// microphone-busy guard. With `preview`, the session decodes the take while
    /// it is recorded and shows each **Live Preview** there, read the way the
    /// take will read; ``stop()`` and ``cancel()`` end it with `nil`.
    func start(preview: (@MainActor (LivePreview?) -> Void)? = nil) -> StartResult {
        guard !stages.microphone.isCapturing else { return .micBusy }
        operations.invalidate()
        do {
            try stages.microphone.startCapture()
            ownsCapture = true
            targetApp = stages.frontmostApp()
            setPhase(.listening)
            if let preview { startPreview(preview) }
            return .started
        } catch {
            return .captureFailed(error)
        }
    }

    /// Stops capture and applies the minimum-duration and silence guards. Does
    /// not advance the epoch — the caller's maximum-duration auto-stop drives
    /// *when* this is called.
    func stop() -> StopResult {
        stopPreview()
        ownsCapture = false
        let result = stopCapture()
        if case .audio = result { setPhase(.transcribing) } else { setPhase(.idle) }
        return result
    }

    /// Transcribes `audio`, reads it through the take's text pipeline, runs the
    /// **Proofread Pass** as the `.proofreading` phase, and — while the
    /// operation is still current — runs the caller-injected `commit` inside the
    /// guarded region, so the post-commit staleness re-check (which suppresses a
    /// success that races a cancel mid-commit) is preserved without exposing the
    /// ticket. Counts the take's **Catches** once it is in. Owns the in-flight
    /// `Task`; callers spawn a fire-and-forget outer `Task` purely to avoid
    /// blocking and do not track it.
    ///
    /// A skipped pass (disabled, model unavailable, the LLM busy, an error)
    /// commits the cleaned text, so the pass can only improve, never gate.
    func transcribeAndCommit(
        _ audio: AudioData,
        onTake: (@MainActor (Take) -> Void)? = nil,
        commit: @escaping @MainActor (Commit) async throws -> Void
    ) async -> Outcome {
        let ticket = operations.capture()
        let app = targetApp
        setPhase(.transcribing)
        let task = Task { () -> Outcome in
            do {
                let result = try await stages.recognizer.transcribe(
                    audio, language: stages.language())

                // A cancel-and-restart since this operation began means a newer
                // operation owns the caller's state — drop this result.
                guard ticket.isCurrent else { return .superseded }

                let reading = stages.read(result.text, in: app)
                guard !reading.cleaned.isEmpty else { return .empty }
                // The owner's words, as the owner's fixes taught them.
                let learned = reading.learned

                setPhase(.proofreading)
                let verdict = await stages.proofread(learned.text)
                // The pass awaited — a cancel-and-restart during it means a
                // newer operation owns the state.
                guard ticket.isCurrent else { return .superseded }
                setPhase(.transcribing)

                var take = Take(
                    rawASR: result.text, cleaned: reading.cleaned, learned: learned.text,
                    catches: learned.catches, verdict: verdict, committedText: learned.text,
                    app: app)
                var edits: [WordEdit] = []
                switch verdict {
                case .corrected(let text, let correctedEdits):
                    take = take.committing(text)
                    edits = correctedEdits
                case .rejected(let reason):
                    let rejectedTake = take.committing(nil)
                    onTake?(rejectedTake)
                    rejected = (rejectedTake, audio.duration)
                    return .rejected(raw: learned.text, reason: reason)
                case .unchanged, nil:
                    break
                }

                // Before the commit, so the caller can link what it stores in
                // the commit (a history entry) to the take it just observed.
                onTake?(take)
                let committed = take.commit(duration: audio.duration)
                try await commit(committed)

                // The commit may suspend (e.g. text injection); a cancel-and-restart
                // during it means the newer operation owns the state — suppress the
                // success the caller would otherwise present.
                guard ticket.isCurrent else { return .superseded }
                rejected = nil
                // Counted once the take is in, and only the catches that
                // landed (the Proofread Pass may have rewritten one): a
                // rejected, failed or superseded take caught nothing.
                stages.learnedWords.recordCatches(committed.catches)
                return .committed(edits: edits)
            } catch is CancellationError {
                return ticket.isCurrent ? .cancelled : .superseded
            } catch {
                return ticket.isCurrent ? .failed(error) : .superseded
            }
        }
        inFlightTask = task
        let outcome = await task.value
        // Identity-guarded: clear only if the slot still holds *this* task, so an
        // overlapping successor started after a cancel isn't freed by this exit.
        if inFlightTask == task { inFlightTask = nil }
        if ticket.isCurrent { setPhase(.idle) }
        return outcome
    }

    /// "Insert anyway" on a take the Proofread Pass rejected: commits its text
    /// as the Learned Words left it through `commit`, exactly like a take that
    /// landed, and counts its catches once it is in. Nil when no rejected take
    /// waits; the take is spent either way, so a second press inserts nothing.
    func commitRejectedTake(
        _ commit: @MainActor (Commit) async throws -> Void
    ) async -> Outcome? {
        guard let waiting = rejected else { return nil }
        rejected = nil
        let committed = waiting.take.committing(waiting.take.learned)
            .commit(duration: waiting.duration)
        do {
            try await commit(committed)
        } catch {
            return .failed(error)
        }
        stages.learnedWords.recordCatches(committed.catches)
        return .committed(edits: [])
    }

    /// Cancels the operation: advances the epoch (so a late success commits
    /// nothing), cancels the in-flight task (aborting a commit suspended mid-flight),
    /// stops capture if running, and tells the engine to cancel. The caller resets
    /// its own presentation state.
    ///
    /// Only this session's own work: the microphone and the recognizer are
    /// shared, and a panel closing must not cut off another surface's take
    /// (dictation's, or the composer's).
    func cancel() {
        operations.invalidate()
        stopPreview()
        let transcribing = inFlightTask != nil
        inFlightTask?.cancel()
        inFlightTask = nil
        if ownsCapture, stages.microphone.isCapturing {
            _ = stages.microphone.stopCapture()
        }
        ownsCapture = false
        if transcribing { stages.recognizer.cancelTranscription() }
        setPhase(.idle)
    }

    // MARK: - Private

    private func setPhase(_ newPhase: Phase) {
        guard phase != newPhase else { return }
        phase = newPhase
        onPhaseChange?(newPhase)
    }

    private func stopCapture() -> StopResult {
        guard var audioData = stages.microphone.stopCapture() else { return .noAudio }
        guard audioData.duration >= Self.minimumRecordingDuration else { return .tooShort }
        // Not evidence either: a silent capture is neither transcribed nor
        // dumped. Nor is an accidental tap or an abandoned (cancelled) take.
        guard !CaptureLevel.isSilent(audioData) else { return .silent }
        var dumpFile: String?
        if let raw = audioData.raw {
            if stages.isCaptureDumpEnabled() {
                dumpFile = stages.captureDump.save(raw)
            }
            // The dump is `raw`'s only consumer. (Bookkeeping more than memory
            // now: `samples` shares the same native-rate storage since the
            // recognizer took over the 16 kHz conversion.)
            audioData.raw = nil
        }
        return .audio(audioData, dumpFile: dumpFile)
    }

    // MARK: - Live Preview (ADR-0086)

    /// Decodes the take while it is recorded, from the end of its last
    /// confirmed segment, and shows what Whisper heard read through the take's
    /// text pipeline (its catches shown, not counted). A decode that resolves
    /// after its take ended is dropped, never shown on the next take; the
    /// paste is always the full pass.
    private func startPreview(_ show: @escaping @MainActor (LivePreview?) -> Void) {
        let ticket = operations.capture()
        let app = targetApp
        showPreview = show
        previewTask = Task { [weak self] in
            var assembler = LivePreviewAssembler()
            while !Task.isCancelled {
                guard let self, ticket.isCurrent, self.phase == .listening else { return }
                if let tail = self.stages.microphone.captureSnapshot(from: assembler.confirmedEnd),
                    let window = assembler.window(ofTail: tail)
                {
                    let result = await self.stages.recognizer.transcribePartial(
                        window, language: self.stages.language())
                    guard !Task.isCancelled, ticket.isCurrent, self.phase == .listening
                    else { return }
                    if let result {
                        assembler.fold(result, windowDuration: window.duration)
                        show(self.preview(of: assembler, in: app))
                    }
                }
                try? await Task.sleep(for: Self.previewInterval)
            }
        }
    }

    private func stopPreview() {
        previewTask?.cancel()
        previewTask = nil
        guard let show = showPreview else { return }
        showPreview = nil
        show(nil)
    }

    /// The preview as the take will read. Its confirmed words are its leading
    /// words that match the confirmed text on its own (read the same way):
    /// where the two readings differ at the boundary, the word is provisional.
    private func preview(of assembler: LivePreviewAssembler, in app: TargetApp?) -> LivePreview? {
        let reading = stages.read(assembler.rawText, in: app)
        guard !reading.cleaned.isEmpty else { return nil }
        let learned = reading.learned
        let confirmed = stages.read(assembler.confirmed, in: app).learned.text
        let confirmedWords = zip(TakeText.tokens(confirmed), TakeText.tokens(learned.text))
            .prefix { $0.bare == $1.bare }.count
        return LivePreview(
            text: learned.text, catches: learned.catches, confirmedTokens: confirmedWords)
    }
}

extension VoiceCaptureSession.Take {
    /// The take committing `text` (nil: rejected).
    fileprivate func committing(_ text: String?) -> Self {
        Self(
            rawASR: rawASR, cleaned: cleaned, learned: learned, catches: catches,
            verdict: verdict, committedText: text, app: app)
    }

    /// What the surface's commit receives: the committed text with the
    /// catches where they stand in it (the Proofread Pass may have moved
    /// them).
    fileprivate func commit(duration: TimeInterval) -> VoiceCaptureSession.Commit {
        let text = committedText ?? learned
        return VoiceCaptureSession.Commit(
            text: text, duration: duration,
            catches: text == learned ? catches : LearnedWordMatcher.relocate(catches, in: text),
            app: app)
    }
}
