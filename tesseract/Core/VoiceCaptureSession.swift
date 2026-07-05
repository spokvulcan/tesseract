//
//  VoiceCaptureSession.swift
//  tesseract
//
//  The **Voice Capture Session** — the single concrete home for the push-to-talk
//  *capture → transcribe → commit* lifecycle that `DictationCoordinator` (the
//  global dictation overlay) and **Voice Input** (`AgentVoiceInputController`, the
//  agent composer leaf) both drive. It owns the subtle, bug-prone parts — the
//  **Operation Guard** ticket discipline re-checked after every `await`, the
//  microphone-busy guard, the minimum-duration and empty-text guards,
//  post-processing, the in-flight transcription `Task`, and cancellation — behind a
//  small value-returning interface. Each caller keeps only what genuinely differs:
//  its own state vocabulary, error model, sounds, maximum-recording-duration
//  auto-stop, error auto-reset, and the commit it injects.
//
//  Composed *directly* (not behind a port), exactly as both callers compose
//  `OperationGuard`: there is one production implementation, so there is no seam to
//  vary across. The interface *is* the test surface — `VoiceCaptureSessionTests`
//  drives this type through `start`/`stop`/`transcribeAndCommit`/`cancel` using the
//  fakes that already sit *below* it.
//

import Foundation

@MainActor
final class VoiceCaptureSession {
    /// The shared error-lingering delay. The auto-reset itself stays caller-owned
    /// (each caller's error lives in a different surface — overlay vs `voiceState`),
    /// but the *duration* lives here so the two callers cannot drift on it.
    static let errorAutoResetDelay: Duration = .seconds(3)

    /// A capture below this is rejected by ``stop()`` before any transcription runs
    /// — an accidental tap must not show a spurious "processing" flash.
    private static let minimumRecordingDuration: TimeInterval = 0.5

    /// The outcome of ``start()``.
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
        case audio(AudioData)
        /// The capture was shorter than the minimum duration.
        case tooShort
        /// The capture engine returned no audio at all.
        case noAudio
    }

    /// The outcome of ``transcribeAndCommit(_:language:commit:)``.
    enum Outcome {
        /// Transcription succeeded, produced non-empty text, and the injected
        /// `commit` ran to completion while this operation was still current.
        case committed
        /// Post-processing yielded empty text — no speech to commit.
        case empty
        /// Transcription (or the commit) failed; the error is surfaced to map.
        case failed(any Error)
        /// The work was cancelled while still the current operation.
        case cancelled
        /// A cancel-and-restart superseded this operation after it began — the
        /// newer operation owns the caller's state, so this result commits nothing.
        case superseded
    }

    private let audioCapture: any AudioCapturing
    private let transcriptionEngine: any Transcribing
    private let postProcessor = TranscriptionPostProcessor()

    /// The **Capture Dump** and its enablement, both caller-injected. The dump
    /// only ever sees captures that passed the minimum-duration guard — an
    /// accidental tap or an abandoned (cancelled) recording is not evidence.
    private let captureDump: (any CaptureDumpStoring)?
    private let isCaptureDumpEnabled: () -> Bool

    /// The **Operation Guard** for this session. `invalidate()`d at *operation
    /// start* and at `cancel()` so a transcription that finishes (or races
    /// cancellation to *success*) after a cancel-and-restart recognizes it is stale
    /// via its `OperationTicket` and commits nothing. See CONTEXT.md →
    /// "Operation staleness".
    private let operations = OperationGuard()

    /// The in-flight transcribe→commit `Task`. Held so `cancel()` can cancel it: the
    /// commit's `await` is itself the side effect and is cancellation-aware, so the
    /// post-`await` ticket check alone can't stop a commit already suspended in
    /// flight — cancelling the task does. Identity-guarded on clear so an overlapping
    /// successor is never clobbered.
    private var inFlightTask: Task<Outcome, Never>?

    init(
        audioCapture: any AudioCapturing,
        transcriptionEngine: any Transcribing,
        captureDump: (any CaptureDumpStoring)? = nil,
        isCaptureDumpEnabled: @escaping () -> Bool = { true }
    ) {
        self.audioCapture = audioCapture
        self.transcriptionEngine = transcriptionEngine
        self.captureDump = captureDump
        self.isCaptureDumpEnabled = isCaptureDumpEnabled
    }

    // MARK: - Interface

    /// Begins capture. Welds the Operation Guard epoch bump to capture start (so a
    /// new operation supersedes an overlapping prior one) and carries the single
    /// microphone-busy guard — both callers gain it here, once.
    func start() -> StartResult {
        guard !audioCapture.isCapturing else { return .micBusy }
        operations.invalidate()
        do {
            try audioCapture.startCapture()
            return .started
        } catch {
            return .captureFailed(error)
        }
    }

    /// Stops capture and applies the minimum-duration guard. Does not advance the
    /// epoch — the caller's maximum-duration auto-stop drives *when* this is called.
    func stop() -> StopResult {
        guard let audioData = audioCapture.stopCapture() else { return .noAudio }
        guard audioData.duration >= Self.minimumRecordingDuration else { return .tooShort }
        if let raw = audioData.raw, let captureDump, isCaptureDumpEnabled() {
            captureDump.save(raw)
        }
        return .audio(audioData)
    }

    /// Transcribes `audio`, post-processes it, and — while the operation is still
    /// current — runs the caller-injected `commit` inside the guarded region, so the
    /// post-commit staleness re-check (which suppresses a success that races a
    /// cancel mid-commit) is preserved without exposing the ticket. Owns the
    /// in-flight `Task`; callers spawn a fire-and-forget outer `Task` purely to avoid
    /// blocking and do not track it.
    func transcribeAndCommit(
        _ audio: AudioData,
        language: String,
        commit: @escaping @MainActor (String, TimeInterval) async throws -> Void
    ) async -> Outcome {
        let ticket = operations.capture()
        let task = Task { () -> Outcome in
            do {
                let result = try await transcriptionEngine.transcribe(audio, language: language)

                // A cancel-and-restart since this operation began means a newer
                // operation owns the caller's state — drop this result.
                guard ticket.isCurrent else { return .superseded }

                let processedText = postProcessor.process(result.text)
                guard !processedText.isEmpty else { return .empty }

                try await commit(processedText, audio.duration)

                // The commit may suspend (e.g. text injection); a cancel-and-restart
                // during it means the newer operation owns the state — suppress the
                // success the caller would otherwise present.
                guard ticket.isCurrent else { return .superseded }
                return .committed
            } catch is CancellationError {
                return ticket.isCurrent ? .cancelled : .superseded
            } catch {
                return ticket.isCurrent ? .failed(error) : .superseded
            }
        }
        inFlightTask = task
        defer {
            // Identity-guarded: clear only if the slot still holds *this* task, so an
            // overlapping successor started after a cancel isn't freed by this exit.
            if inFlightTask == task { inFlightTask = nil }
        }
        return await task.value
    }

    /// Cancels the operation: advances the epoch (so a late success commits
    /// nothing), cancels the in-flight task (aborting a commit suspended mid-flight),
    /// stops capture if running, and tells the engine to cancel. The caller resets
    /// its own presentation state.
    func cancel() {
        operations.invalidate()
        inFlightTask?.cancel()
        inFlightTask = nil
        if audioCapture.isCapturing {
            _ = audioCapture.stopCapture()
        }
        transcriptionEngine.cancelTranscription()
    }
}
