//
//  VoiceCaptureSessionTests.swift
//  tesseractTests
//
//  Drives the **Voice Capture Session** through its own interface
//  (`start`/`stop`/`transcribeAndCommit`/`cancel`) — the seam where the shared
//  capture→transcribe→commit lifecycle now lives. Uses the fakes that already sit
//  *below* the session (`FakeAudioCapture` for `AudioCapturing`,
//  `ControllableTranscribing` for `Transcribing`) plus a recorder commit closure
//  standing in for whatever a caller injects. The deep staleness/supersede races —
//  late-success-after-cancel, cancel-during-commit suppression — are asserted here
//  once, instead of being duplicated across the two coordinator suites.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct VoiceCaptureSessionTests {

    // MARK: - A commit closure that records and can suspend

    /// Stands in for a caller's injected `commit`. When `gated`, `commit` suspends
    /// until `releaseGate()` — so a test can hold the success path *inside* the
    /// commit and race a `cancel()` against it (the injection-suspension window).
    /// Mirrors the real commit's cancellation-awareness: a cancelled in-flight task
    /// aborts the side effect rather than recording it.
    @MainActor
    final class CommitRecorder {
        private(set) var commits: [(text: String, duration: TimeInterval)] = []
        var gated = false
        private var gate: CheckedContinuation<Void, Never>?
        var isAwaitingGate: Bool { gate != nil }

        func commit(_ text: String, _ duration: TimeInterval) async throws {
            if gated {
                await withCheckedContinuation { gate = $0 }
            }
            try Task.checkCancellation()
            commits.append((text, duration))
        }

        func releaseGate() {
            gate?.resume()
            gate = nil
        }
    }

    private func makeAudio(duration: TimeInterval = 2.0) -> AudioData {
        AudioData(samples: [0.1, 0.2], sampleRate: 16_000, duration: duration)
    }

    // MARK: - start()

    @Test func startBeginsCaptureAndReportsStarted() {
        let capture = FakeAudioCapture(cannedAudio: makeAudio())
        let session = VoiceCaptureSession(
            audioCapture: capture, transcriptionEngine: ControllableTranscribing())

        let result = session.start()

        guard case .started = result else {
            Issue.record("expected .started, got \(result)")
            return
        }
        #expect(capture.isCapturing)
        #expect(capture.startCount == 1)
    }

    @Test func startWhileMicrophoneBusyReportsMicBusyWithoutStarting() {
        let capture = FakeAudioCapture(cannedAudio: makeAudio())
        capture.isCapturing = true  // shared engine already capturing
        let session = VoiceCaptureSession(
            audioCapture: capture, transcriptionEngine: ControllableTranscribing())

        let result = session.start()

        guard case .micBusy = result else {
            Issue.record("expected .micBusy, got \(result)")
            return
        }
        #expect(capture.startCount == 0)
    }

    @Test func startSurfacesCaptureFailure() {
        struct Boom: Error {}
        let capture = FakeAudioCapture(cannedAudio: makeAudio())
        capture.startError = Boom()
        let session = VoiceCaptureSession(
            audioCapture: capture, transcriptionEngine: ControllableTranscribing())

        let result = session.start()

        guard case .captureFailed(let error) = result else {
            Issue.record("expected .captureFailed, got \(result)")
            return
        }
        #expect(error is Boom)
    }

    // MARK: - stop()

    @Test func stopReturnsCapturedAudioAtOrAboveMinimum() {
        let capture = FakeAudioCapture(cannedAudio: makeAudio(duration: 2.0))
        let session = VoiceCaptureSession(
            audioCapture: capture, transcriptionEngine: ControllableTranscribing())
        _ = session.start()

        let result = session.stop()

        guard case .audio(let audio) = result else {
            Issue.record("expected .audio, got \(result)")
            return
        }
        #expect(audio.duration == 2.0)
    }

    @Test func stopRejectsRecordingShorterThanMinimum() {
        let capture = FakeAudioCapture(cannedAudio: makeAudio(duration: 0.2))
        let session = VoiceCaptureSession(
            audioCapture: capture, transcriptionEngine: ControllableTranscribing())
        _ = session.start()

        let result = session.stop()

        guard case .tooShort = result else {
            Issue.record("expected .tooShort, got \(result)")
            return
        }
    }

    @Test func stopReportsNoAudioWhenEngineReturnsNothing() {
        let capture = FakeAudioCapture(cannedAudio: nil)
        let session = VoiceCaptureSession(
            audioCapture: capture, transcriptionEngine: ControllableTranscribing())
        _ = session.start()

        let result = session.stop()

        guard case .noAudio = result else {
            Issue.record("expected .noAudio, got \(result)")
            return
        }
    }

    // MARK: - transcribeAndCommit()

    @Test func transcribeAndCommitDeliversProcessedTextThenReportsCommitted() async throws {
        let engine = ControllableTranscribing(
            result: TranscriptionResult(
                text: "hello world", segments: [], language: "en", processingTime: 0))
        let session = VoiceCaptureSession(
            audioCapture: FakeAudioCapture(cannedAudio: makeAudio()), transcriptionEngine: engine)
        let recorder = CommitRecorder()

        let audio = makeAudio()
        let task = Task {
            await session.transcribeAndCommit(audio, language: "en") { text, duration in
                try await recorder.commit(text, duration)
            }
        }
        while !engine.isAwaiting { await Task.yield() }
        engine.completeWithSuccess()
        let outcome = await task.value

        guard case .committed = outcome else {
            Issue.record("expected .committed, got \(outcome)")
            return
        }
        let expected = TranscriptionPostProcessor().process("hello world")
        #expect(recorder.commits.count == 1)
        #expect(recorder.commits.first?.text == expected)
        #expect(recorder.commits.first?.duration == audio.duration)
    }

    @Test func transcribeAndCommitReportsEmptyWhenPostProcessingYieldsNothing() async throws {
        let engine = ControllableTranscribing(
            result: TranscriptionResult(
                text: "   ", segments: [], language: "en", processingTime: 0))
        let session = VoiceCaptureSession(
            audioCapture: FakeAudioCapture(cannedAudio: makeAudio()), transcriptionEngine: engine)
        let recorder = CommitRecorder()

        let task = Task {
            await session.transcribeAndCommit(makeAudio(), language: "en") { text, duration in
                try await recorder.commit(text, duration)
            }
        }
        while !engine.isAwaiting { await Task.yield() }
        engine.completeWithSuccess()
        let outcome = await task.value

        guard case .empty = outcome else {
            Issue.record("expected .empty, got \(outcome)")
            return
        }
        #expect(recorder.commits.isEmpty)
    }

    @Test func transcribeAndCommitReportsFailureOnTranscriptionError() async throws {
        struct Boom: Error {}
        let engine = ControllableTranscribing()
        let session = VoiceCaptureSession(
            audioCapture: FakeAudioCapture(cannedAudio: makeAudio()), transcriptionEngine: engine)
        let recorder = CommitRecorder()

        let task = Task {
            await session.transcribeAndCommit(makeAudio(), language: "en") { text, duration in
                try await recorder.commit(text, duration)
            }
        }
        while !engine.isAwaiting { await Task.yield() }
        engine.completeWithFailure(Boom())
        let outcome = await task.value

        guard case .failed(let error) = outcome else {
            Issue.record("expected .failed, got \(outcome)")
            return
        }
        #expect(error is Boom)
        #expect(recorder.commits.isEmpty)
    }

    /// A `CancellationError` while this is still the current operation returns
    /// `.cancelled` (not `.superseded`) and commits nothing.
    @Test func transcribeAndCommitReportsCancelledOnCancellationWhileCurrent() async throws {
        let engine = ControllableTranscribing()
        let session = VoiceCaptureSession(
            audioCapture: FakeAudioCapture(cannedAudio: makeAudio()), transcriptionEngine: engine)
        let recorder = CommitRecorder()

        let task = Task {
            await session.transcribeAndCommit(makeAudio(), language: "en") { text, duration in
                try await recorder.commit(text, duration)
            }
        }
        while !engine.isAwaiting { await Task.yield() }
        engine.completeWithCancellation()
        let outcome = await task.value

        guard case .cancelled = outcome else {
            Issue.record("expected .cancelled, got \(outcome)")
            return
        }
        #expect(recorder.commits.isEmpty)
    }

    /// The load-bearing staleness case: a transcription that completes
    /// *successfully* after `cancel()` is recognized as stale and commits nothing —
    /// `.superseded`. This is the "recognizer ignores cancellation and returns
    /// success anyway" scenario the Operation Guard exists for.
    @Test func lateSuccessAfterCancelIsSupersededAndCommitsNothing() async throws {
        let engine = ControllableTranscribing(
            result: TranscriptionResult(
                text: "late success", segments: [], language: "en", processingTime: 0))
        let session = VoiceCaptureSession(
            audioCapture: FakeAudioCapture(cannedAudio: makeAudio()), transcriptionEngine: engine)
        let recorder = CommitRecorder()

        let task = Task {
            await session.transcribeAndCommit(makeAudio(), language: "en") { text, duration in
                try await recorder.commit(text, duration)
            }
        }
        while !engine.isAwaiting { await Task.yield() }

        session.cancel()
        #expect(engine.cancelCount == 1)

        engine.completeWithSuccess()
        let outcome = await task.value

        guard case .superseded = outcome else {
            Issue.record("expected .superseded, got \(outcome)")
            return
        }
        #expect(recorder.commits.isEmpty)
    }

    /// A cancel that races an in-flight *commit* (the injection-suspension window)
    /// must suppress the success: the commit's side effect is aborted and the
    /// outcome is `.superseded`.
    @Test func cancelDuringCommitSuppressesTheSideEffect() async throws {
        let engine = ControllableTranscribing(
            result: TranscriptionResult(
                text: "done", segments: [], language: "en", processingTime: 0))
        let session = VoiceCaptureSession(
            audioCapture: FakeAudioCapture(cannedAudio: makeAudio()), transcriptionEngine: engine)
        let recorder = CommitRecorder()
        recorder.gated = true

        let task = Task {
            await session.transcribeAndCommit(makeAudio(), language: "en") { text, duration in
                try await recorder.commit(text, duration)
            }
        }
        while !engine.isAwaiting { await Task.yield() }
        engine.completeWithSuccess()

        // The commit is now suspended inside the gate; cancel races it.
        while !recorder.isAwaitingGate { await Task.yield() }
        session.cancel()
        recorder.releaseGate()
        let outcome = await task.value

        guard case .superseded = outcome else {
            Issue.record("expected .superseded, got \(outcome)")
            return
        }
        #expect(recorder.commits.isEmpty)
    }
}
