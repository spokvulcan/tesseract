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

    /// Records what the session hands to the **Capture Dump** seam.
    @MainActor
    final class FakeCaptureDump: CaptureDumpStoring {
        private(set) var saved: [RawCapture] = []
        private(set) var deleteAllCount = 0

        @discardableResult
        func save(_ capture: RawCapture) -> String? {
            saved.append(capture)
            return "capture-\(saved.count).wav"
        }
        func deleteAll() { deleteAllCount += 1 }
    }

    private func makeAudio(
        duration: TimeInterval = 2.0, raw: RawCapture? = nil
    ) -> AudioData {
        AudioData(samples: [0.1, 0.2], sampleRate: 16_000, duration: duration, raw: raw)
    }

    private func makeRaw(voiceProcessed: Bool = false) -> RawCapture {
        RawCapture(
            samples: [0.3, 0.4, 0.5], sampleRate: 48_000, voiceProcessed: voiceProcessed)
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

        guard case .audio(let audio, _) = result else {
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

    // MARK: - Capture Dump

    /// A successful stop hands the raw capture — conditions intact — to the
    /// dump, and surfaces the dump's file name on the result (the Correction
    /// Pair's audio reference, ticket #289).
    @Test func stopSavesTheRawCaptureToTheDump() {
        let raw = makeRaw(voiceProcessed: true)
        let capture = FakeAudioCapture(cannedAudio: makeAudio(duration: 2.0, raw: raw))
        let dump = FakeCaptureDump()
        let session = VoiceCaptureSession(
            audioCapture: capture,
            transcriptionEngine: ControllableTranscribing(),
            captureDump: dump,
            isCaptureDumpEnabled: { true }
        )
        _ = session.start()

        let result = session.stop()

        #expect(dump.saved.count == 1)
        #expect(dump.saved.first?.samples == raw.samples)
        #expect(dump.saved.first?.sampleRate == raw.sampleRate)
        #expect(dump.saved.first?.voiceProcessed == true)
        guard case .audio(_, let dumpFile) = result else {
            Issue.record("expected .audio, got \(result)")
            return
        }
        #expect(dumpFile == "capture-1.wav")
    }

    @Test func stopDoesNotDumpARecordingShorterThanMinimum() {
        let capture = FakeAudioCapture(
            cannedAudio: makeAudio(duration: 0.2, raw: makeRaw()))
        let dump = FakeCaptureDump()
        let session = VoiceCaptureSession(
            audioCapture: capture,
            transcriptionEngine: ControllableTranscribing(),
            captureDump: dump,
            isCaptureDumpEnabled: { true }
        )
        _ = session.start()

        _ = session.stop()

        #expect(dump.saved.isEmpty)
    }

    /// Dump disabled means the store is never invoked — not invoked-and-discarded.
    @Test func stopDoesNotDumpWhenTheSettingIsDisabled() {
        let capture = FakeAudioCapture(
            cannedAudio: makeAudio(duration: 2.0, raw: makeRaw()))
        let dump = FakeCaptureDump()
        let session = VoiceCaptureSession(
            audioCapture: capture,
            transcriptionEngine: ControllableTranscribing(),
            captureDump: dump,
            isCaptureDumpEnabled: { false }
        )
        _ = session.start()

        _ = session.stop()

        #expect(dump.saved.isEmpty)
    }

    @Test func stopDoesNotDumpWhenTheCaptureCarriesNoRawAudio() {
        let capture = FakeAudioCapture(cannedAudio: makeAudio(duration: 2.0, raw: nil))
        let dump = FakeCaptureDump()
        let session = VoiceCaptureSession(
            audioCapture: capture,
            transcriptionEngine: ControllableTranscribing(),
            captureDump: dump,
            isCaptureDumpEnabled: { true }
        )
        _ = session.start()

        _ = session.stop()

        #expect(dump.saved.isEmpty)
    }

    /// `cancel()` also stops a running capture — but an abandoned recording is
    /// not evidence; nothing may reach the dump.
    @Test func cancelDoesNotDumpTheAbandonedCapture() {
        let capture = FakeAudioCapture(
            cannedAudio: makeAudio(duration: 2.0, raw: makeRaw()))
        let dump = FakeCaptureDump()
        let session = VoiceCaptureSession(
            audioCapture: capture,
            transcriptionEngine: ControllableTranscribing(),
            captureDump: dump,
            isCaptureDumpEnabled: { true }
        )
        _ = session.start()

        session.cancel()

        #expect(dump.saved.isEmpty)
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

    // MARK: - The Proofread Pass arm

    /// A gate-able stand-in for the caller's proofread closure, so a test can
    /// hold the operation *inside* the pass and race a `cancel()` against it.
    @MainActor
    final class ProofreadStub {
        var verdict: ProofreadVerdict?
        var gated = false
        private var gate: CheckedContinuation<Void, Never>?
        var isAwaitingGate: Bool { gate != nil }
        private(set) var seenTexts: [String] = []

        func proofread(_ text: String) async -> ProofreadVerdict? {
            seenTexts.append(text)
            if gated {
                await withCheckedContinuation { gate = $0 }
            }
            return verdict
        }

        func releaseGate() {
            gate?.resume()
            gate = nil
        }
    }

    private func runThroughSession(
        engineText: String,
        stub: ProofreadStub,
        recorder: CommitRecorder
    ) async -> (outcome: VoiceCaptureSession.Outcome, session: VoiceCaptureSession) {
        let engine = ControllableTranscribing(
            result: TranscriptionResult(
                text: engineText, segments: [], language: "en", processingTime: 0))
        let session = VoiceCaptureSession(
            audioCapture: FakeAudioCapture(cannedAudio: makeAudio()), transcriptionEngine: engine)
        let task = Task {
            await session.transcribeAndCommit(
                makeAudio(), language: "en",
                proofread: { await stub.proofread($0) }
            ) { text, duration in
                try await recorder.commit(text, duration)
            }
        }
        while !engine.isAwaiting { await Task.yield() }
        engine.completeWithSuccess()
        return (await task.value, session)
    }

    /// `.corrected` commits the *corrected* text and surfaces the edits.
    @Test func proofreadCorrectedCommitsTheCorrectedTextWithEdits() async throws {
        let stub = ProofreadStub()
        let edit = WordEdit(original: "peace", replacement: "piece")
        stub.verdict = .corrected(text: "piece of cake", edits: [edit])
        let recorder = CommitRecorder()

        let (outcome, _) = await runThroughSession(
            engineText: "peace of cake", stub: stub, recorder: recorder)

        guard case .committed(let edits) = outcome else {
            Issue.record("expected .committed, got \(outcome)")
            return
        }
        #expect(edits == [edit])
        #expect(recorder.commits.count == 1)
        #expect(recorder.commits.first?.text == "piece of cake")
        // The pass saw the *post-processed* transcription, not the engine raw.
        #expect(stub.seenTexts == [TranscriptionPostProcessor().process("peace of cake")])
    }

    /// `.rejected` commits nothing; the raw text rides the outcome for
    /// "insert raw anyway".
    @Test func proofreadRejectedCommitsNothingAndCarriesTheRaw() async throws {
        let stub = ProofreadStub()
        stub.verdict = .rejected(reason: "unintelligible mumbling")
        let recorder = CommitRecorder()

        let (outcome, _) = await runThroughSession(
            engineText: "asdf ghjk", stub: stub, recorder: recorder)

        guard case .rejected(let raw, let reason) = outcome else {
            Issue.record("expected .rejected, got \(outcome)")
            return
        }
        #expect(raw == TranscriptionPostProcessor().process("asdf ghjk"))
        #expect(reason == "unintelligible mumbling")
        #expect(recorder.commits.isEmpty)
    }

    /// A skipped pass (`nil`) is fail-open: the raw text commits, no edits.
    @Test func proofreadSkipCommitsTheRawText() async throws {
        let stub = ProofreadStub()
        stub.verdict = nil
        let recorder = CommitRecorder()

        let (outcome, _) = await runThroughSession(
            engineText: "hello world", stub: stub, recorder: recorder)

        guard case .committed(let edits) = outcome else {
            Issue.record("expected .committed, got \(outcome)")
            return
        }
        #expect(edits.isEmpty)
        #expect(recorder.commits.first?.text == TranscriptionPostProcessor().process("hello world"))
    }

    // MARK: - The take observer (Correction Pair source, ticket #289)

    /// `onTake` delivers the take's full lineage — raw ASR before any
    /// cleanup, the cleaned text, verdict, committed text — and fires
    /// *before* the commit, so the caller can link what the commit stores.
    @Test func onTakeDeliversTheLineageBeforeTheCommit() async throws {
        let stub = ProofreadStub()
        stub.verdict = .corrected(text: "piece of cake", edits: [])
        let recorder = CommitRecorder()
        let engine = ControllableTranscribing(
            result: TranscriptionResult(
                text: "  peace of cake  ", segments: [], language: "en", processingTime: 0))
        let session = VoiceCaptureSession(
            audioCapture: FakeAudioCapture(cannedAudio: makeAudio()), transcriptionEngine: engine)

        var takes: [VoiceCaptureSession.Take] = []
        var commitCountAtTake = -1
        let task = Task {
            await session.transcribeAndCommit(
                makeAudio(), language: "en",
                proofread: { await stub.proofread($0) },
                onTake: { take in
                    takes.append(take)
                    commitCountAtTake = recorder.commits.count
                }
            ) { text, duration in
                try await recorder.commit(text, duration)
            }
        }
        while !engine.isAwaiting { await Task.yield() }
        engine.completeWithSuccess()
        _ = await task.value

        #expect(takes.count == 1)
        #expect(takes.first?.rawASR == "  peace of cake  ")
        #expect(takes.first?.cleaned == TranscriptionPostProcessor().process("  peace of cake  "))
        #expect(takes.first?.committedText == "piece of cake")
        #expect(commitCountAtTake == 0)  // observed before the commit ran
        if case .corrected = takes.first?.verdict {
        } else {
            Issue.record("expected a corrected verdict on the take")
        }
    }

    /// A rejected take is still observed — with no committed text — so the
    /// flywheel records the rejection.
    @Test func onTakeDeliversARejectedTakeWithoutCommittedText() async throws {
        let stub = ProofreadStub()
        stub.verdict = .rejected(reason: "unintelligible")
        let recorder = CommitRecorder()

        var takes: [VoiceCaptureSession.Take] = []
        let engine = ControllableTranscribing(
            result: TranscriptionResult(
                text: "asdf ghjk", segments: [], language: "en", processingTime: 0))
        let session = VoiceCaptureSession(
            audioCapture: FakeAudioCapture(cannedAudio: makeAudio()), transcriptionEngine: engine)
        let task = Task {
            await session.transcribeAndCommit(
                makeAudio(), language: "en",
                proofread: { await stub.proofread($0) },
                onTake: { takes.append($0) }
            ) { text, duration in
                try await recorder.commit(text, duration)
            }
        }
        while !engine.isAwaiting { await Task.yield() }
        engine.completeWithSuccess()
        let outcome = await task.value

        guard case .rejected = outcome else {
            Issue.record("expected .rejected, got \(outcome)")
            return
        }
        #expect(takes.count == 1)
        #expect(takes.first?.committedText == nil)
        #expect(recorder.commits.isEmpty)
    }

    /// A cancel-and-restart that lands while the operation is suspended *inside
    /// the pass* must supersede it — the ticket re-check after the proofread
    /// await is load-bearing.
    @Test func cancelDuringProofreadSupersedesAndCommitsNothing() async throws {
        let engine = ControllableTranscribing(
            result: TranscriptionResult(
                text: "hello world", segments: [], language: "en", processingTime: 0))
        let session = VoiceCaptureSession(
            audioCapture: FakeAudioCapture(cannedAudio: makeAudio()), transcriptionEngine: engine)
        let stub = ProofreadStub()
        stub.gated = true
        stub.verdict = .corrected(text: "hello walrus", edits: [])
        let recorder = CommitRecorder()

        let task = Task {
            await session.transcribeAndCommit(
                makeAudio(), language: "en",
                proofread: { await stub.proofread($0) }
            ) { text, duration in
                try await recorder.commit(text, duration)
            }
        }
        while !engine.isAwaiting { await Task.yield() }
        engine.completeWithSuccess()

        // The operation is now suspended inside the pass; cancel races it.
        while !stub.isAwaitingGate { await Task.yield() }
        session.cancel()
        stub.releaseGate()
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
