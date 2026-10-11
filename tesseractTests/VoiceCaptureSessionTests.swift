//
//  VoiceCaptureSessionTests.swift
//  tesseractTests
//
//  Drives **Voice Capture** and its **Voice Capture Session** through their
//  interface (`makeSession`, then `start`/`stop`/`transcribeAndCommit`/
//  `commitRejectedTake`/`cancel`, and the open-take facts) — the seam where
//  the shared capture→transcribe→commit lifecycle lives, for every surface.
//  Uses the peers below the module (`FakeAudioCapture` for `AudioCapturing`,
//  `ControllableTranscribing` and the recognizer peers under a real
//  `TranscriptionEngine` for `Transcribing`, `VoiceCapturePeers.swift`) plus a
//  recorder commit closure standing in for whatever a surface injects. The deep
//  staleness/supersede races — late-success-after-cancel, cancel-during-commit
//  suppression — are asserted here once.
//

import Foundation
import Observation
import Testing

@testable import Tesseract_Agent

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct VoiceCaptureSessionTests {

    // MARK: - A commit closure that records and can suspend

    /// Stands in for a surface's injected `commit`. When `gated`, `commit` suspends
    /// until `releaseGate()` — so a test can hold the success path *inside* the
    /// commit and race a `cancel()` against it (the injection-suspension window).
    /// Mirrors the real commit's cancellation-awareness: a cancelled in-flight task
    /// aborts the side effect rather than recording it.
    @MainActor @Observable
    final class CommitRecorder {
        private(set) var commits: [VoiceCaptureSession.Commit] = []
        var gated = false
        private var gate: CheckedContinuation<Void, Never>?
        var isAwaitingGate: Bool { gate != nil }

        func commit(_ take: VoiceCaptureSession.Commit) async throws {
            if gated {
                await withCheckedContinuation { gate = $0 }
            }
            try Task.checkCancellation()
            commits.append(take)
        }

        func releaseGate() {
            gate?.resume()
            gate = nil
        }
    }

    /// A gate-able stand-in for the Proofread Pass stage, so a test can hold
    /// the operation *inside* the pass and race a `cancel()` against it.
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

    private func makeAudio(
        duration: TimeInterval = 2.0, raw: RawCapture? = nil
    ) -> AudioData {
        AudioData(samples: [0.1, 0.2], sampleRate: 16_000, duration: duration, raw: raw)
    }

    private func makeRaw(voiceProcessed: Bool = false) -> RawCapture {
        RawCapture(
            samples: [0.3, 0.4, 0.5], sampleRate: 48_000, voiceProcessed: voiceProcessed)
    }

    /// A session over `engine`, whose microphone hands back `makeAudio()`.
    private func makeSession(
        engine: ControllableTranscribing, stub: ProofreadStub? = nil
    ) -> VoiceCaptureSession {
        makeVoiceCapture(
            microphone: FakeAudioCapture(cannedAudio: makeAudio()), recognizer: engine,
            proofread: { text in await stub?.proofread(text) }
        ).makeSession()
    }

    // MARK: - start()

    @Test func startBeginsCaptureAndReportsStarted() {
        let capture = FakeAudioCapture(cannedAudio: makeAudio())
        let session = makeVoiceCapture(
            microphone: capture, recognizer: ControllableTranscribing()
        ).makeSession()

        let result = session.start()

        guard case .started = result else {
            Issue.record("expected .started, got \(result)")
            return
        }
        #expect(capture.isCapturing)
        #expect(capture.startCount == 1)
        #expect(session.phase == .listening)
    }

    @Test func startWhileMicrophoneBusyReportsMicBusyWithoutStarting() {
        let capture = FakeAudioCapture(cannedAudio: makeAudio())
        capture.isCapturing = true  // shared engine already capturing
        let session = makeVoiceCapture(
            microphone: capture, recognizer: ControllableTranscribing()
        ).makeSession()

        let result = session.start()

        guard case .micBusy = result else {
            Issue.record("expected .micBusy, got \(result)")
            return
        }
        #expect(capture.startCount == 0)
        #expect(session.phase == .idle)
    }

    @Test func startSurfacesCaptureFailure() {
        struct Boom: Error {}
        let capture = FakeAudioCapture(cannedAudio: makeAudio())
        capture.startError = Boom()
        let session = makeVoiceCapture(
            microphone: capture, recognizer: ControllableTranscribing()
        ).makeSession()

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
        let session = makeVoiceCapture(
            microphone: capture, recognizer: ControllableTranscribing()
        ).makeSession()
        _ = session.start()

        let result = session.stop()

        guard case .audio(let audio, _) = result else {
            Issue.record("expected .audio, got \(result)")
            return
        }
        #expect(audio.duration == 2.0)
        // The take is open until it is transcribed.
        #expect(session.phase == .transcribing)
    }

    @Test func stopRejectsRecordingShorterThanMinimum() {
        let capture = FakeAudioCapture(cannedAudio: makeAudio(duration: 0.2))
        let session = makeVoiceCapture(
            microphone: capture, recognizer: ControllableTranscribing()
        ).makeSession()
        _ = session.start()

        let result = session.stop()

        guard case .tooShort = result else {
            Issue.record("expected .tooShort, got \(result)")
            return
        }
        #expect(session.phase == .idle)
    }

    @Test func stopReportsNoAudioWhenEngineReturnsNothing() {
        let capture = FakeAudioCapture(cannedAudio: nil)
        let session = makeVoiceCapture(
            microphone: capture, recognizer: ControllableTranscribing()
        ).makeSession()
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
        let session = makeVoiceCapture(
            microphone: capture, recognizer: ControllableTranscribing(), captureDump: dump
        ).makeSession()
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
        let session = makeVoiceCapture(
            microphone: capture, recognizer: ControllableTranscribing(), captureDump: dump
        ).makeSession()
        _ = session.start()

        _ = session.stop()

        #expect(dump.saved.isEmpty)
    }

    /// Dump disabled means the store is never invoked — not invoked-and-discarded.
    @Test func stopDoesNotDumpWhenTheSettingIsDisabled() {
        let capture = FakeAudioCapture(
            cannedAudio: makeAudio(duration: 2.0, raw: makeRaw()))
        let dump = FakeCaptureDump()
        let session = makeVoiceCapture(
            microphone: capture, recognizer: ControllableTranscribing(), captureDump: dump,
            isCaptureDumpEnabled: { false }
        ).makeSession()
        _ = session.start()

        _ = session.stop()

        #expect(dump.saved.isEmpty)
    }

    @Test func stopDoesNotDumpWhenTheCaptureCarriesNoRawAudio() {
        let capture = FakeAudioCapture(cannedAudio: makeAudio(duration: 2.0, raw: nil))
        let dump = FakeCaptureDump()
        let session = makeVoiceCapture(
            microphone: capture, recognizer: ControllableTranscribing(), captureDump: dump
        ).makeSession()
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
        let session = makeVoiceCapture(
            microphone: capture, recognizer: ControllableTranscribing(), captureDump: dump
        ).makeSession()
        _ = session.start()

        session.cancel()

        #expect(dump.saved.isEmpty)
        #expect(session.phase == .idle)
    }

    /// The microphone and the recognizer are shared: a session that started
    /// nothing (a panel closing while dictation records, then transcribes)
    /// stops no one else's capture and cancels no one else's take.
    @Test func cancelLeavesAnotherSessionsCaptureAndTake() async throws {
        let capture = FakeAudioCapture(cannedAudio: makeAudio())
        let engine = ControllableTranscribing()
        let voiceCapture = makeVoiceCapture(microphone: capture, recognizer: engine)
        let dictation = voiceCapture.makeSession()
        let panel = voiceCapture.makeSession()
        guard case .started = dictation.start() else {
            Issue.record("expected dictation to start")
            return
        }

        panel.cancel()
        #expect(capture.isCapturing)
        #expect(capture.stopCount == 0)

        guard case .audio(let audio, _) = dictation.stop() else {
            Issue.record("expected audio")
            return
        }
        let recorder = CommitRecorder()
        let take = Task {
            await dictation.transcribeAndCommit(audio) { try await recorder.commit($0) }
        }
        while !engine.isAwaiting { await Task.yield() }
        panel.cancel()
        #expect(engine.cancelCount == 0)

        engine.completeWithSuccess()
        guard case .committed = await take.value else {
            Issue.record("expected dictation's take to commit")
            return
        }
        #expect(recorder.commits.count == 1)

        // Its own capture a session does stop.
        _ = panel.start()
        panel.cancel()
        #expect(!capture.isCapturing)
    }

    /// The recognizer is shared and cancels whichever transcription it runs:
    /// a session whose take is proofreading cancels only its own work, never
    /// the transcription another surface's take is waiting on (the capture
    /// panel's Escape while dictation transcribes).
    @Test func cancelWhileProofreadingLeavesAnotherSessionsTranscription() async throws {
        let engine = ControllableTranscribing(
            result: TranscriptionResult(
                text: "hello world", segments: [], language: "en", processingTime: 0))
        let stub = ProofreadStub()
        stub.gated = true
        let voiceCapture = makeVoiceCapture(
            microphone: FakeAudioCapture(cannedAudio: makeAudio()), recognizer: engine,
            proofread: { await stub.proofread($0) })
        let panel = voiceCapture.makeSession()
        let dictation = voiceCapture.makeSession()
        let recorder = CommitRecorder()

        let panelTake = Task {
            await panel.transcribeAndCommit(makeAudio()) { try await recorder.commit($0) }
        }
        while !engine.isAwaiting { await Task.yield() }
        engine.completeWithSuccess()
        while !stub.isAwaitingGate { await Task.yield() }
        #expect(panel.phase == .proofreading)
        stub.gated = false

        let dictationTake = Task {
            await dictation.transcribeAndCommit(makeAudio()) { try await recorder.commit($0) }
        }
        while !engine.isAwaiting { await Task.yield() }
        panel.cancel()
        #expect(engine.cancelCount == 0)

        engine.completeWithSuccess()
        guard case .committed = await dictationTake.value else {
            Issue.record("expected dictation's take to commit")
            return
        }
        #expect(recorder.commits.count == 1)
        stub.releaseGate()
        guard case .superseded = await panelTake.value else {
            Issue.record("expected the panel's take to be superseded")
            return
        }
        #expect(recorder.commits.count == 1)
    }

    /// While its own take is transcribing, a session's cancel does tell the
    /// recognizer.
    @Test func cancelWhileTranscribingCancelsItsOwnTranscription() async throws {
        let engine = ControllableTranscribing()
        let session = makeSession(engine: engine)
        let recorder = CommitRecorder()
        let take = Task {
            await session.transcribeAndCommit(makeAudio()) { try await recorder.commit($0) }
        }
        while !engine.isAwaiting { await Task.yield() }

        session.cancel()

        #expect(engine.cancelCount == 1)
        engine.completeWithCancellation()
        _ = await take.value
        #expect(recorder.commits.isEmpty)
    }

    // MARK: - transcribeAndCommit()

    @Test func transcribeAndCommitDeliversProcessedTextThenReportsCommitted() async throws {
        let engine = ControllableTranscribing(
            result: TranscriptionResult(
                text: "hello world", segments: [], language: "en", processingTime: 0))
        let session = makeSession(engine: engine)
        let recorder = CommitRecorder()

        let audio = makeAudio()
        let task = Task {
            await session.transcribeAndCommit(audio) { try await recorder.commit($0) }
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
        #expect(session.phase == .idle)
    }

    @Test func transcribeAndCommitReportsEmptyWhenPostProcessingYieldsNothing() async throws {
        let engine = ControllableTranscribing(
            result: TranscriptionResult(
                text: "   ", segments: [], language: "en", processingTime: 0))
        let session = makeSession(engine: engine)
        let recorder = CommitRecorder()

        let task = Task {
            await session.transcribeAndCommit(makeAudio()) { try await recorder.commit($0) }
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
        let session = makeSession(engine: engine)
        let recorder = CommitRecorder()

        let task = Task {
            await session.transcribeAndCommit(makeAudio()) { try await recorder.commit($0) }
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
        let session = makeSession(engine: engine)
        let recorder = CommitRecorder()

        let task = Task {
            await session.transcribeAndCommit(makeAudio()) { try await recorder.commit($0) }
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
        let session = makeSession(engine: engine)
        let recorder = CommitRecorder()

        let task = Task {
            await session.transcribeAndCommit(makeAudio()) { try await recorder.commit($0) }
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

    // MARK: - The Proofread Pass stage

    private func runThroughSession(
        engineText: String,
        stub: ProofreadStub,
        recorder: CommitRecorder
    ) async -> (outcome: VoiceCaptureSession.Outcome, session: VoiceCaptureSession) {
        let engine = ControllableTranscribing(
            result: TranscriptionResult(
                text: engineText, segments: [], language: "en", processingTime: 0))
        let session = makeSession(engine: engine, stub: stub)
        let task = Task {
            await session.transcribeAndCommit(makeAudio()) { try await recorder.commit($0) }
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

    /// `.rejected` commits nothing; the raw text rides the outcome, and the
    /// session keeps the take for "insert anyway".
    @Test func proofreadRejectedCommitsNothingAndCarriesTheRaw() async throws {
        let stub = ProofreadStub()
        stub.verdict = .rejected(reason: "unintelligible mumbling")
        let recorder = CommitRecorder()

        let (outcome, session) = await runThroughSession(
            engineText: "asdf ghjk", stub: stub, recorder: recorder)

        guard case .rejected(let raw, let reason) = outcome else {
            Issue.record("expected .rejected, got \(outcome)")
            return
        }
        #expect(raw == TranscriptionPostProcessor().process("asdf ghjk"))
        #expect(reason == "unintelligible mumbling")
        #expect(recorder.commits.isEmpty)
        #expect(session.hasRejectedTake)
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
        let session = makeSession(engine: engine, stub: stub)

        var takes: [VoiceCaptureSession.Take] = []
        var commitCountAtTake = -1
        let task = Task {
            await session.transcribeAndCommit(
                makeAudio(),
                onTake: { take in
                    takes.append(take)
                    commitCountAtTake = recorder.commits.count
                }
            ) { try await recorder.commit($0) }
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
        let session = makeSession(engine: engine, stub: stub)
        let task = Task {
            await session.transcribeAndCommit(makeAudio(), onTake: { takes.append($0) }) {
                try await recorder.commit($0)
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
        let stub = ProofreadStub()
        stub.gated = true
        stub.verdict = .corrected(text: "hello walrus", edits: [])
        let session = makeSession(engine: engine, stub: stub)
        let recorder = CommitRecorder()

        let task = Task {
            await session.transcribeAndCommit(makeAudio()) { try await recorder.commit($0) }
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
        let session = makeSession(engine: engine)
        let recorder = CommitRecorder()
        recorder.gated = true

        let task = Task {
            await session.transcribeAndCommit(makeAudio()) { try await recorder.commit($0) }
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

    // MARK: - Learned Words (PRD #612)

    private static let notes = TargetApp(bundleID: "com.apple.Notes", name: "Notes", pid: 42)

    private func makeLearnedWords() -> LearnedWordStore {
        LearnedWordStore(
            directory: FileManager.default.temporaryDirectory
                .appendingPathComponent("session-learned-\(UUID().uuidString)", isDirectory: true))
    }

    /// Runs one take through a session that knows the given Learned Words.
    private func runLearned(
        engineText: String, words: LearnedWordStore, proofread: ProofreadStub? = nil,
        app: TargetApp? = notes
    ) async -> (
        VoiceCaptureSession.Outcome, [VoiceCaptureSession.Take], CommitRecorder,
        VoiceCaptureSession
    ) {
        let engine = ControllableTranscribing(
            result: TranscriptionResult(
                text: engineText, segments: [], language: "en", processingTime: 0))
        let session = makeVoiceCapture(
            microphone: FakeAudioCapture(cannedAudio: makeAudio()), recognizer: engine,
            learnedWords: words, proofread: { text in await proofread?.proofread(text) },
            frontmostApp: { app }
        ).makeSession()
        _ = session.start()
        _ = session.stop()
        let recorder = CommitRecorder()
        var takes: [VoiceCaptureSession.Take] = []
        let task = Task {
            await session.transcribeAndCommit(makeAudio(), onTake: { takes.append($0) }) {
                try await recorder.commit($0)
            }
        }
        while !engine.isAwaiting { await Task.yield() }
        engine.completeWithSuccess()
        return (await task.value, takes, recorder, session)
    }

    /// Learned Words apply after the regex cleanup, so the take commits the
    /// owner's spelling, and the take carries both texts and the catches.
    @Test func learnedWordsApplyAfterTheCleanupAndBeforeTheCommit() async throws {
        let words = makeLearnedWords()
        words.learn(heard: "sract", meant: "Tesseract")
        let (outcome, takes, recorder, _) = await runLearned(
            engineText: "run the  SRACT tests", words: words)

        guard case .committed = outcome else {
            Issue.record("expected .committed, got \(outcome)")
            return
        }
        #expect(recorder.commits.first?.text == "Run the Tesseract tests")
        #expect(recorder.commits.first?.catches.map(\.meant) == ["Tesseract"])
        #expect(recorder.commits.first?.app == Self.notes)
        let take = try #require(takes.first)
        #expect(take.cleaned == "Run the SRACT tests")
        #expect(take.learned == "Run the Tesseract tests")
        #expect(take.catches.map(\.heard) == ["SRACT"])
        #expect(take.app == Self.notes)
    }

    /// A catch counts once the take is in, and only then.
    @Test func catchesCountWhenTheTakeCommits() async throws {
        let words = makeLearnedWords()
        let id = try #require(words.learn(heard: "cloud", meant: "Claude")?.wordID)
        _ = await runLearned(engineText: "ask cloud", words: words)
        #expect(words.word(withID: id)?.totalCatches == 1)

        let stub = ProofreadStub()
        stub.verdict = .rejected(reason: "mumbling")
        _ = await runLearned(engineText: "ask cloud", words: words, proofread: stub)
        #expect(words.word(withID: id)?.totalCatches == 1)
    }

    /// The Proofread Pass, when on, reads the text with the owner's words.
    @Test func theProofreadPassSeesTheLearnedText() async throws {
        let words = makeLearnedWords()
        words.learn(heard: "cloud", meant: "Claude")
        let stub = ProofreadStub()
        stub.verdict = nil
        _ = await runLearned(engineText: "ask cloud", words: words, proofread: stub)
        #expect(stub.seenTexts == ["Ask Claude"])
    }

    /// A word left alone in the take's app does not apply there.
    @Test func aWordLeftAloneInTheAppIsSkipped() async throws {
        let words = makeLearnedWords()
        let id = try #require(words.learn(heard: "cloud", meant: "Claude")?.wordID)
        words.leaveAlone(id, in: LearnedWord.App(bundleID: "com.apple.Notes", name: "Notes"))

        let (_, takes, recorder, _) = await runLearned(
            engineText: "the cloud lifts", words: words)
        #expect(recorder.commits.first?.text == "The cloud lifts")
        #expect(takes.first?.catches.isEmpty == true)
    }

    /// The take's app is the one in front when it started.
    @Test func startRemembersTheAppInFront() {
        let session = makeVoiceCapture(
            microphone: FakeAudioCapture(cannedAudio: makeAudio()),
            recognizer: ControllableTranscribing(), frontmostApp: { Self.notes }
        ).makeSession()
        #expect(session.targetApp == nil)
        _ = session.start()
        #expect(session.targetApp == Self.notes)
    }

    // MARK: - Insert anyway (ADR-0034, ADR-0086)

    /// "Insert anyway" commits the rejected take through the surface's commit
    /// like any take: its text as the Learned Words left it, its catches where
    /// they stand, its length and app — and counts the catches once it is in.
    /// The take is spent: a second press inserts nothing.
    @Test func aRejectedTakeCommitsAnywayLikeAnyTake() async throws {
        let words = makeLearnedWords()
        let id = try #require(words.learn(heard: "sract", meant: "Tesseract")?.wordID)
        let stub = ProofreadStub()
        stub.verdict = .rejected(reason: "mumbling")
        let (outcome, _, _, session) = await runLearned(
            engineText: "open the SRACT repo", words: words, proofread: stub)
        guard case .rejected = outcome else {
            Issue.record("expected .rejected, got \(outcome)")
            return
        }
        #expect(words.word(withID: id)?.totalCatches == 0)

        let recorder = CommitRecorder()
        let inserted = await session.commitRejectedTake { try await recorder.commit($0) }

        guard case .success = inserted else {
            Issue.record("expected the rejected take to commit")
            return
        }
        let take = try #require(recorder.commits.first)
        #expect(take.text == "Open the Tesseract repo")
        #expect(take.catches.map(\.meant) == ["Tesseract"])
        #expect(take.catches.first?.tokenStart == 2)
        #expect(take.duration == makeAudio().duration)
        #expect(take.app == Self.notes)
        #expect(words.word(withID: id)?.totalCatches == 1)
        #expect(!session.hasRejectedTake)
        let again = await session.commitRejectedTake { try await recorder.commit($0) }
        #expect(again == nil)
        #expect(recorder.commits.count == 1)
    }

    /// A take that commits leaves no rejected take behind to insert.
    @Test func aCommittedTakeLeavesNothingToInsertAnyway() async throws {
        let stub = ProofreadStub()
        let engine = ControllableTranscribing(
            result: TranscriptionResult(
                text: "asdf", segments: [], language: "en", processingTime: 0))
        let session = makeSession(engine: engine, stub: stub)
        let recorder = CommitRecorder()
        for verdict in [ProofreadVerdict.rejected(reason: "mumbling"), .unchanged] {
            stub.verdict = verdict
            let task = Task {
                await session.transcribeAndCommit(makeAudio()) { try await recorder.commit($0) }
            }
            while !engine.isAwaiting { await Task.yield() }
            engine.completeWithSuccess()
            _ = await task.value
        }
        #expect(!session.hasRejectedTake)
        #expect(recorder.commits.count == 1)
    }

    // MARK: - Silent captures (PRD #612)

    /// A capture that never rose above silence is neither transcribed nor
    /// dumped: Whisper writes "Thank you." over silence.
    @Test func stopRejectsASilentCaptureWithoutDumpingIt() {
        let silence = AudioData(
            samples: [Float](repeating: 0.0001, count: 16_000), sampleRate: 16_000, duration: 1.0,
            raw: makeRaw())
        let capture = FakeAudioCapture(cannedAudio: silence)
        let dump = FakeCaptureDump()
        let session = makeVoiceCapture(
            microphone: capture, recognizer: ControllableTranscribing(), captureDump: dump
        ).makeSession()
        _ = session.start()

        let result = session.stop()

        guard case .silent = result else {
            Issue.record("expected .silent, got \(result)")
            return
        }
        #expect(dump.saved.isEmpty)
    }

    // MARK: - An open take is one fact (the Models section)

    /// Whether a take is open — listening, transcribing, proofreading — is
    /// one fact across every surface's session: the Models section's
    /// Dictation and Proofreader rows read it, whatever surface took the take.
    @Test func aTakeOnAnySurfaceIsOneOpenTake() async throws {
        let engine = ControllableTranscribing()
        let stub = ProofreadStub()
        stub.gated = true
        let voiceCapture = makeVoiceCapture(
            microphone: FakeAudioCapture(cannedAudio: makeAudio()), recognizer: engine,
            proofread: { await stub.proofread($0) })
        let dictation = voiceCapture.makeSession()
        let today = voiceCapture.makeSession()
        #expect(!voiceCapture.isListening && !voiceCapture.isTranscribing)

        _ = today.start()
        #expect(voiceCapture.isListening)
        #expect(dictation.phase == .idle)

        guard case .audio(let audio, _) = today.stop() else {
            Issue.record("expected audio")
            return
        }
        #expect(!voiceCapture.isListening)
        #expect(voiceCapture.isTranscribing)

        let recorder = CommitRecorder()
        let task = Task {
            await today.transcribeAndCommit(audio) { try await recorder.commit($0) }
        }
        while !engine.isAwaiting { await Task.yield() }
        engine.completeWithSuccess()
        while !stub.isAwaitingGate { await Task.yield() }
        #expect(voiceCapture.isProofreading)
        #expect(!voiceCapture.isTranscribing)

        stub.releaseGate()
        _ = await task.value
        #expect(!voiceCapture.isListening && !voiceCapture.isTranscribing)
        #expect(!voiceCapture.isProofreading)
    }

    /// A press that can't open the microphone still supersedes the take in
    /// flight (it commits nothing), so the session reads idle at once — never
    /// stuck transcribing a take that will never land.
    @Test func aStartThatFailsLeavesNoTakeOpen() async throws {
        struct Boom: Error {}
        let engine = ControllableTranscribing()
        let capture = FakeAudioCapture(cannedAudio: makeAudio())
        let voiceCapture = makeVoiceCapture(microphone: capture, recognizer: engine)
        let session = voiceCapture.makeSession()
        _ = session.start()
        guard case .audio(let audio, _) = session.stop() else {
            Issue.record("expected audio")
            return
        }
        let recorder = CommitRecorder()
        let task = Task {
            await session.transcribeAndCommit(audio) { try await recorder.commit($0) }
        }
        while !engine.isAwaiting { await Task.yield() }
        #expect(voiceCapture.isTranscribing)

        capture.startError = Boom()
        guard case .captureFailed = session.start() else {
            Issue.record("expected the start to fail")
            return
        }
        #expect(session.phase == .idle)

        engine.completeWithSuccess()
        guard case .superseded = await task.value else {
            Issue.record("expected the take in flight to be superseded")
            return
        }
        #expect(session.phase == .idle)
        #expect(!voiceCapture.isTranscribing)
        #expect(recorder.commits.isEmpty)
    }

    // MARK: - The Live Preview reads like the take (ADR-0086)

    @MainActor @Observable
    final class PreviewRecorder {
        private(set) var shown: [LivePreview?] = []
        func show(_ preview: LivePreview?) { shown.append(preview) }
        var last: LivePreview? { shown.last ?? nil }
    }

    private func makeFakeModelBundle() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("session-preview-\(UUID().uuidString)", isDirectory: true)
        for model in ["AudioEncoder.mlmodelc", "TextDecoder.mlmodelc"] {
            try FileManager.default.createDirectory(
                at: directory.appendingPathComponent(model), withIntermediateDirectories: true)
        }
        return directory
    }

    /// The same take, heard the same way, reads the same in the preview as it
    /// commits: one text pipeline (the regex cleanup, then the Learned Words
    /// for the take's app) serves both. Catches show in the preview and count
    /// only once the take commits; the preview clears at release.
    @Test func thePreviewAndTheTakeReadTheSame() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }
        let recognizer = InMemorySpeechRecognizer(
            result: TranscriptionResult(
                text: "ask  cloud about it",
                segments: [
                    TranscriptionSegment(text: " ask  cloud about it", startTime: 0, endTime: 0.9)
                ],
                language: "en", processingTime: 0))
        let engine = TranscriptionEngine(makeRecognizer: { recognizer })
        try await engine.loadModel(from: bundle)
        let words = makeLearnedWords()
        let id = try #require(words.learn(heard: "cloud", meant: "Claude")?.wordID)
        let microphone = FakeAudioCapture(cannedAudio: makeAudio())
        microphone.cannedSnapshot = AudioData(
            samples: [Float](repeating: 0.1, count: 16_000), sampleRate: 16_000, duration: 1.0)
        let session = makeVoiceCapture(
            microphone: microphone, recognizer: engine, learnedWords: words,
            frontmostApp: { Self.notes }
        ).makeSession()
        let previews = PreviewRecorder()

        _ = session.start(preview: { previews.show($0) })
        #expect(await observe(until: { previews.last != nil }))
        let preview = try #require(previews.last)
        #expect(words.word(withID: id)?.totalCatches == 0)

        guard case .audio(let audio, _) = session.stop() else {
            Issue.record("expected audio")
            return
        }
        #expect(previews.last == nil)
        let recorder = CommitRecorder()
        _ = await session.transcribeAndCommit(audio) { try await recorder.commit($0) }

        let take = try #require(recorder.commits.first)
        #expect(preview.text == take.text)
        #expect(preview.text == "Ask Claude about it")
        #expect(preview.catches == take.catches)
        #expect(words.word(withID: id)?.totalCatches == 1)
    }
}
