//
//  TranscriptionEngineTests.swift
//  tesseractTests
//
//  Exercises `TranscriptionEngine`'s real orchestration — the facade above the
//  `SpeechRecognizer` port — through the public interface, with an
//  `InMemorySpeechRecognizer` substituted below the seam. No WhisperKit, no model
//  files: model-file (`.mlmodelc`) verification is exercised against a temporary
//  fake bundle (empty `AudioEncoder.mlmodelc` / `TextDecoder.mlmodelc`), so the
//  facade's file-check is on a test surface while staying hermetic.
//

import Foundation
import os
import Testing

@testable import Tesseract_Agent

/// A `Sendable` factory spy: records how many recognizers the engine built (one
/// per `loadModel`) and hands back each instance, so a test can assert
/// reload-via-factory and inspect the fresh adapter. `Sendable` without
/// `@unchecked` — its state lives behind an `OSAllocatedUnfairLock`.
final class RecognizerFactorySpy: Sendable {
    private struct State {
        var builtCount = 0
        var recognizers: [InMemorySpeechRecognizer] = []
    }
    private let state = OSAllocatedUnfairLock(initialState: State())
    private let result: TranscriptionResult

    init(result: TranscriptionResult) {
        self.result = result
    }

    func make() -> any SpeechRecognizer {
        let recognizer = InMemorySpeechRecognizer(result: result)
        state.withLock {
            $0.builtCount += 1
            $0.recognizers.append(recognizer)
        }
        return recognizer
    }

    var builtCount: Int { state.withLock { $0.builtCount } }
    var recognizers: [InMemorySpeechRecognizer] { state.withLock { $0.recognizers } }
}

/// A recognizer that deliberately ignores task cancellation. It suspends until
/// the test releases it, then returns success even if `cancelTranscription()`
/// already cancelled the engine's retained task.
actor CancellationIgnoringSpeechRecognizer: SpeechRecognizer {
    private let result: TranscriptionResult
    private var continuation: CheckedContinuation<Void, Never>?
    private(set) var loadCount = 0
    private(set) var transcribeCount = 0

    init(
        result: TranscriptionResult = TranscriptionResult(
            text: "late success",
            segments: [],
            language: "en",
            processingTime: 0
        )
    ) {
        self.result = result
    }

    var isAwaitingRelease: Bool { continuation != nil }

    func load(modelPath: URL) async throws {
        loadCount += 1
    }

    func transcribe(_ audioData: AudioData, language: String?) async throws -> TranscriptionResult {
        transcribeCount += 1
        await withCheckedContinuation { continuation = $0 }
        return result
    }

    func release() {
        continuation?.resume()
        continuation = nil
    }
}

@MainActor
struct TranscriptionEngineTests {

    // MARK: - Helpers

    /// Creates a temporary fake model bundle with empty `.mlmodelc` entries that
    /// the facade's file verification accepts. Caller removes it.
    private func makeFakeModelBundle() throws -> URL {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory
            .appendingPathComponent(
                "TranscriptionEngineTests-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(
            at: dir.appendingPathComponent("AudioEncoder.mlmodelc"),
            withIntermediateDirectories: true)
        try fm.createDirectory(
            at: dir.appendingPathComponent("TextDecoder.mlmodelc"),
            withIntermediateDirectories: true)
        return dir
    }

    private func sampleAudio(duration: TimeInterval = 1.0) -> AudioData {
        AudioData(samples: [0.1, 0.2, 0.3], sampleRate: 16_000, duration: duration)
    }

    /// Runs a throwing transcription op and returns the `DictationError` it threw,
    /// failing the test if it didn't throw a `DictationError`. (`DictationError`
    /// is not `Equatable`, so cases are matched with `if case` at the call site.)
    private func captureDictationError(
        _ op: () async throws -> TranscriptionResult,
        sourceLocation: SourceLocation = #_sourceLocation
    ) async -> DictationError? {
        do {
            _ = try await op()
            Issue.record("expected a DictationError to be thrown", sourceLocation: sourceLocation)
            return nil
        } catch let error as DictationError {
            return error
        } catch {
            Issue.record("expected DictationError, got \(error)", sourceLocation: sourceLocation)
            return nil
        }
    }

    // MARK: - Tracer: load + transcribe across the seam

    @Test
    func loadsFakeBundleAndTranscribesThroughTheSeam() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }

        let recognizer = InMemorySpeechRecognizer(
            result: TranscriptionResult(
                text: "hello world", segments: [], language: "en", processingTime: 0)
        )
        let engine = TranscriptionEngine(makeRecognizer: { recognizer })

        try await engine.loadModel(from: bundle)
        #expect(engine.isModelLoaded)

        let result = try await engine.transcribe(sampleAudio(), language: "auto")

        #expect(result.text == "hello world")
        #expect(await recognizer.loadCount == 1)
        #expect(await recognizer.loadedPaths == [bundle])
        #expect(await recognizer.transcribeCount == 1)
        // "auto" maps to nil (auto-detect) at the facade.
        #expect(await recognizer.recordedLanguages == [nil])
        #expect(!engine.isTranscribing)
    }

    @Test
    func explicitLanguageIsPassedThroughToTheRecognizer() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }
        let recognizer = InMemorySpeechRecognizer()
        let engine = TranscriptionEngine(makeRecognizer: { recognizer })
        try await engine.loadModel(from: bundle)

        _ = try await engine.transcribe(sampleAudio(), language: "es")

        #expect(await recognizer.recordedLanguages == ["es"])
    }

    // MARK: - Model-file verification (stays on the facade)

    @Test
    func loadRejectsBundleMissingModelFiles() async {
        // A directory without AudioEncoder.mlmodelc / TextDecoder.mlmodelc.
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent(
            "TE-empty-\(UUID().uuidString)", isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }

        let recognizer = InMemorySpeechRecognizer()
        let engine = TranscriptionEngine(makeRecognizer: { recognizer })

        await #expect(throws: DictationError.self) {
            try await engine.loadModel(from: dir)
        }
        #expect(!engine.isModelLoaded)
        // The port was never asked to load — verification is above the seam.
        #expect(await recognizer.loadCount == 0)
    }

    // MARK: - Empty-audio guard

    @Test
    func emptyAudioIsRejectedWithoutCallingTheRecognizer() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }
        let recognizer = InMemorySpeechRecognizer()
        let engine = TranscriptionEngine(makeRecognizer: { recognizer })
        try await engine.loadModel(from: bundle)

        let empty = AudioData(samples: [], sampleRate: 16_000, duration: 0)
        let error = await captureDictationError {
            try await engine.transcribe(empty, language: "auto")
        }

        guard case .noSpeechDetected = error else {
            Issue.record("expected .noSpeechDetected, got \(String(describing: error))")
            return
        }
        #expect(await recognizer.transcribeCount == 0)
    }

    // MARK: - Timeout race (stays on the facade)

    @Test
    func timeoutFiresWhenRecognizerOverrunsTheBudget() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }
        let recognizer = InMemorySpeechRecognizer(latency: .seconds(60))
        let engine = TranscriptionEngine(
            makeRecognizer: { recognizer },
            timeout: { _ in .milliseconds(20) }
        )
        try await engine.loadModel(from: bundle)

        let error = await captureDictationError {
            try await engine.transcribe(sampleAudio(), language: "auto")
        }

        guard case .transcriptionFailed = error else {
            Issue.record(
                "expected .transcriptionFailed (timeout), got \(String(describing: error))")
            return
        }
        #expect(!engine.isTranscribing)
    }

    // MARK: - Cancellation reaches the adapter

    @Test
    func cancelTranscriptionInterruptsInFlightTranscribeAtTheAdapter() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }
        let recognizer = InMemorySpeechRecognizer(latency: .seconds(60))
        let engine = TranscriptionEngine(
            makeRecognizer: { recognizer }, timeout: { _ in .seconds(120) })
        try await engine.loadModel(from: bundle)

        let task = Task { try await engine.transcribe(sampleAudio(), language: "auto") }
        while await recognizer.transcribeCount == 0 { await Task.yield() }

        engine.cancelTranscription()

        // The over-running transcribe is interrupted (CancellationError propagates
        // from the retained task through the race into the adapter).
        await #expect(throws: CancellationError.self) { try await task.value }
        while await !recognizer.transcribeWasInterrupted { await Task.yield() }
        #expect(await recognizer.transcribeWasInterrupted)
        #expect(!engine.isTranscribing)
    }

    @Test
    func cancelTranscriptionRejectsLateSuccessFromCancellationIgnoringRecognizer() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }
        let recognizer = CancellationIgnoringSpeechRecognizer()
        let engine = TranscriptionEngine(
            makeRecognizer: { recognizer }, timeout: { _ in .seconds(120) })
        try await engine.loadModel(from: bundle)

        let task = Task { try await engine.transcribe(sampleAudio(), language: "auto") }
        while true {
            let transcribeCount = await recognizer.transcribeCount
            let isAwaitingRelease = await recognizer.isAwaitingRelease
            if transcribeCount > 0 && isAwaitingRelease { break }
            await Task.yield()
        }

        engine.cancelTranscription()

        // The caller is released by the race's cancellation arm immediately —
        // even though the recognizer is still suspended and ignoring the
        // cooperative cancel. Its late success (after release) goes nowhere.
        await #expect(throws: CancellationError.self) { try await task.value }
        await recognizer.release()
        #expect(!engine.isTranscribing)
    }

    /// The item-10 regression (audit #285): a recognizer hung *between*
    /// cancellation checks used to defeat the timeout entirely — the
    /// structured race could not exit until the hung child yielded, pinning
    /// `.processing` forever. The abandonment race must release the caller
    /// the moment the budget expires, while the recognizer is still suspended.
    @Test
    func timeoutReleasesTheCallerWhileTheRecognizerIsStillHung() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }
        let recognizer = CancellationIgnoringSpeechRecognizer()
        let engine = TranscriptionEngine(
            makeRecognizer: { recognizer },
            timeout: { _ in .milliseconds(20) }
        )
        try await engine.loadModel(from: bundle)

        // The transcribe returns (with the timeout error) while the recognizer
        // is STILL suspended — nothing below the seam has yielded.
        let error = await captureDictationError {
            try await engine.transcribe(sampleAudio(), language: "auto")
        }
        guard case .transcriptionFailed = error else {
            Issue.record(
                "expected .transcriptionFailed (timeout), got \(String(describing: error))")
            return
        }
        #expect(await recognizer.isAwaitingRelease)
        #expect(!engine.isTranscribing)

        // The slot is free again: a fresh engine cycle works after the orphan
        // is finally released and its late result is dropped.
        await recognizer.release()
    }

    // MARK: - Error mapping (stays on the facade)

    @Test
    func modelFailureIsMappedOntoDictationError() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }
        let recognizer = InMemorySpeechRecognizer(
            transcribeError: FakeModelError(message: "kernel panic"))
        let engine = TranscriptionEngine(makeRecognizer: { recognizer })
        try await engine.loadModel(from: bundle)

        let error = await captureDictationError {
            try await engine.transcribe(sampleAudio(), language: "auto")
        }

        guard case .transcriptionFailed = error else {
            Issue.record(
                "expected model failure mapped to .transcriptionFailed, got \(String(describing: error))"
            )
            return
        }
        #expect(!engine.isTranscribing)
    }

    // MARK: - Single-in-flight semantics

    @Test
    func secondTranscribeWhileInFlightThrowsTranscriptionInProgress() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }
        let recognizer = InMemorySpeechRecognizer(latency: .seconds(60))
        let engine = TranscriptionEngine(
            makeRecognizer: { recognizer }, timeout: { _ in .seconds(120) })
        try await engine.loadModel(from: bundle)

        let first = Task { try await engine.transcribe(sampleAudio(), language: "auto") }
        // Wait until the first transcription is actually running below the seam,
        // which means the in-flight slot is occupied.
        while await recognizer.transcribeCount == 0 { await Task.yield() }

        let error = await captureDictationError {
            try await engine.transcribe(sampleAudio(), language: "auto")
        }
        guard case .transcriptionInProgress = error else {
            Issue.record("expected .transcriptionInProgress, got \(String(describing: error))")
            return
        }

        // Cancel the in-flight one and let it unwind. The identity-guarded `defer`
        // must free the in-flight slot on *this* engine.
        engine.cancelTranscription()
        _ = try? await first.value

        // A fresh transcribe on the SAME engine now succeeds — proving the slot
        // was cleared after cancellation (not just that a brand-new engine works).
        await recognizer.setLatency(nil)
        let result = try await engine.transcribe(sampleAudio(), language: "auto")
        #expect(result.text == "canned transcription")
    }

    @Test
    func freshTranscribeSucceedsAfterPreviousCompletes() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }
        let recognizer = InMemorySpeechRecognizer(
            result: TranscriptionResult(text: "ok", segments: [], language: "en", processingTime: 0)
        )
        let engine = TranscriptionEngine(makeRecognizer: { recognizer })
        try await engine.loadModel(from: bundle)

        // Two sequential transcriptions: the slot must be free again after the first.
        let r1 = try await engine.transcribe(sampleAudio(), language: "auto")
        let r2 = try await engine.transcribe(sampleAudio(), language: "auto")
        #expect(r1.text == "ok")
        #expect(r2.text == "ok")
        #expect(await recognizer.transcribeCount == 2)
    }

    @Test
    func transcriptionInProgressErrorHasExpectedMessaging() {
        let error = DictationError.transcriptionInProgress
        #expect(error.errorDescription == "A transcription is already in progress.")
        #expect(
            error.recoverySuggestion
                == "Wait for the current transcription to finish, or cancel it first.")
    }

    // MARK: - Lazy load + factory lifecycle

    @Test
    func modelLoadsOnceAcrossRepeatedTranscriptions() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }
        let spy = RecognizerFactorySpy(
            result: TranscriptionResult(text: "x", segments: [], language: "en", processingTime: 0))
        let engine = TranscriptionEngine(makeRecognizer: { spy.make() })
        try await engine.loadModel(from: bundle)

        _ = try await engine.transcribe(sampleAudio(), language: "auto")
        _ = try await engine.transcribe(sampleAudio(), language: "auto")

        #expect(spy.builtCount == 1)
        #expect(await spy.recognizers[0].loadCount == 1)
        #expect(await spy.recognizers[0].transcribeCount == 2)
    }

    @Test
    func unloadThenTranscribeReloadsViaTheFactory() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }
        let spy = RecognizerFactorySpy(
            result: TranscriptionResult(text: "x", segments: [], language: "en", processingTime: 0))
        let engine = TranscriptionEngine(makeRecognizer: { spy.make() })

        try await engine.loadModel(from: bundle)
        _ = try await engine.transcribe(sampleAudio(), language: "auto")

        engine.unloadModel()
        #expect(!engine.isModelLoaded)

        // Lazy reload on next use, via a *fresh* adapter from the factory.
        _ = try await engine.transcribe(sampleAudio(), language: "auto")

        #expect(spy.builtCount == 2)
        #expect(engine.isModelLoaded)
        #expect(await spy.recognizers[1].loadCount == 1)
    }
}
