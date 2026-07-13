// Test adapters: a scripted Speech Synthesizer and a recording GPU lease.

import Foundation
@testable import TesseractSpeech

/// Deterministic synthesizer: N chunks of M samples per segment, optional
/// per-chunk delay, full request recording, cancellation observation.
actor ScriptedSynthesizer: SpeechSynthesizing {
    struct Script {
        var chunksPerSegment = 3
        var samplesPerChunk = 1920 * 2  // 2 frames per chunk
        var chunkDelayNanos: UInt64 = 0
        var failOnSegmentIndex: Int? = nil
    }

    var script = Script()
    private(set) var requests: [SegmentRequest] = []
    private(set) var loadCount = 0
    private(set) var warmUpCount = 0
    private(set) var primedVoices: [String?] = []
    private(set) var unloadCount = 0
    private(set) var trimCount = 0
    private(set) var segmentsStarted = 0
    private(set) var segmentsFinished = 0
    private(set) var sawCancellation = false

    func configure(_ script: Script) { self.script = script }

    func load(_ spec: TTSModelSpec, onPhase: (@Sendable (EnginePhase) -> Void)?) async throws {
        loadCount += 1
        onPhase?(.ready)
    }

    func warmUp() async throws { warmUpCount += 1 }

    func primeVoice(description: String?, language: String?) async throws {
        primedVoices.append(description)
    }

    func unload() async { unloadCount += 1 }

    func audioFormat() async -> AudioFormat? {
        AudioFormat(sampleRate: 24_000, samplesPerFrame: 1920)
    }

    func alignmentOffsets(for text: String) async throws -> [Int] {
        Array(0..<max(1, text.count / 4)).map { $0 * 4 }
    }

    func trimCaches() async { trimCount += 1 }

    private func noteStarted() { segmentsStarted += 1 }
    private func noteFinished() { segmentsFinished += 1 }
    private func noteCancelled() { sawCancellation = true }
    private func record(_ request: SegmentRequest) { requests.append(request) }

    func synthesizeSegment(_ request: SegmentRequest) async
        -> AsyncThrowingStream<SynthesisEvent, Error>
    {
        record(request)
        let script = self.script
        let segmentIndex = requests.count - 1
        return AsyncThrowingStream { continuation in
            let task = Task {
                await self.noteStarted()
                do {
                    if script.failOnSegmentIndex == segmentIndex {
                        throw SpeechEngineError.generationFailed("scripted failure")
                    }
                    for _ in 0..<script.chunksPerSegment {
                        if script.chunkDelayNanos > 0 {
                            try await Task.sleep(nanoseconds: script.chunkDelayNanos)
                        }
                        try Task.checkCancellation()
                        continuation.yield(.chunk([Float](repeating: 0.1, count: script.samplesPerChunk)))
                    }
                    var captured: AnchorHandle?
                    if let steps = request.captureAnchorSteps {
                        captured = AnchorHandle(
                            codeFrames: (0..<min(steps, 4)).map { _ in [Int32](repeating: 7, count: 3) },
                            voiceDescription: request.voiceDescription,
                            language: request.language)
                    }
                    continuation.yield(.done(capturedAnchor: captured))
                    await self.noteFinished()
                    continuation.finish()
                } catch is CancellationError {
                    await self.noteCancelled()
                    continuation.finish(throwing: CancellationError())
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

/// Records lease usage; asserts non-nesting. `depth` observable for the
/// "never held across a demand wait" check.
actor RecordingLease: GPULeasing {
    private(set) var acquisitions = 0
    private(set) var depth = 0
    private(set) var maxDepth = 0

    func withLease<T: Sendable>(_ body: @Sendable () async throws -> T) async throws -> T {
        enter()
        defer { exit() }
        return try await body()
    }

    private func enter() {
        acquisitions += 1
        depth += 1
        maxDepth = max(maxDepth, depth)
    }

    private func exit() { depth -= 1 }
}
