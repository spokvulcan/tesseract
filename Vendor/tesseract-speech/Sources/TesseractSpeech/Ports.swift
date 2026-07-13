// TesseractSpeech — the engine's downward seams (ADR-0038 §dependency strategy).
// Each port has two adapters: a production one and a test one.

import Foundation

// MARK: - Speech Synthesizer port (CONTEXT.md name kept; model-only)

/// One segment's synthesis request. Conditioning is *data in the request* —
/// the port has no mutable voice state (v1's T3/M3 made unrepresentable).
public struct SegmentRequest: Sendable {
    public var text: String
    public var voiceDescription: String?
    public var language: String?
    public var parameters: TTSParameters
    /// Resolved by the engine: `.entropy` becomes a fresh random value per
    /// utterance, so the port only ever sees a concrete seed.
    public var seed: UInt64
    /// Condition generation on this anchor (same realization as its source).
    public var anchor: AnchorHandle?
    /// Capture an anchor from this segment's first N codec frames; it arrives
    /// with `SynthesisEvent.done` — no post-hoc build call, no race window.
    public var captureAnchorSteps: Int?

    public init(
        text: String, voiceDescription: String?, language: String?,
        parameters: TTSParameters, seed: UInt64,
        anchor: AnchorHandle? = nil, captureAnchorSteps: Int? = nil
    ) {
        self.text = text
        self.voiceDescription = voiceDescription
        self.language = language
        self.parameters = parameters
        self.seed = seed
        self.anchor = anchor
        self.captureAnchorSteps = captureAnchorSteps
    }
}

/// A voice realization's anchor as a value: the code frames are the durable
/// ingredients (they serialize into PinnedVoice); adapters rebuild KV from
/// them whenever the active anchor changes.
public struct AnchorHandle: Sendable, Equatable {
    public let codeFrames: [[Int32]]
    public let voiceDescription: String?
    public let language: String?

    public init(codeFrames: [[Int32]], voiceDescription: String?, language: String?) {
        self.codeFrames = codeFrames
        self.voiceDescription = voiceDescription
        self.language = language
    }
}

public enum SynthesisEvent: Sendable {
    case chunk([Float])
    /// Terminal on success. Carries the captured anchor iff requested.
    case done(capturedAnchor: AnchorHandle?)
}

public struct AudioFormat: Sendable, Equatable {
    public let sampleRate: Int
    public let samplesPerFrame: Int

    public init(sampleRate: Int, samplesPerFrame: Int) {
        self.sampleRate = sampleRate
        self.samplesPerFrame = samplesPerFrame
    }

    public var framesPerSecond: Double {
        Double(sampleRate) / Double(samplesPerFrame)
    }
}

public protocol SpeechSynthesizing: Sendable {
    /// Resolve/download + load weights. Idempotent for the same spec.
    func load(_ spec: TTSModelSpec, onPhase: (@Sendable (EnginePhase) -> Void)?) async throws
    /// Kernel/JIT + fused-weight + tokenizer warmup. Requires loaded.
    func warmUp() async throws
    /// Precompute the instruct-prefix KV for a voice, off the hot path.
    func primeVoice(description: String?, language: String?) async throws
    /// Deterministic release of weights, KV, caches; GPU-stream sync.
    func unload() async
    /// Available once loaded.
    func audioFormat() async -> AudioFormat?
    /// Per-alignment-token character offsets for `text` (the K1 path).
    func alignmentOffsets(for text: String) async throws -> [Int]
    /// Stream one segment. Chunks arrive in order; `.done` is terminal on
    /// success; errors/cancellation terminate the stream. Cancellation of the
    /// consuming task stops generation within one decoder step.
    func synthesizeSegment(_ request: SegmentRequest) async
        -> AsyncThrowingStream<SynthesisEvent, Error>
    /// One buffer-pool trim (utterance end / teardown). Never touches the
    /// process-global cache limit (ADR-0039).
    func trimCaches() async
}

// MARK: - GPU leasing port (production adapter wraps InferenceArbiter)

public protocol GPULeasing: Sendable {
    func withLease<T: Sendable>(_ body: @Sendable () async throws -> T) async throws -> T
}

// MARK: - Diagnostics tap (injected, default off — autopsy constraint 8)

public protocol SpeechDiagnosticsTap: Sendable {
    func event(_ name: StaticString, _ detail: @autoclosure @Sendable () -> String)
}
