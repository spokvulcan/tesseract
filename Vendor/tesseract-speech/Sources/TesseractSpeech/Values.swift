// TesseractSpeech — engine v2 public values.
// Normative contracts: docs/voice-engine-v2-spec.md §4, ADR-0038.

import Foundation

// MARK: - Model

public struct TTSModelSpec: Sendable, Equatable, Codable {
    public var repo: String
    public var precision: Precision

    public enum Precision: String, Sendable, Codable, Equatable {
        case q6, q8, bf16
    }

    public init(repo: String, precision: Precision) {
        self.repo = repo
        self.precision = precision
    }

    /// The one shipping checkpoint family (ADR-0037): quantized 1.7B VoiceDesign.
    /// The precision gate (ADR-0037/0039) picks q8 if the measured long-form peak
    /// RSS fits the ≤3 GB envelope, else q6.
    public static func voiceDesign17B(_ precision: Precision) -> TTSModelSpec {
        let suffix: String
        switch precision {
        case .q6: suffix = "6bit"
        case .q8: suffix = "8bit"
        case .bf16: suffix = "bf16"
        }
        return TTSModelSpec(
            repo: "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-\(suffix)",
            precision: precision)
    }

    /// Fingerprint component for PinnedVoice compatibility checks.
    var fingerprint: String { "\(repo)#\(precision.rawValue)" }
}

// MARK: - Voice

public enum Voice: Sendable, Equatable {
    /// Model-default voice.
    case standard(language: String?)
    /// VoiceDesign instruct description — what a voice *is*.
    case designed(description: String, language: String?)
    /// A prior realization, pinned: survives sessions and relaunches.
    case pinned(PinnedVoice)

    var description: String? {
        switch self {
        case .standard: return nil
        case .designed(let d, _): return d.isEmpty ? nil : d
        case .pinned(let p): return p.voiceDescription
        }
    }

    var language: String? {
        switch self {
        case .standard(let l), .designed(_, let l): return l
        case .pinned(let p): return p.language
        }
    }
}

/// Voice identity as a serializable value (ADR-0038): the design description plus
/// the anchor's code frames — a few KB, rebuildable into KV, fingerprinted so a
/// voice can never silently condition a different checkpoint or precision (#339:
/// voices do not survive precision changes; a seed never pins a voice).
public struct PinnedVoice: Sendable, Equatable, Codable {
    public static let currentSchema = 1

    public let schema: Int
    public let modelFingerprint: String
    public let voiceDescription: String?
    public let language: String?
    public let codeFrames: [[Int32]]

    public init(
        schema: Int = PinnedVoice.currentSchema,
        modelFingerprint: String,
        voiceDescription: String?,
        language: String?,
        codeFrames: [[Int32]]
    ) {
        self.schema = schema
        self.modelFingerprint = modelFingerprint
        self.voiceDescription = voiceDescription
        self.language = language
        self.codeFrames = codeFrames
    }

    public func serialized() throws -> Data {
        try JSONEncoder().encode(self)
    }

    public init(validating data: Data) throws {
        let decoded = try JSONDecoder().decode(PinnedVoice.self, from: data)
        guard decoded.schema == PinnedVoice.currentSchema else {
            throw SpeechEngineError.voiceIncompatible(
                expected: "schema \(PinnedVoice.currentSchema)", found: "schema \(decoded.schema)")
        }
        self = decoded
    }
}

// MARK: - Parameters & options

/// Sampler defaults. Deliberately no `seed` — seed is a per-utterance
/// reproducibility knob (`SpeechOptions`), never voice identity (ADR-0038).
public struct TTSParameters: Sendable, Equatable, Codable {
    public var temperature: Float
    public var topP: Float
    public var repetitionPenalty: Float
    public var maxTokens: Int

    public init(
        temperature: Float = 0.6,
        topP: Float = 0.8,
        repetitionPenalty: Float = 1.3,
        maxTokens: Int = 4096
    ) {
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.maxTokens = maxTokens
    }
}

public struct SpeechOptions: Sendable, Equatable {
    /// Reproducibility, not identity: `.fixed` reproduces the waveform only for
    /// identical (checkpoint, precision, text, options, anchor state).
    public var seed: Seed
    /// Per-utterance override of the session's parameter defaults.
    public var parameters: TTSParameters?

    public enum Seed: Sendable, Equatable {
        case entropy
        case fixed(UInt64)
    }

    public init(seed: Seed = .entropy, parameters: TTSParameters? = nil) {
        self.seed = seed
        self.parameters = parameters
    }

    public static let `default` = SpeechOptions()
}

// MARK: - Session profile (ADR-0037: roles are configs)

public enum AnchorPolicy: Sendable, Equatable {
    case none
    /// Anchor from the first `steps` codec frames of each utterance's first
    /// segment; conditions that utterance's later segments; dies with it.
    case perUtterance(steps: Int = 48)
    /// Anchor once from the session's first generated audio; retained until
    /// `close()` — every utterance speaks the same realization.
    case pinned(steps: Int = 48)
}

public enum PacingPolicy: Sendable, Equatable {
    /// Generate flat-out (short utterances, server, export).
    case eager
    /// At most the in-flight segment plus `segments` completed-but-undelivered
    /// segments exist; the engine then suspends *without the GPU lease* until
    /// the consumer pulls. Pause falls out: stop pulling.
    case lookahead(segments: Int = 1)
}

public struct SessionProfile: Sendable, Equatable {
    public var anchor: AnchorPolicy
    public var defaults: TTSParameters
    public var pacing: PacingPolicy

    public init(anchor: AnchorPolicy, defaults: TTSParameters = TTSParameters(), pacing: PacingPolicy) {
        self.anchor = anchor
        self.defaults = defaults
        self.pacing = pacing
    }

    /// Quality role: long-form read-aloud.
    public static let readAloud = SessionProfile(
        anchor: .perUtterance(), pacing: .lookahead(segments: 1))
    /// Fast role: companion utterances, pinned timbre.
    public static let companion = SessionProfile(
        anchor: .pinned(), pacing: .eager)
}

// MARK: - Lifecycle

public enum Readiness: Int, Sendable, Equatable, Comparable {
    case unloaded = 0
    case loaded = 1
    case warm = 2

    public static func < (lhs: Readiness, rhs: Readiness) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

public enum EnginePhase: Sendable, Equatable {
    case downloading(fraction: Double)
    case loadingWeights
    case warmingKernels
    case primingVoice
    case ready
}

/// Engine-scoped memory policy (ADR-0039). The engine NEVER writes the
/// process-global MLX cache limit — that knob belongs to the LLM stack.
public struct MemoryPolicy: Sendable, Equatable {
    /// One `Memory.clearCache()` when an utterance finishes (success or not).
    public var clearsCacheAtUtteranceEnd: Bool

    public init(clearsCacheAtUtteranceEnd: Bool = true) {
        self.clearsCacheAtUtteranceEnd = clearsCacheAtUtteranceEnd
    }

    public static let `default` = MemoryPolicy()
}

// MARK: - Stream values

public struct SegmentScript: Sendable, Equatable {
    public let index: Int
    public let text: String
    /// tokenCharOffsets[k] = character offset where alignment token k begins;
    /// one alignment token per codec frame (the K1 invariant), so frame
    /// (startFrame + k) lights text up to offsets[k+1].
    public let tokenCharOffsets: [Int]
    /// First codec frame of this segment, cumulative over the utterance —
    /// the Segment Window as ground truth.
    public let startFrame: Int

    public init(index: Int, text: String, tokenCharOffsets: [Int], startFrame: Int) {
        self.index = index
        self.text = text
        self.tokenCharOffsets = tokenCharOffsets
        self.startFrame = startFrame
    }
}

public struct AudioChunk: Sendable, Equatable {
    public let samples: [Float]
    /// Codec frames covered; utterance-cumulative, contiguous, gap-free.
    public let frames: Range<Int>
    public let segmentIndex: Int

    public init(samples: [Float], frames: Range<Int>, segmentIndex: Int) {
        self.samples = samples
        self.frames = frames
        self.segmentIndex = segmentIndex
    }
}

public enum SpeechEvent: Sendable, Equatable {
    /// Precedes every audio chunk of its segment.
    case segment(SegmentScript)
    case audio(AudioChunk)
    /// Follows the last chunk of segment `index`.
    case segmentDone(index: Int)
    /// Exactly once, iff the full text rendered. Terminal.
    case finished(SessionSummary)
}

public struct SessionSummary: Sendable, Equatable {
    public let totalFrames: Int
    public let segmentFrameCounts: [Int]

    public init(totalFrames: Int, segmentFrameCounts: [Int]) {
        self.totalFrames = totalFrames
        self.segmentFrameCounts = segmentFrameCounts
    }
}

// MARK: - Errors

public enum SpeechEngineError: Error, Sendable, Equatable {
    case modelUnavailable(String)
    case voiceIncompatible(expected: String, found: String)
    case generationFailed(String)
    case sessionClosed
    case engineUnloaded
}
