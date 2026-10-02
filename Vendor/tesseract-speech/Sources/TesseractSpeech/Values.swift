// TesseractSpeech — engine v2 public values.
// Normative contracts: docs/voice-engine-v2-spec.md §4, ADR-0038.

import Foundation

// MARK: - Model

public struct TTSModelSpec: Sendable, Equatable, Codable {
    public var repo: String
    public var precision: Precision
    /// The talker head whose attention follows the text, which times each
    /// spoken word (ADR-0077); nil for a checkpoint nobody measured, whose
    /// words then come untimed.
    public var alignmentHead: AlignmentHead?

    public enum Precision: String, Sendable, Codable, Equatable {
        case q6, q8, bf16
    }

    public init(repo: String, precision: Precision, alignmentHead: AlignmentHead? = nil) {
        self.repo = repo
        self.precision = precision
        self.alignmentHead = alignmentHead
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
            precision: precision, alignmentHead: .qwen3TTS17B)
    }

    /// Fingerprint component for PinnedVoice compatibility checks: a take
    /// only conditions the checkpoint and precision that rendered it.
    public var fingerprint: String { "\(repo)#\(precision.rawValue)" }
}

/// A talker attention head that follows the text while the voice speaks it:
/// its attention sits on the token being said, frame by frame (ADR-0077). A
/// property of the network, found once per checkpoint family by measurement
/// (docs/research/2026-09-27-word-timing-from-attention.md).
public struct AlignmentHead: Sendable, Equatable, Codable {
    public let layer: Int
    public let head: Int

    public init(layer: Int, head: Int) {
        self.layer = layer
        self.head = head
    }

    /// Qwen3-TTS 1.7B (VoiceDesign, at 6 and 8 bits; any voice or language).
    public static let qwen3TTS17B = AlignmentHead(layer: 3, head: 0)
    /// Qwen3-TTS 0.6B (CustomVoice).
    public static let qwen3TTS06B = AlignmentHead(layer: 6, head: 5)
}

// MARK: - Voice

public enum Voice: Sendable, Equatable {
    /// Model-default voice.
    case standard(language: String?)
    /// VoiceDesign instruct description — what a voice *is*.
    case designed(description: String, language: String?)
    /// A designed voice with its Reference Take: survives sessions and
    /// relaunches.
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

/// Voice identity as a serializable value (ADR-0038, ADR-0072): the design
/// description plus its Reference Take (the codec frames and the text they
/// speak), a few KB. Fingerprinted so a voice can never condition a different
/// checkpoint or precision (#339: voices do not survive precision changes; a
/// seed never pins a voice).
public struct PinnedVoice: Sendable, Equatable, Codable {
    /// 2: a Reference Take (frames + text). 1 was the 48-frame anchor, which
    /// carried no text and cannot condition the in-context layout.
    public static let currentSchema = 2

    public let schema: Int
    public let modelFingerprint: String
    public let voiceDescription: String?
    public let language: String?
    public let referenceText: String
    public let codeFrames: [[Int32]]

    public init(
        schema: Int = PinnedVoice.currentSchema,
        modelFingerprint: String,
        voiceDescription: String?,
        language: String?,
        referenceText: String,
        codeFrames: [[Int32]]
    ) {
        self.schema = schema
        self.modelFingerprint = modelFingerprint
        self.voiceDescription = voiceDescription
        self.language = language
        self.referenceText = referenceText
        self.codeFrames = codeFrames
    }

    public func serialized() throws -> Data {
        try JSONEncoder().encode(self)
    }

    /// Decodes a serialized voice; any other schema is `voiceIncompatible`.
    public init(validating data: Data) throws {
        struct Envelope: Decodable { let schema: Int }
        let found = try JSONDecoder().decode(Envelope.self, from: data).schema
        guard found == PinnedVoice.currentSchema else {
            throw SpeechEngineError.voiceIncompatible(
                expected: "schema \(PinnedVoice.currentSchema)", found: "schema \(found)")
        }
        self = try JSONDecoder().decode(PinnedVoice.self, from: data)
    }

    var referenceTake: ReferenceTake {
        ReferenceTake(codeFrames: codeFrames, text: referenceText)
    }
}

// MARK: - Parameters & options

/// Sampler defaults. Deliberately no `seed` — seed is a per-utterance
/// reproducibility knob (`SpeechOptions`), never voice identity (ADR-0038).
///
/// Two temperatures (ADR-0072): `temperature` for the talker, which carries
/// the words and the delivery, and `detailTemperature` for the code
/// predictor's acoustic detail, where most of the timbre lives. The defaults
/// are the setting the owner listened to: an expressive 0.9 reading with a
/// steady 0.5 timbre and Qwen's own repetition penalty.
public struct TTSParameters: Sendable, Equatable, Codable {
    public var temperature: Float
    public var topP: Float
    public var repetitionPenalty: Float
    public var detailTemperature: Float
    public var maxTokens: Int

    public init(
        temperature: Float = 0.9,
        topP: Float = 1.0,
        repetitionPenalty: Float = 1.05,
        detailTemperature: Float = 0.5,
        maxTokens: Int = 4096
    ) {
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.detailTemperature = detailTemperature
        self.maxTokens = maxTokens
    }
}

public struct SpeechOptions: Sendable, Equatable {
    /// Reproducibility, not identity: `.fixed` reproduces the waveform only for
    /// identical (checkpoint, precision, text, options, Reference Take).
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

/// Whether a session keeps its voice (ADR-0072).
public enum ReferencePolicy: Sendable, Equatable {
    /// Every segment is rendered from the description alone, so each one is
    /// a new realization of it. For measurements and tests.
    case none
    /// One Reference Take per session: the pinned voice it opened with, else
    /// the first segment it renders. Every other segment continues it, until
    /// `close()` or a retake.
    case pinned
}

public enum PacingPolicy: Sendable, Equatable {
    /// Generate flat-out (short utterances, server, export).
    case eager
    /// At most the in-flight segment plus `segments` completed-but-undelivered
    /// segments exist; the engine then suspends until the consumer pulls.
    /// Pause falls out: stop pulling.
    case lookahead(segments: Int = 1)
}

public struct SessionProfile: Sendable, Equatable {
    public var reference: ReferencePolicy
    public var defaults: TTSParameters
    public var pacing: PacingPolicy

    public init(
        reference: ReferencePolicy, defaults: TTSParameters = TTSParameters(),
        pacing: PacingPolicy
    ) {
        self.reference = reference
        self.defaults = defaults
        self.pacing = pacing
    }

    /// Quality role: long-form read-aloud.
    public static let readAloud = SessionProfile(
        reference: .pinned, pacing: .lookahead(segments: 1))
    /// Fast role: companion utterances.
    public static let companion = SessionProfile(
        reference: .pinned, pacing: .eager)
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

/// No download phase: the engine only loads a checkpoint already on disk.
/// Fetching it is the app's model download manager's job.
public enum EnginePhase: Sendable, Equatable {
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
    /// First codec frame of this segment, cumulative over the utterance —
    /// the Segment Window as ground truth.
    public let startFrame: Int

    public init(index: Int, text: String, startFrame: Int) {
        self.index = index
        self.text = text
        self.startFrame = startFrame
    }
}

extension Character {
    /// What splits speech into words: whitespace and newlines, judged on the
    /// whole Character, so a mark joined to a space is part of the space.
    /// `WordStart.word` counts words split by it, so anything that finds
    /// those words in the text splits by it too.
    public var separatesWords: Bool { isWhitespace || isNewline }
}

/// Where a spoken word starts: its place among the segment's words (split by
/// `Character.separatesWords`, from 0) and the codec frame its sound starts in.
public struct WordStart: Sendable, Equatable {
    public let word: Int
    public let frame: Int

    public init(word: Int, frame: Int) {
        self.word = word
        self.frame = frame
    }
}

/// Newly timed words of one segment, in order, with frames counted over the
/// utterance (ADR-0077). Every frame is one of audio already delivered.
public struct WordTiming: Sendable, Equatable {
    public let segmentIndex: Int
    public let starts: [WordStart]

    public init(segmentIndex: Int, starts: [WordStart]) {
        self.segmentIndex = segmentIndex
        self.starts = starts
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
    /// Words of the open segment whose starts are now known, after the audio
    /// they start in. Only a model with an alignment head times its words.
    case words(WordTiming)
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

extension SpeechEngineError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .modelUnavailable(let detail):
            return "Voice model unavailable: \(detail)"
        case .voiceIncompatible(let expected, let found):
            return "This pinned voice was made with \(found); the engine is running \(expected)."
        case .generationFailed(let detail):
            return "Speech generation failed: \(detail)"
        case .sessionClosed:
            return "The speech session is closed."
        case .engineUnloaded:
            return "The speech engine is unloaded."
        }
    }
}
