@preconcurrency import MLX
@preconcurrency import MLXLMCommon
import MLXAudioCore

public protocol SpeechGenerationModel: AnyObject {
    var sampleRate: Int { get }

    /// Random seed for deterministic generation. Set before calling generate/generateStream.
    var seed: UInt64 { get set }

    /// Last generated codec codes from the most recent generation call.
    /// Used to build a voice anchor for consistent voice across segments.
    var lastGeneratedCodes: [MLXArray]? { get }

    func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray

    func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error>

    func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters,
        useVoiceAnchor: Bool
    ) -> AsyncThrowingStream<AudioGeneration, Error>

    /// Build a voice anchor KV cache from previously generated codec codes.
    func buildVoiceAnchor(
        referenceCount: Int,
        instruct: String?,
        language: String?
    )

    /// Clear the voice anchor KV cache and associated state.
    func clearVoiceAnchor()

    /// Cancel any in-progress generation immediately.
    func cancelGeneration()

    /// Tokenize text and return per-token character offsets for word-level alignment.
    /// Each token maps to one codec step (~80ms of audio).
    func tokenizeForAlignment(text: String) -> [Int]
}

extension SpeechGenerationModel {
    public var seed: UInt64 {
        get { 0 }
        set { }
    }

    public var lastGeneratedCodes: [MLXArray]? { nil }

    /// Default: ignores the voice anchor flag and delegates to the base overload.
    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters,
        useVoiceAnchor: Bool
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(
            text: text, voice: voice, refAudio: refAudio,
            refText: refText, language: language,
            generationParameters: generationParameters
        )
    }

    public func buildVoiceAnchor(referenceCount: Int, instruct: String?, language: String?) {}
    public func clearVoiceAnchor() {}
    public func cancelGeneration() {}
    public func tokenizeForAlignment(text: String) -> [Int] { [] }
}
