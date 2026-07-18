import Foundation
import MLX
import MLXLMCommon

/// The **Prefill Strategy** (ADR-0044): the chunked-vs-single-shot route for
/// one raw-generation prompt, decided once from the prompt's shape.
///
/// VLM-class models (2D token tensors) run upstream `prepare` as a single
/// forward pass over the whole prompt, so long text-only prompts chunk
/// through the app's prefill driver to keep peak memory bounded (ADR-0006).
/// Media-bearing prompts stay single-shot — the model's own `prepare` places
/// their tokens. 1D (LLM-class) prompts also stay single-shot: upstream's
/// `TokenIterator` chunks those internally.
///
/// This rule used to live as three hand-written guards (the agent chat arm,
/// the thinking-continuation arm, the parity bench) that had already drifted
/// apart; `decide` is now its one home.
nonisolated enum PrefillStrategy: Equatable, Sendable {

    /// Drive the prompt through the app's chunked-prefill driver
    /// (``PrefillExecutor``) in steps of `stepSize`, then decode from the
    /// warmed cache.
    case chunked(stepSize: Int)

    /// Hand the prompt to `TokenIterator` in one piece.
    case singleShot

    /// Step size used when the generation parameters don't carry one.
    static let fallbackStepSize = 512

    /// The route, from the prompt's shape facts alone.
    ///
    /// - Parameters:
    ///   - tokenNDim: dimensionality of the token tensor (1 = LLM-class,
    ///     2 = VLM-class conditional generation).
    ///   - sequenceLength: last-axis token count of the prompt
    ///     (`tokens.dim(-1)`).
    ///   - prefillStepSize: the parameters' step size; `nil` falls back to
    ///     ``fallbackStepSize``.
    static func decide(
        tokenNDim: Int,
        sequenceLength: Int,
        hasImage: Bool,
        hasVideo: Bool,
        hasAudio: Bool,
        prefillStepSize: Int?
    ) -> PrefillStrategy {
        let stepSize = prefillStepSize ?? fallbackStepSize
        guard tokenNDim >= 2,
            !hasImage, !hasVideo, !hasAudio,
            sequenceLength > stepSize
        else {
            return .singleShot
        }
        return .chunked(stepSize: stepSize)
    }

    /// The route for a prepared input — reads the shape facts off the
    /// `LMInput` and delegates to ``decide(tokenNDim:sequenceLength:hasImage:hasVideo:hasAudio:prefillStepSize:)``.
    static func decide(for input: LMInput, prefillStepSize: Int?) -> PrefillStrategy {
        decide(
            tokenNDim: input.text.tokens.ndim,
            sequenceLength: input.text.tokens.dim(-1),
            hasImage: input.image != nil,
            hasVideo: input.video != nil,
            hasAudio: input.audio != nil,
            prefillStepSize: prefillStepSize
        )
    }

    /// Execute the route: build the post-decision `TokenIterator`, warming
    /// the cache through ``PrefillExecutor`` on the chunked arm.
    ///
    /// **Metal-affinity contract:** must run inside `ModelContainer.perform`.
    func makeIterator(
        input: LMInput,
        model: any LanguageModel,
        parameters: GenerateParameters
    ) throws -> TokenIterator {
        switch self {
        case .chunked(let stepSize):
            var cache = model.newCache(parameters: parameters)
            let warmed = try PrefillExecutor.run(
                model: model,
                text: input.text,
                cache: cache,
                prefillStepSize: stepSize
            )
            return try PrefillExecutor.makeIterator(
                model: model,
                fullText: input.text,
                remainder: warmed.remainder,
                cache: &cache,
                parameters: parameters
            )
        case .singleShot:
            return try TokenIterator(
                input: input,
                model: model,
                cache: nil,
                parameters: parameters
            )
        }
    }
}
