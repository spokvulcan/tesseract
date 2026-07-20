//
//  AgentLogitProcessors.swift
//  tesseract
//
//  Output-only presence penalty, and the one factory the app's generation
//  paths use instead of the vendor's `GenerateParameters.processor()`.
//
//  Why: Qwen's recommended `presence_penalty = 1.5` assumes vLLM/OpenAI
//  semantics — an additive penalty on every token the model has *generated*
//  this request, over the whole output. The vendor's `PresencePenaltyContext`
//  is a different beast: a 20-token sliding window seeded with the prompt
//  tail. Against the observed failure mode (a ~250-token turn replayed
//  verbatim) a 20-token window sees nothing, so the recommended setting was
//  silently inert. And the naive fix — raising the window size — would be
//  worse than the disease: `TokenRing.loadPrompt` keeps the trailing window of
//  the *prompt*, so a large window puts the system prompt and history under a
//  −1.5 blanket.
//
//  `OutputPresencePenalty` restores the intended semantics: `prompt()` is a
//  no-op, only sampled tokens enter the ring, and the ring spans the whole
//  realistic generation. Repetition and frequency penalties keep their vendor
//  (mlx-lm-matching) semantics untouched.
//

import Foundation
import MLX
import MLXLMCommon

/// Additive presence penalty over the tokens generated *this request* —
/// vLLM/OpenAI semantics. Prompt tokens are never penalized.
///
/// Mirrors the vendor penalty processors' GPU-only discipline: the ring is an
/// `MLXArray` updated by mask writes, so `didSample` never forces a CPU←GPU
/// sync and `asyncEval` pipelining survives.
nonisolated struct OutputPresencePenalty: LogitProcessor {

    /// "Whole generation" in practice: no agent or server response
    /// legitimately exceeds this many generated tokens, and past the cap the
    /// ring wraps (oldest tokens fall out) rather than growing unbounded.
    static let defaultCapacity = 32_768

    private let penalty: Float
    private let capacity: Int
    private var buffer: MLXArray
    private var positions: MLXArray
    private var sampledCount = 0
    private var writeIndex = 0

    init(penalty: Float, capacity: Int = OutputPresencePenalty.defaultCapacity) {
        precondition(capacity > 0)
        self.penalty = penalty
        self.capacity = capacity
        self.buffer = MLXArray.zeros([capacity], type: Int32.self)
        self.positions = MLXArray.arange(capacity)
    }

    /// Output-only: the prompt never enters the ring.
    mutating func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        guard sampledCount > 0 else { return logits }
        let valid = sampledCount < capacity ? buffer[..<sampledCount] : buffer
        let broadcastIndices = valid.asType(.uint32)[.newAxis, 0...]
        // Scatter-write of `value − penalty` per unique index: writing the
        // same value to the same index twice is idempotent, so a token is
        // penalized once regardless of how often it was sampled.
        let selectedLogits = takeAlong(logits, broadcastIndices, axis: -1) - penalty
        return putAlong(logits, broadcastIndices, values: selectedLogits, axis: -1)
    }

    mutating func didSample(token: MLXArray) {
        let mask = positions .== Int32(writeIndex)
        buffer = MLX.where(mask, token.asType(.int32), buffer)
        writeIndex = (writeIndex + 1) % capacity
        sampledCount = min(sampledCount + 1, capacity)
    }
}

/// Chains logit processors: `process` folds left-to-right, lifecycle calls
/// fan out to every element.
nonisolated struct CompositeLogitProcessor: LogitProcessor {
    private var processors: [any LogitProcessor]

    init(_ processors: [any LogitProcessor]) {
        self.processors = processors
    }

    mutating func prompt(_ prompt: MLXArray) {
        for index in processors.indices {
            processors[index].prompt(prompt)
        }
    }

    func process(logits: MLXArray) -> MLXArray {
        processors.reduce(logits) { $1.process(logits: $0) }
    }

    mutating func didSample(token: MLXArray) {
        for index in processors.indices {
            processors[index].didSample(token: token)
        }
    }
}

/// The app's replacement for `GenerateParameters.processor()`: identical for
/// repetition/frequency penalties, but presence is built as
/// ``OutputPresencePenalty`` (output-only, whole-generation) instead of the
/// vendor's prompt-seeded 20-token window. `presenceContextSize` is
/// deliberately ignored — it parameterizes the window semantics this factory
/// exists to replace.
nonisolated enum AgentLogitProcessors {
    static func processor(for parameters: GenerateParameters) -> (any LogitProcessor)? {
        var presence: OutputPresencePenalty?
        if let presencePenalty = parameters.presencePenalty, presencePenalty != 0 {
            presence = OutputPresencePenalty(penalty: presencePenalty)
        }

        var stripped = parameters
        stripped.presencePenalty = nil
        let vendorPenalties = stripped.processor()

        switch (vendorPenalties, presence) {
        case (nil, nil): return nil
        case (let vendor?, nil): return vendor
        case (nil, let presence?): return presence
        case (let vendor?, let presence?):
            return CompositeLogitProcessor([vendor, presence])
        }
    }
}
