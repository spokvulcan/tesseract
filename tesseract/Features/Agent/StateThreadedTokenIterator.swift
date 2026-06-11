import Foundation
import MLX
import MLXLMCommon

/// The cache-aware path's decode iterator: threads one `LMOutput.State` from
/// prefill through the prime forward and every decode step.
///
/// Upstream's `TokenIterator` discards `prepare()`'s returned state (its
/// `.logits` branch never stores `result.state`), so on the Qwen3.5 vision
/// container the prime forward and the first decode step recompute M-RoPE
/// positions from zero — the decode mis-positioning the VLM spike measured
/// (`iteratorStateDropDiverges`, ADR-0007). Cache-aware generation therefore
/// owns decode (PRD #72): the same step/sample shape as upstream, with the
/// state seam upstream lacks.
///
/// **Metal-affinity contract:** construct and consume inside
/// `ModelContainer.perform` / the generation task it hands off to.
nonisolated struct StateThreadedTokenIterator: TokenIteratorProtocol {
    private let model: any LanguageModel
    private var cache: [any KVCache]
    private var state: LMOutput.State?
    private var processor: (any LogitProcessor)?
    private let sampler: any LogitSampler
    private var y: LMInput.Text

    private(set) var tokenCount = 0
    let maxTokens: Int?
    private(set) var promptPrefillTime: TimeInterval = 0

    /// Post-prefill form: the cache already covers everything but `remainder`
    /// (the final prompt token `PrefillExecutor.run` keeps back), and `state`
    /// is the model state the last prefill chunk returned (or a seeded
    /// **Position Anchor** on a restored cache). The prime forward runs here.
    ///
    /// Penalty processors are seeded with `fullText` — the iterator's own
    /// input is a single token, which would otherwise be the entire
    /// repetition/presence/frequency context (same contract as
    /// `PrefillExecutor.makeIterator`).
    init(
        remainder: LMInput.Text,
        fullText: LMInput.Text,
        model: any LanguageModel,
        cache: [any KVCache],
        state: LMOutput.State?,
        parameters: GenerateParameters
    ) {
        self.model = model
        self.cache = cache
        self.state = state
        self.processor = parameters.processor()
        self.sampler = parameters.sampler()
        self.maxTokens = parameters.maxTokens
        self.y = remainder

        processor?.prompt(fullText.tokens)
        let token = step(previous: y)
        y = .init(tokens: token)
        asyncEval(y.tokens)
    }

    /// Whole-prompt form for the **Unkeyed Completion**: runs the vendor
    /// `prepare` like upstream's `TokenIterator.init`, but keeps the state a
    /// `.logits` prepare returns instead of dropping it — so decode resumes
    /// the prepare's positions instead of restarting at zero.
    init(
        preparing input: LMInput,
        model: any LanguageModel,
        cache: [any KVCache],
        parameters: GenerateParameters
    ) throws {
        self.model = model
        self.cache = cache
        self.processor = parameters.processor()
        self.sampler = parameters.sampler()
        self.maxTokens = parameters.maxTokens
        self.y = input.text

        processor?.prompt(input.text.tokens)
        let started = Date.timeIntervalSinceReferenceDate
        switch try model.prepare(input, cache: cache, windowSize: parameters.prefillStepSize) {
        case .tokens(let tail):
            y = tail
            let token = step(previous: y)
            y = .init(tokens: token)
            asyncEval(y.tokens)
        case .logits(let result):
            state = result.state
            y = .init(tokens: convertToToken(logits: result.logits))
            asyncEval(y.tokens)
        }
        promptPrefillTime = Date.timeIntervalSinceReferenceDate - started
    }

    private mutating func convertToToken(logits: MLXArray) -> MLXArray {
        var logits = logits[0..., -1, 0...]
        logits = processor?.process(logits: logits) ?? logits
        let token = sampler.sample(logits: logits)
        processor?.didSample(token: token)
        return token
    }

    /// One forward with the threaded state — mirrors upstream's `step`, minus
    /// the per-step cache quantization (the cache-aware path quantizes once
    /// before the iterator so the array it retains stays the live final cache).
    private mutating func step(previous: LMInput.Text) -> MLXArray {
        let input = previous.tokens.ndim >= 2 ? previous : previous[text: .newAxis]
        let result = model(input, cache: cache.isEmpty ? nil : cache, state: state)
        state = result.state
        return convertToToken(logits: result.logits)
    }

    mutating func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }
        let previousY = y
        let token = step(previous: previousY)
        y = .init(tokens: token)
        asyncEval(token)
        tokenCount += 1
        return previousY.tokens.item(Int.self)
    }
}
