import Foundation
import MLX
import MLXLMCommon
import MLXNN

@testable import Tesseract_Agent

/// The **Model Session** test peer (ADR-0016): a scripted toy `LanguageModel`
/// the sequencing suites run the module's *real* verb implementations over —
/// genuine `newCache`, the real `PrefillExecutor`, a real
/// `StateThreadedTokenIterator` whose init runs its genuine prime forward —
/// on microscopic tensors. Only the model is substituted across the seam.
///
/// Semantics: the toy "believes in" one token sequence, `script` (prompt +
/// completion). The predicted token for absolute position `p` is
/// `script[p + 1]`; past the script's end it predicts `eosTokenId`. Logits
/// are one-hot, so with `temperature: 0` (argmax sampling) decode reproduces
/// the scripted completion deterministically — sequencing assertions never
/// flake on sampling.
///
/// Each forward derives K/V content from the input tokens and pushes it
/// through `KVCache.update`, so cache offsets advance exactly as a real
/// model's would and capture/restore round-trips carry content-dependent
/// payloads.
nonisolated final class ToyLanguageModel: Module, LanguageModel, KVCacheDimensionProvider {
    let kvHeads: [Int]
    let headDim: Int
    let vocabSize: Int
    let script: [Int]
    let eosTokenId: Int

    init(
        script: [Int],
        eosTokenId: Int = ToyVocabulary.eosTokenId,
        vocabSize: Int = ToyVocabulary.size,
        layers: Int = 2,
        headDim: Int = 4
    ) {
        precondition(
            script.allSatisfy { $0 >= 0 && $0 < vocabSize },
            "script tokens must fit the toy vocabulary"
        )
        self.script = script
        self.eosTokenId = eosTokenId
        self.vocabSize = vocabSize
        self.kvHeads = Array(repeating: 1, count: layers)
        self.headDim = headDim
        super.init()
    }

    func predictedToken(at position: Int) -> Int {
        let next = position + 1
        return next < script.count ? script[next] : eosTokenId
    }

    /// Single-shot prepare, the vendor-LLM shape: forward the whole prompt,
    /// return the `.logits` the decode iterator samples its first token from.
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        let tokens = input.text.tokens
        let batched = tokens.ndim >= 2 ? tokens : tokens[.newAxis]
        return .logits(LMOutput(logits: callAsFunction(batched, cache: cache)))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let batched = inputs.ndim >= 2 ? inputs : inputs[.newAxis]
        let tokenCount = batched.dim(-1)
        let offset = cache?.first?.offset ?? 0

        if let cache {
            // Content-derived K/V — one head, `headDim` copies of the token
            // value — so snapshots capture real, position-dependent payloads.
            let content = batched.asType(.float32).reshaped([1, 1, tokenCount, 1])
            let keysValues = broadcast(content, to: [1, 1, tokenCount, headDim])
            for layer in cache {
                _ = layer.update(keys: keysValues, values: keysValues)
            }
        }

        var rows = [Float](repeating: 0, count: tokenCount * vocabSize)
        for row in 0..<tokenCount {
            rows[row * vocabSize + predictedToken(at: offset + row)] = 10
        }
        return MLXArray(rows, [1, tokenCount, vocabSize])
    }
}

/// Shared constants for the toy vocabulary: the byte-level
/// `FakeChatMLTokenizer` occupies 0–255, so the EOS sentinel sits above it.
nonisolated enum ToyVocabulary {
    static let size = 512
    static let eosTokenId = 300

    /// A `ModelConfiguration` whose stop-token set matches the toy model.
    static func configuration(name: String = "toy/model") -> ModelConfiguration {
        var configuration = ModelConfiguration(id: name)
        configuration.eosTokenIds = [eosTokenId]
        return configuration
    }
}

/// Toy `UserInputProcessor`: renders messages through the byte-level ChatML
/// template and returns 1D prepared tokens — the pure-LLM prepare shape.
nonisolated struct ToyUserInputProcessor: UserInputProcessor {
    let tokenizer: FakeChatMLTokenizer

    func prepare(input: UserInput) async throws -> LMInput {
        let messages: [Message]
        switch input.prompt {
        case .messages(let value):
            messages = value
        case .text(let text):
            messages = [["role": "user", "content": text]]
        case .chat(let chat):
            messages = chat.map { ["role": "\($0.role)", "content": $0.content] }
        }
        let tokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext
        )
        return LMInput(tokens: MLXArray(tokens.map(Int32.init)))
    }
}

/// The verbs a **Model Session** exposes, as recordable facts — the
/// sequencing suites assert their order (the seam's contract).
nonisolated enum ModelVerb: String, Equatable, Sendable {
    case prepare
    case newCache
    case restore
    case prefill
    case makeDecodeIterator
    case makePreparingDecodeIterator
    case quantizeKVCache
    case captureSnapshot
    case visionContinuationQuery
}

/// Thread-safe verb log: verbs land from the session's isolation, assertions
/// read from the test's.
nonisolated final class ModelVerbRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var _verbs: [ModelVerb] = []

    var verbs: [ModelVerb] {
        lock.withLock { _verbs }
    }

    func record(_ verb: ModelVerb) {
        lock.withLock { _verbs.append(verb) }
    }
}

/// Decorator over the real verb implementations: records each verb, then
/// forwards to `ContextBackedModelSession` — nothing is reimplemented.
nonisolated struct RecordingModelSession: ModelSession {
    let base: any ModelSession
    let recorder: ModelVerbRecorder

    var configuration: ModelConfiguration { base.configuration }
    var tokenizer: any Tokenizer { base.tokenizer }
    var windowedVisionContinuation: (any WindowedVisionContinuation)? {
        recorder.record(.visionContinuationQuery)
        return base.windowedVisionContinuation
    }

    func prepare(_ input: UserInput) async throws -> LMInput {
        recorder.record(.prepare)
        return try await base.prepare(input)
    }

    func newCache(parameters: GenerateParameters) -> [any KVCache] {
        recorder.record(.newCache)
        return base.newCache(parameters: parameters)
    }

    func restore(_ snapshot: HybridCacheSnapshot) throws -> [any KVCache] {
        recorder.record(.restore)
        return try base.restore(snapshot)
    }

    // swiftlint:disable:next function_parameter_count
    func prefill(
        text: LMInput.Text,
        cache: [any KVCache],
        checkpoints: [Int: HybridCacheSnapshot.CheckpointType],
        checkpointBaseOffset: Int,
        prefillStepSize: Int,
        consumeAll: Bool,
        initialState: LMOutput.State?,
        evalPolicy: PrefillExecutor.EvalPolicy
    ) throws -> PrefillExecutor.Output {
        recorder.record(.prefill)
        return try base.prefill(
            text: text,
            cache: cache,
            checkpoints: checkpoints,
            checkpointBaseOffset: checkpointBaseOffset,
            prefillStepSize: prefillStepSize,
            consumeAll: consumeAll,
            initialState: initialState,
            evalPolicy: evalPolicy
        )
    }

    func makeDecodeIterator(
        remainder: LMInput.Text,
        fullText: LMInput.Text,
        cache: [any KVCache],
        state: LMOutput.State?,
        parameters: GenerateParameters
    ) -> StateThreadedTokenIterator {
        recorder.record(.makeDecodeIterator)
        return base.makeDecodeIterator(
            remainder: remainder,
            fullText: fullText,
            cache: cache,
            state: state,
            parameters: parameters
        )
    }

    func makePreparingDecodeIterator(
        _ input: LMInput,
        cache: [any KVCache],
        parameters: GenerateParameters,
        prepare: ((LMInput, [any KVCache], Int) throws -> PrepareResult)?
    ) throws -> StateThreadedTokenIterator {
        recorder.record(.makePreparingDecodeIterator)
        return try base.makePreparingDecodeIterator(
            input,
            cache: cache,
            parameters: parameters,
            prepare: prepare
        )
    }

    func quantizeKVCache(_ cache: inout [any KVCache], parameters: GenerateParameters) {
        recorder.record(.quantizeKVCache)
        base.quantizeKVCache(&cache, parameters: parameters)
    }

    func captureSnapshot(
        cache: [any KVCache],
        offset: Int,
        type: HybridCacheSnapshot.CheckpointType
    ) -> HybridCacheSnapshot? {
        recorder.record(.captureSnapshot)
        return base.captureSnapshot(cache: cache, offset: offset, type: type)
    }
}

/// The seam's second adapter (ADR-0016): a `ModelSessionProviding` over a
/// real `ModelContainer` wrapping the toy context — the identical
/// serial-access execution shape production uses — with every session
/// decorated by the verb recorder.
nonisolated struct ToyModelSessionProvider: ModelSessionProviding {
    let container: ModelContainer
    let recorder = ModelVerbRecorder()

    init(
        model: ToyLanguageModel,
        tokenizer: FakeChatMLTokenizer = FakeChatMLTokenizer(),
        configuration: ModelConfiguration = ToyVocabulary.configuration()
    ) {
        self.container = ModelContainer(
            context: ModelContext(
                configuration: configuration,
                model: model,
                processor: ToyUserInputProcessor(tokenizer: tokenizer),
                tokenizer: tokenizer
            )
        )
    }

    func withSession<R: Sendable>(
        _ body: @Sendable (any ModelSession) async throws -> R
    ) async throws -> R {
        let recorder = self.recorder
        return try await container.perform { context in
            try await body(
                RecordingModelSession(
                    base: ContextBackedModelSession(context: context),
                    recorder: recorder
                )
            )
        }
    }
}
