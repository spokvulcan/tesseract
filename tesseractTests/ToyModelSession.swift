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
    /// Test hook, fired at the top of every forward with the pre-update
    /// cache offset — the cancellation suites use it to pause the prefill at
    /// a deterministic point and land a cancel at a chunk boundary.
    let onForward: (@Sendable (Int) -> Void)?
    /// Content-relative scripting (the Emitted Path suites): when set, the
    /// queue — not `script` — answers what the toy predicts, keyed on the
    /// tokens actually fed rather than on absolute positions.
    let completions: ToyCompletionQueue?

    init(
        script: [Int],
        eosTokenId: Int = ToyVocabulary.eosTokenId,
        vocabSize: Int = ToyVocabulary.size,
        layers: Int = 2,
        headDim: Int = 4,
        onForward: (@Sendable (Int) -> Void)? = nil
    ) {
        self.onForward = onForward
        precondition(
            script.allSatisfy { $0 >= 0 && $0 < vocabSize },
            "script tokens must fit the toy vocabulary"
        )
        self.script = script
        self.completions = nil
        self.eosTokenId = eosTokenId
        self.vocabSize = vocabSize
        self.kvHeads = Array(repeating: 1, count: layers)
        self.headDim = headDim
        super.init()
    }

    /// The content-relative toy: every forward records what it was fed, and
    /// the prediction at a forward's last row comes from `completions` —
    /// the next queued completion once the fed tokens end with the
    /// generation prompt, then that completion token by token as long as
    /// the loop feeds back exactly what was predicted.
    init(
        completions: ToyCompletionQueue,
        vocabSize: Int = ToyVocabulary.size,
        layers: Int = 2,
        headDim: Int = 4,
        onForward: (@Sendable (Int) -> Void)? = nil
    ) {
        self.onForward = onForward
        self.script = []
        self.completions = completions
        self.eosTokenId = completions.eosTokenId
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
    /// The toy keeps no positional state, so `state` is ignored.
    func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill _: PrefillParameters
    ) throws -> PrepareResult {
        let tokens = input.text.tokens
        let batched = tokens.ndim >= 2 ? tokens : tokens[.newAxis]
        return .logits(LMOutput(logits: callAsFunction(batched, cache: cache)))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let batched = inputs.ndim >= 2 ? inputs : inputs[.newAxis]
        let tokenCount = batched.dim(-1)
        let offset = cache?.first?.offset ?? 0
        onForward?(offset)

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
        if let completions {
            let fed = batched.asType(.int32).reshaped([tokenCount]).asArray(Int32.self).map(
                Int.init)
            completions.record(fed: fed, at: offset)
            // Only the last row's logits are ever sampled; the rest predict
            // EOS so a stray sample can never consume a queued completion.
            for row in 0..<tokenCount {
                let predicted =
                    row == tokenCount - 1
                    ? completions.predict(at: offset + row) : completions.eosTokenId
                precondition(
                    predicted < vocabSize, "completion token \(predicted) outside the vocabulary")
                rows[row * vocabSize + predicted] = 10
            }
        } else {
            for row in 0..<tokenCount {
                rows[row * vocabSize + predictedToken(at: offset + row)] = 10
            }
        }
        return MLXArray(rows, [1, tokenCount, vocabSize])
    }
}

/// The content-relative script behind `ToyLanguageModel(completions:)`: a
/// queue of completions, each served once the tokens fed so far end with
/// the generation prompt — at whatever absolute position the request's
/// restore and prefill put it — then continued token by token while the
/// loop feeds back exactly the token predicted (a decode step), and closed
/// with `eosTokenId`. Anything else predicts EOS.
///
/// Also the tape of everything the model was fed, in feed order, which the
/// Emitted Path suites read to prove what a request prefilled: the served
/// composition on a hit, the canonical encode on a miss, never a token
/// outside the vocabulary.
nonisolated final class ToyCompletionQueue: @unchecked Sendable {
    /// The tokens the template's generation prompt encodes to; the
    /// trigger is the fed tokens ending with them.
    let generationPrompts: [[Int]]
    let eosTokenId: Int

    private let lock = NSLock()
    private var queue: [[Int]] = []
    private var tape: [Int: Int] = [:]
    private var feeds: [(position: Int, id: Int)] = []
    private var active: [Int] = []
    private var activeIndex = 0
    private var triggerPosition: Int?
    private var expectedNext: (position: Int, id: Int)?

    /// `generationPrompts`: every generation prompt the template can emit
    /// (thinking and non-thinking), each as its token ids.
    init(generationPrompts: [[Int]], eosTokenId: Int) {
        self.generationPrompts = generationPrompts
        self.eosTokenId = eosTokenId
    }

    /// Queue the next completion, without its EOS (the queue appends it).
    func enqueue(_ completion: [Int]) {
        lock.withLock { queue.append(completion) }
    }

    /// Everything fed since the last drain, in feed order.
    func drainFeeds() -> [(position: Int, id: Int)] {
        lock.withLock {
            let drained = feeds
            feeds.removeAll()
            return drained
        }
    }

    func record(fed: [Int], at offset: Int) {
        lock.withLock {
            for (row, id) in fed.enumerated() {
                tape[offset + row] = id
                feeds.append((offset + row, id))
            }
        }
    }

    func predict(at position: Int) -> Int {
        lock.withLock {
            if endsWithGenerationPrompt(at: position) {
                // The same position forwarded twice (a prefill chunk and a
                // prime forward both ending there) re-serves the same
                // first token rather than consuming another completion.
                if triggerPosition == position, activeIndex == 1, !active.isEmpty {
                    return active[0]
                }
                guard !queue.isEmpty else { return eosTokenId }
                active = queue.removeFirst() + [eosTokenId]
                activeIndex = 0
                triggerPosition = position
                return advance(at: position)
            }
            if let expectedNext, expectedNext.position == position,
                tape[position] == expectedNext.id, activeIndex < active.count
            {
                return advance(at: position)
            }
            return eosTokenId
        }
    }

    private func advance(at position: Int) -> Int {
        let token = active[activeIndex]
        activeIndex += 1
        expectedNext = (position + 1, token)
        return token
    }

    private func endsWithGenerationPrompt(at position: Int) -> Bool {
        generationPrompts.contains { prompt in
            guard !prompt.isEmpty, position + 1 >= prompt.count else { return false }
            return prompt.enumerated().allSatisfy { index, id in
                tape[position - prompt.count + 1 + index] == id
            }
        }
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

/// Toy `UserInputProcessor`: renders messages through the given tokenizer's
/// chat template and returns 1D prepared tokens — the pure-LLM prepare shape.
///
/// With a `VisionStub` installed it becomes the 2D-token toy variant (PRD
/// #137, user story 12): image-bearing input appends one placeholder pad run
/// per image and returns a `ProcessedImage` whose frames carry the stub's
/// grid — the prepared shape the **Cache Key Space** and the ADR-0014 patch
/// guard price, with no vision tower behind it.
nonisolated struct ToyUserInputProcessor: UserInputProcessor {
    /// The image-keying facts the stub fabricates per attached image.
    struct VisionStub {
        let padTokenId: Int
        let padRunLength: Int
        let frame: THW
        /// When set, the template already rendered each image's placeholder
        /// run in place (the Qwen-VL shape, where text follows the image);
        /// the stub then supplies only the frames. Off, the run is appended
        /// after the render.
        var inlineRuns = false
    }

    let tokenizer: any Tokenizer
    var vision: VisionStub?

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
        var tokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext
        )
        guard let vision, !input.images.isEmpty else {
            return LMInput(tokens: MLXArray(tokens.map(Int32.init)))
        }
        var frames: [THW] = []
        for _ in input.images {
            if !vision.inlineRuns {
                tokens += Array(repeating: vision.padTokenId, count: vision.padRunLength)
            }
            frames.append(vision.frame)
        }
        // Image-bearing prepares emit the VLM 2D `[batch, seq]` token shape —
        // the keyed arm's image-span slicing indexes both axes.
        return LMInput(
            text: .init(tokens: MLXArray(tokens.map(Int32.init))[.newAxis], mask: nil),
            image: LMInput.ProcessedImage(pixels: MLXArray.zeros([4, 3]), frames: frames)
        )
    }
}

/// One-shot gate for the toy model's forward hook: pauses the model thread
/// the first time a forward starts at or past `threshold`, so a test can
/// land a deterministic cancel at a chunk boundary, then releases it.
/// Construct with `armed: false` and call `arm()` when a setup phase must
/// run over the same offsets without tripping the gate.
nonisolated final class ForwardGate: @unchecked Sendable {
    private let lock = NSLock()
    private var armed: Bool
    private let threshold: Int
    private let release = DispatchSemaphore(value: 0)
    private let reachedStream: AsyncStream<Void>
    private let reachedContinuation: AsyncStream<Void>.Continuation

    init(threshold: Int, armed: Bool = true) {
        self.threshold = threshold
        self.armed = armed
        (reachedStream, reachedContinuation) = AsyncStream.makeStream()
    }

    func arm() {
        lock.withLock { armed = true }
    }

    func onForward(offset: Int) {
        let shouldBlock = lock.withLock {
            guard armed, offset >= threshold else { return false }
            armed = false
            return true
        }
        guard shouldBlock else { return }
        reachedContinuation.yield()
        release.wait()
    }

    /// Awaits the gate being reached (async-safe for the test's isolation).
    func reached() async {
        for await _ in reachedStream { break }
    }

    func open() {
        release.signal()
    }
}

/// Records every toy-model forward's pre-update cache offset, in order —
/// the observable difference between the chunked and single-shot prefill
/// routes, and the "did it allocate at all" fact the vision-guard ordering
/// suites assert. Pass `onForward` as the toy's hook.
nonisolated final class ForwardLog: @unchecked Sendable {
    private let lock = NSLock()
    private var _offsets: [Int] = []

    var offsets: [Int] { lock.withLock { _offsets } }
    var hasForwarded: Bool { !offsets.isEmpty }

    func onForward(_ offset: Int) {
        lock.withLock { _offsets.append(offset) }
    }
}

/// Collects `ServerInferenceProgressEvent`s across isolations: the handler
/// fires on the MainActor, assertions read after the drive settles.
nonisolated final class ProgressEventLog: @unchecked Sendable {
    private let lock = NSLock()
    private var _events: [ServerInferenceProgressEvent] = []

    var events: [ServerInferenceProgressEvent] {
        lock.withLock { _events }
    }

    func append(_ event: ServerInferenceProgressEvent) {
        lock.withLock { _events.append(event) }
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
    case makeRawDecodeIterator
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
    /// Forces the LLM-class `producesFlatTextTokens` answer over the toy
    /// context — the shape of a vision-family checkpoint the VLM factory
    /// rejected, silently loaded as a text-only instance. No toy model class
    /// can express it directly: `ContextBackedModelSession` derives the fact
    /// from `model is any LLMModel`, and the toy is neither marker.
    var producesFlatTextTokensOverride: Bool?
    /// The toy's own anchored vision `prepare` (the vision-container
    /// feature the keyed image path requires), when the provider anchors
    /// vision: the toy forward over the image span's tokens, no tower.
    var anchoredVisionPrepareOverride: AnchoredVisionPrepare?

    var configuration: ModelConfiguration { base.configuration }
    var tokenizer: any Tokenizer { base.tokenizer }
    var mtpDrafter: (any MTPDrafterModel)? { base.mtpDrafter }
    var anchoredVisionPrepare: AnchoredVisionPrepare? {
        recorder.record(.visionContinuationQuery)
        return anchoredVisionPrepareOverride ?? base.anchoredVisionPrepare
    }
    var producesFlatTextTokens: Bool {
        producesFlatTextTokensOverride ?? base.producesFlatTextTokens
    }

    func prepare(_ input: UserInput) async throws -> LMInput {
        recorder.record(.prepare)
        return try await base.prepare(input)
    }

    func templateMessages(for input: UserInput) -> [Message]? {
        base.templateMessages(for: input)
    }

    func makeRawDecodeIterator(
        _ input: LMInput,
        parameters: GenerateParameters
    ) throws -> TokenIterator {
        recorder.record(.makeRawDecodeIterator)
        return try base.makeRawDecodeIterator(input, parameters: parameters)
    }

    func newCache(parameters: GenerateParameters) throws -> [any KVCache] {
        recorder.record(.newCache)
        return try base.newCache(parameters: parameters)
    }

    func restore(_ snapshot: HybridCacheSnapshot) throws -> [any KVCache] {
        recorder.record(.restore)
        return try base.restore(snapshot)
    }

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
        prepare: ((LMInput, [any KVCache], Int?) throws -> PrepareResult)?
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
    /// When true, sessions report the LLM-class `producesFlatTextTokens` —
    /// the text-only-fallback shape the instance-truth keying suites model.
    let reportsFlatTextTokens: Bool
    /// When true, sessions expose the toy's anchored vision `prepare`, so
    /// a keyed image-bearing request runs end to end over the toy (the
    /// vision-container shape; the stub's pad run stands in for the tower).
    let anchorsVision: Bool
    let hasMTPDrafter: Bool

    init(
        model: ToyLanguageModel,
        tokenizer: any Tokenizer = FakeChatMLTokenizer(),
        configuration: ModelConfiguration = ToyVocabulary.configuration(),
        vision: ToyUserInputProcessor.VisionStub? = nil,
        reportsFlatTextTokens: Bool = false,
        anchorsVision: Bool = false,
        hasMTPDrafter: Bool = false
    ) {
        self.reportsFlatTextTokens = reportsFlatTextTokens
        self.anchorsVision = anchorsVision
        self.hasMTPDrafter = hasMTPDrafter
        self.container = ModelContainer(
            context: ModelContext(
                configuration: configuration,
                model: model,
                processor: ToyUserInputProcessor(tokenizer: tokenizer, vision: vision),
                tokenizer: tokenizer
            )
        )
    }

    func withSession<V, R: Sendable>(
        nonSendable payload: sending V,
        _ body: @Sendable (any ModelSession, V) async throws -> R
    ) async throws -> R {
        let recorder = self.recorder
        let reportsFlatTextTokens = self.reportsFlatTextTokens
        let anchorsVision = self.anchorsVision
        let hasMTPDrafter = self.hasMTPDrafter
        return try await container.perform(nonSendable: payload) { context, payload in
            return try await body(
                RecordingModelSession(
                    base: ContextBackedModelSession(
                        context: context, mtpDrafter: hasMTPDrafter ? InactiveMTPDrafter() : nil),
                    recorder: recorder,
                    producesFlatTextTokensOverride: reportsFlatTextTokens ? true : nil,
                    anchoredVisionPrepareOverride: anchorsVision
                        ? Self.toyAnchoredVisionPrepare(context) : nil
                ),
                payload
            )
        }
    }

    /// The toy's anchored `prepare`: the same single-shot forward its
    /// `prepare` runs, over the image span's already-expanded tokens.
    private static func toyAnchoredVisionPrepare(_ context: ModelContext) -> AnchoredVisionPrepare?
    {
        guard let model = context.model as? ToyLanguageModel else { return nil }
        return { input, cache, state, windowSize in
            try model.prepare(
                input, cache: cache, state: state, prefill: .init(stepSize: windowSize))
        }
    }
}

/// Presence-only drafter for requests whose policy must select ordinary
/// decoding. Any attempt to use the MTP path fails at the model boundary.
nonisolated final class InactiveMTPDrafter: Module, MTPDrafterModel {
    func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray {
        preconditionFailure("MTP must not engage in this fixture")
    }
}
