import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// The **Decode Start** (ADR-0093) through its interface, over the toy Model
/// Session: one table of Decode Handover × KV Scheme × penalties. Each row
/// warms the cache the way its generation path does, makes the start, reports
/// the prefill, runs the loop to the end and reads what came out: the verbs
/// the handover called, the text, which cache form the prompt's last token and
/// the first decode step ran over, the badge, and the array the owner keeps.
///
/// The toy predicts by position with one-hot logits of 10, so a penalty above
/// 10 replaces a token outright: the text shows which penalties applied and
/// what they were seeded with. The prompt holds an `x`, and the scripted
/// completion is `xyx`:
/// - no penalties: `xyx`;
/// - the app's output-only presence penalty: the generated `x` repeats, so
///   the third token is replaced (argmax of the zeros, token 0): `xy␀`;
/// - a frequency penalty seeded with the whole prompt: the prompt's `x`
///   already counts, so the first `x` is replaced too: `␀y␁`. Seeded with
///   the iterator's own one-token remainder it would read `xy␀`.
nonisolated struct DecodeStartTests {

    enum Kind: String, CaseIterable, Sendable {
        /// The keyed turn's standard handover: the app prefilled all but the
        /// last prompt token.
        case standard
        /// The keyed DFlash2 turn: the app prefilled up to a split.
        case speculativeSplit
        /// A Speculation Plan's iterator over the whole prompt from zero.
        case speculativeWholePrompt
        /// The Unkeyed Completion: the app's iterator prefills.
        case preparing
        /// The Raw Generation Start without a plan, single-shot (flat tokens).
        case prefillStrategySingleShot
        /// The Raw Generation Start without a plan, chunked (batched tokens
        /// longer than one step).
        case prefillStrategyChunked

        var speculates: Bool { self == .speculativeSplit || self == .speculativeWholePrompt }
        var keepsCache: Bool {
            self != .prefillStrategySingleShot && self != .prefillStrategyChunked
        }
    }

    enum Penalty: String, CaseIterable, Sendable {
        case none, presence, frequency
    }

    struct Row: Sendable, CustomTestStringConvertible {
        let kind: Kind
        let scheme: KVScheme?
        let penalty: Penalty
        var testDescription: String {
            "\(kind.rawValue) × \(scheme?.rawValue ?? "fullPrecision") × \(penalty.rawValue)"
        }
    }

    /// Every handover × {no scheme, turbo8v4} × every penalty. The toy's
    /// DFlash2 verify moves only full-precision rows, so the speculative
    /// kinds run without a scheme here; the loaded-model gates and the
    /// vendor's `testDFlash2IteratorOverTurboQuantCache` cover the iterator
    /// over a scheme.
    static let rows: [Row] = Kind.allCases.flatMap { kind in
        ([nil] + (kind.speculates ? [] : [KVScheme.turbo8v4])).flatMap { scheme in
            Penalty.allCases.map { Row(kind: kind, scheme: scheme, penalty: $0) }
        }
    }

    static let prompt = Array("hello x world".utf8).map(Int.init)
    static let completion = Array("xyx".utf8).map(Int.init)
    /// Where the prompt's last token and the first decode step sit.
    static let lastPromptPosition = prompt.count - 1
    static let firstDecodePosition = prompt.count

    private static func expectedText(_ penalty: Penalty) -> String {
        let ids: [Int] =
            switch penalty {
            case .none: completion
            case .presence: [120, 121, 0]
            case .frequency: [0, 121, 1]
            }
        return String(decoding: ids.map { UInt8($0) }, as: UTF8.self)
    }

    /// Where the scheme converts, by handover: before the last prompt token
    /// when the app prefilled (and the DFlash2 iterator, before its final
    /// position), after it when an iterator ran the whole prefill. The chunked
    /// Prefill Strategy route converts up front only with penalties — the
    /// named row ADR-0093 keeps.
    private static func lastTokenConverted(_ row: Row) -> Bool {
        switch row.kind {
        case .standard, .speculativeSplit, .speculativeWholePrompt: true
        case .preparing, .prefillStrategySingleShot: false
        case .prefillStrategyChunked: row.penalty != .none
        }
    }

    private static func parameters(_ row: Row, kvBits: Int? = nil) -> GenerateParameters {
        var parameters = GenerateParameters(kvBits: kvBits, kvGroupSize: 64, temperature: 0)
        parameters.prefill = PrefillParameters(stepSize: 4)
        parameters.kvScheme = row.scheme?.rawValue
        switch row.penalty {
        case .none: break
        case .presence: parameters.presencePenalty = 15
        case .frequency:
            parameters.frequencyPenalty = 15
            parameters.frequencyContextSize = 20
        }
        return parameters
    }

    struct Outcome: Sendable {
        let stream: AsyncStream<RawGeneration>
        let task: Task<Void, Never>
        let owner: FinalGenerationCache?
        let verbs: [ModelVerb]
        let arm: SpeculativeArm?
    }

    /// Warm the cache as `row.kind`'s path does, make the Decode Start, report
    /// the prefill and start the loop, all in one session.
    private static func start(
        _ row: Row, provider: ToyModelSessionProvider, parameters: GenerateParameters,
        progress: ProgressEventLog
    ) async throws -> Outcome {
        try await provider.withSession { session in
            let text = LMInput.Text(tokens: MLXArray(prompt.map(Int32.init)))
            var cache: [any KVCache] = []
            let handover: DecodeHandover
            switch row.kind {
            case .standard:
                cache = try session.newCache(parameters: parameters)
                let warmed = try session.prefill(
                    text: text, cache: cache, checkpoints: [:], checkpointBaseOffset: 0,
                    prefillStepSize: 4, consumeAll: false, initialState: nil,
                    evalPolicy: .pipelined, storedForm: nil)
                handover = .standard(
                    remainder: warmed.remainder, fullText: text, state: warmed.state)
            case .speculativeSplit, .speculativeWholePrompt:
                let plan = try #require(
                    session.speculation.plan(
                        for: SpeculationRequest(
                            isTextOnly: true, kvBits: nil, kvScheme: row.scheme,
                            temperature: 0, promptTokens: prompt.count, restoresPrefix: false,
                            storedLeaf: nil)))
                cache = try session.newCache(parameters: parameters)
                var split = 0
                if row.kind == .speculativeSplit {
                    split = 8
                    _ = try session.prefill(
                        text: LMInput.Text(tokens: MLXArray(prompt[...split].map(Int32.init))),
                        cache: cache, checkpoints: [:], checkpointBaseOffset: 0,
                        prefillStepSize: 4, consumeAll: false, initialState: nil,
                        evalPolicy: .pipelined, storedForm: nil)
                }
                handover = .speculative(
                    plan, input: LMInput(text: text), prefilledPrefixTokens: split,
                    positionDelta: 0)
            case .preparing:
                cache = try session.newCache(parameters: parameters)
                handover = .preparing(LMInput(text: text), prepare: nil)
            case .prefillStrategySingleShot:
                handover = .prefillStrategy(LMInput(text: text))
            case .prefillStrategyChunked:
                handover = .prefillStrategy(LMInput(tokens: text.tokens[.newAxis]))
            }
            let verbsBefore = provider.recorder.verbs.count
            let decode = try DecodeStart.make(
                handover, cache: &cache, session: session, parameters: parameters)
            await decode.reportPrefillFinished(
                .init(
                    promptTokens: prompt.count, cachedTokens: 0,
                    newTokensToPrefill: prompt.count, prefillMs: 1),
                to: { event in progress.append(event) })
            let owner = row.kind.keepsCache ? decode.cacheOwner(lease: nil) : nil
            let (stream, task) = decode.startLoop(
                promptTokenCount: prompt.count, session: session, tools: nil,
                generatedTokens: row.kind.keepsCache ? GeneratedTokenRecorder() : nil)
            return Outcome(
                stream: stream, task: task, owner: owner,
                verbs: Array(provider.recorder.verbs.dropFirst(verbsBefore)), arm: decode.arm)
        }
    }

    private static func drain(_ outcome: Outcome) async -> String {
        var text = ""
        for await event in outcome.stream {
            if case .chunk(let chunk) = event { text += chunk }
        }
        await outcome.task.value
        return text
    }

    @Test(arguments: rows)
    func theHandoverDecidesTheVerbTheConversionAndThePenaltySeed(_ row: Row) async throws {
        let model = ToyLanguageModel(script: Self.prompt + Self.completion, headDim: 64)
        let provider = ToyModelSessionProvider(
            model: model, tokenizer: ToySequencingTokenizer(),
            speculation: row.kind.speculates ? .scriptedDFlash2(over: model) : .none)
        let progress = ProgressEventLog()

        let outcome = try await Self.start(
            row, provider: provider, parameters: Self.parameters(row), progress: progress)
        let text = await Self.drain(outcome)

        #expect(text == Self.expectedText(row.penalty))

        let expectedVerbs: [ModelVerb] =
            switch row.kind {
            case .standard: [.quantizeKVCache, .makeDecodeIterator]
            case .speculativeSplit, .speculativeWholePrompt: [.makeSpeculativeDecodeIterator]
            case .preparing: [.makePreparingDecodeIterator]
            case .prefillStrategySingleShot, .prefillStrategyChunked: [.makeRawDecodeIterator]
            }
        #expect(outcome.verbs == expectedVerbs)

        // The badge comes first, and only from a speculative start.
        #expect(outcome.arm == (row.kind.speculates ? .dflash2 : nil))
        let kinds = progress.events.map { event -> String in
            switch event {
            case .speculationEngaged(let arm): "badge:\(arm.rawValue)"
            case .prefillFinished: "prefillFinished"
            default: "other"
            }
        }
        #expect(kinds == (row.kind.speculates ? ["badge:dflash2"] : []) + ["prefillFinished"])

        // Where the scheme converted: the prompt's last token, then decode.
        let forms = model.forwardForms
        let converted: ToyForwardForms.Form = row.scheme == nil ? .fullPrecision : .turboQuant
        #expect(
            forms.form(at: Self.lastPromptPosition)
                == (Self.lastTokenConverted(row) ? converted : .fullPrecision))
        #expect(forms.form(at: Self.firstDecodePosition) == converted)

        // The owner keeps the array decode advanced, in the scheme's form.
        if let owner = outcome.owner {
            #expect(owner.cache.count == 2)
            for layer in owner.cache {
                #expect(layer.offset > Self.prompt.count, "decode advanced the kept array")
                #expect(
                    row.scheme == nil
                        ? layer is KVCacheSimple : layer is TurboQuantKVCache,
                    "kept \(type(of: layer))")
            }
        }
    }

    /// The single-shot Prefill Strategy arm with affine `kvBits` keeps the
    /// vendor's own penalties (the `kvBits` carve-out, theoretical: #252): its
    /// presence window is seeded with the prompt, whose `x` replaces the first
    /// one, and its plan quantizes after the prefill.
    @Test func theSingleShotRouteWithKVBitsKeepsTheVendorPenalties() async throws {
        let row = Row(kind: .prefillStrategySingleShot, scheme: nil, penalty: .presence)
        let model = ToyLanguageModel(script: Self.prompt + Self.completion, headDim: 64)
        let provider = ToyModelSessionProvider(model: model, tokenizer: ToySequencingTokenizer())

        let outcome = try await Self.start(
            row, provider: provider, parameters: Self.parameters(row, kvBits: 8),
            progress: ProgressEventLog())

        #expect(await Self.drain(outcome) == Self.expectedText(.frequency))
        #expect(model.forwardForms.form(at: Self.lastPromptPosition) == .fullPrecision)
        #expect(model.forwardForms.form(at: Self.firstDecodePosition) == .affineQuantized)
    }

    /// A leased leaf's cache adopts the array decode advances: a conversion
    /// replaced its attention entries, and the lease's check-in must store
    /// those, not the full-precision layers it lent.
    @Test func aLeaseAdoptsTheConvertedArray() async throws {
        let row = Row(kind: .standard, scheme: .turbo8v4, penalty: .none)
        let model = ToyLanguageModel(script: Self.prompt + Self.completion, headDim: 64)
        let provider = ToyModelSessionProvider(model: model, tokenizer: ToySequencingTokenizer())
        let parameters = Self.parameters(row)

        let (lease, owner, kept) = try await provider.withSession { session in
            let text = LMInput.Text(tokens: MLXArray(Self.prompt.map(Int32.init)))
            var cache = try session.newCache(parameters: parameters)
            let lease = FinalGenerationCache(cache)
            let warmed = try session.prefill(
                text: text, cache: cache, checkpoints: [:], checkpointBaseOffset: 0,
                prefillStepSize: 4, consumeAll: false, initialState: nil,
                evalPolicy: .pipelined, storedForm: nil)
            let decode = try DecodeStart.make(
                .standard(remainder: warmed.remainder, fullText: text, state: warmed.state),
                cache: &cache, session: session, parameters: parameters)
            let owner = decode.cacheOwner(lease: lease)
            return (lease, owner, UnsafeSendableBox(cache))
        }

        #expect(owner === lease)
        #expect(lease.cache.allSatisfy { $0 is TurboQuantKVCache })
        #expect(
            zip(lease.cache, kept.value).allSatisfy { ($0 as AnyObject) === ($1 as AnyObject) })
    }
}
