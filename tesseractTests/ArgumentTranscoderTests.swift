import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

// MARK: - Fixtures

/// Tool schema used for schema-typed parameter conversion.
private let demoToolSpecs: [ToolSpec] = [
    [
        "type": "function",
        "function": [
            "name": "demo",
            "description": "demo tool",
            "parameters": [
                "type": "object",
                "properties": [
                    "text": ["type": "string"] as [String: any Sendable],
                    "count": ["type": "integer"] as [String: any Sendable],
                    "ratio": ["type": "number"] as [String: any Sendable],
                    "flag": ["type": "boolean"] as [String: any Sendable],
                    "config": ["type": "object"] as [String: any Sendable],
                    "items": ["type": "array"] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]
]

/// The in-flight text of one tool-call block as the delta stream carries it:
/// open tag included, close tag never included (the tracker withholds it).
private struct CorpusBlock {
    let name: String
    let deltaText: String
    /// Full block content handed to the reference parser (close tag added).
    var parserContent: String { deltaText + "</tool_call>" }
}

private let xmlCorpus: [CorpusBlock] = [
    CorpusBlock(
        name: "canonical single string param",
        deltaText:
            "<tool_call>\n<function=demo>\n<parameter=text>\nhello world\n</parameter>\n</function>\n"
    ),
    CorpusBlock(
        name: "mixed schema-typed params",
        deltaText: "<tool_call>\n<function=demo>\n"
            + "<parameter=text>\nline one\nline two\n</parameter>\n"
            + "<parameter=count>\n42\n</parameter>\n"
            + "<parameter=ratio>\n3.5\n</parameter>\n"
            + "<parameter=flag>\ntrue\n</parameter>\n"
            + #"<parameter=config>{"a": 1, "b": [1, 2]}</parameter>"# + "\n"
            + #"<parameter=items>["x", "y"]</parameter>"# + "\n"
            + "</function>\n"
    ),
    CorpusBlock(
        name: "quotes backslashes and unicode",
        deltaText: "<tool_call>\n<function=demo>\n<parameter=text>\n"
            + "say \"hi\" \\ done — π ≈ 3.14159 🚀 日本語\ttab\n"
            + "</parameter>\n</function>\n"
    ),
    CorpusBlock(
        name: "angle brackets and near-terminators in content",
        deltaText: "<tool_call>\n<function=demo>\n<parameter=text>\n"
            + "if a < b { </p is not a tag, nor is <function= or <parameter\n"
            + "</parameter>\n</function>\n"
    ),
    CorpusBlock(
        name: "zero parameters",
        deltaText: "<tool_call>\n<function=demo>\n</function>\n"
    ),
    CorpusBlock(
        name: "unknown function defaults to string typing",
        deltaText:
            "<tool_call>\n<function=mystery>\n<parameter=whatever>\nvalue\n</parameter>\n</function>\n"
    ),
    CorpusBlock(
        name: "empty string value",
        deltaText: "<tool_call>\n<function=demo>\n<parameter=text>\n</parameter>\n</function>\n"
    ),
    CorpusBlock(
        name: "value without edge newlines",
        deltaText: "<tool_call><function=demo><parameter=text>abc</parameter></function>"
    ),
    CorpusBlock(
        name: "trailing double newline keeps one",
        deltaText:
            "<tool_call>\n<function=demo>\n<parameter=text>\nabc\n\n</parameter>\n</function>\n"
    ),
]

private let jsonCorpus: [CorpusBlock] = [
    CorpusBlock(
        name: "canonical wrapper",
        deltaText: "<tool_call>\n" + #"{"name": "demo", "arguments": {"text": "hello"}}"# + "\n"
    ),
    CorpusBlock(
        name: "nested structures and escapes",
        deltaText: "<tool_call>"
            + #"{"name": "demo", "arguments": {"config": {"a": 1, "b": [true, null, "s"]}, "text": "x\"y\\z\nnl"}}"#
    ),
    CorpusBlock(
        name: "arguments before name",
        deltaText: "<tool_call>" + #"{"arguments": {"text": "hi"}, "name": "demo"}"#
    ),
    CorpusBlock(
        name: "empty arguments object",
        deltaText: "<tool_call>" + #"{"name": "demo", "arguments": {}}"#
    ),
    CorpusBlock(
        name: "unicode escapes",
        deltaText: "<tool_call>"
            + #"{"name": "demo", "arguments": {"text": "A\n\t tail — 🚀"}}"#
    ),
    CorpusBlock(
        name: "explicit unicode escape sequence",
        deltaText: "<tool_call>"
            + "{\"name\": \"demo\", \"arguments\": {\"text\": \"pre \\u0041\\u00e9 post\"}}"
    ),
    CorpusBlock(
        name: "extra wrapper keys",
        deltaText: "<tool_call>"
            + #"{"name": "demo", "id": 3, "arguments": {"text": "hi"}, "extra": "x"}"#
    ),
    CorpusBlock(
        name: "whitespace-rich formatting",
        deltaText:
            "<tool_call>\n{\n  \"name\" : \"demo\",\n  \"arguments\" : {\n    \"count\": 42,\n    \"text\": \"hi\"\n  }\n}\n"
    ),
]

// MARK: - Helpers

/// Deterministic LCG so randomized splits are reproducible.
private struct SplitRNG {
    var state: UInt64
    mutating func next(_ bound: Int) -> Int {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Int(state >> 33) % max(bound, 1)
    }
}

/// Split `text` into fragments at character granularity.
private func randomSplit(_ text: String, seed: UInt64) -> [String] {
    var rng = SplitRNG(state: seed)
    var pieces: [String] = []
    var rest = Substring(text)
    while !rest.isEmpty {
        let take = min(rest.count, 1 + rng.next(7))
        pieces.append(String(rest.prefix(take)))
        rest = rest.dropFirst(take)
    }
    return pieces
}

private func charSplit(_ text: String) -> [String] {
    text.map(String.init)
}

/// A transcoder with a deterministic id mint.
private func makeTranscoder(
    format: ToolCallFormat, toolSpecs: [ToolSpec]?
) -> ArgumentTranscoder {
    var counter = 0
    return ArgumentTranscoder(format: format, toolSpecs: toolSpecs) {
        counter += 1
        return "call_test_\(counter)"
    }
}

/// One reconstructed wire call: the engagement delta plus its concatenated
/// argument fragments.
private struct WireCall {
    var id: String?
    var name: String?
    var index: Int
    var arguments = ""
    /// Argument text after each delta, for prefix-parseability assertions.
    var argumentAccumulations: [String] = []
}

private func reconstruct(_ deltas: [OpenAI.ToolCall]) -> [WireCall] {
    var calls: [Int: WireCall] = [:]
    for delta in deltas {
        guard let index = delta.index else {
            Issue.record("wire delta without index: \(delta)")
            continue
        }
        var call = calls[index] ?? WireCall(index: index)
        if let id = delta.id { call.id = id }
        if let name = delta.function?.name { call.name = name }
        if let piece = delta.function?.arguments, !piece.isEmpty {
            call.arguments += piece
            call.argumentAccumulations.append(call.arguments)
        }
        calls[index] = call
    }
    return calls.keys.sorted().compactMap { calls[$0] }
}

private func parsesAsJSONObject(_ text: String) -> Bool {
    guard let data = text.data(using: .utf8) else { return false }
    return (try? JSONSerialization.jsonObject(with: data)) is [String: Any]
}

private func decodeArguments(_ text: String) -> [String: JSONValue]? {
    guard let data = text.data(using: .utf8) else { return nil }
    return try? JSONDecoder().decode([String: JSONValue].self, from: data)
}

/// Reference parse of a corpus block through the same vendor parser the
/// generation pipeline uses.
private func referenceParse(
    _ block: CorpusBlock, format: ToolCallFormat, toolSpecs: [ToolSpec]?
) -> ToolCall? {
    format.createParser().parse(content: block.parserContent, tools: toolSpecs)
}

/// Drive a fresh transcoder with the given fragmentation, then close with
/// the parser's `.toolCall`. Returns the wire deltas.
private func transcode(
    _ pieces: [String],
    format: ToolCallFormat,
    toolSpecs: [ToolSpec]?,
    close: AgentGeneration?
) -> (deltas: [OpenAI.ToolCall], transcoder: ArgumentTranscoder) {
    var transcoder = makeTranscoder(format: format, toolSpecs: toolSpecs)
    var deltas: [OpenAI.ToolCall] = []
    for piece in pieces {
        deltas += transcoder.ingest(.toolCallDelta(name: nil, argumentsDelta: piece))
    }
    if let close {
        deltas += transcoder.ingest(close)
    }
    return (deltas, transcoder)
}

private func corpus(for format: ToolCallFormat) -> [CorpusBlock] {
    format == .xmlFunction ? xmlCorpus : jsonCorpus
}

// MARK: - Corpus properties

struct ArgumentTranscoderCorpusTests {

    /// Every fragmentation of the same block yields byte-identical
    /// concatenated arguments, semantically equal to the reference parse.
    @Test(arguments: [ToolCallFormat.xmlFunction, .json])
    func fragmentationInvariance(format: ToolCallFormat) throws {
        for block in corpus(for: format) {
            let oracle = try #require(
                referenceParse(block, format: format, toolSpecs: demoToolSpecs),
                "reference parser rejected corpus block: \(block.name)"
            )

            var fragmentations = [charSplit(block.deltaText), [block.deltaText]]
            for seed: UInt64 in [1, 2, 3, 42, 999] {
                fragmentations.append(randomSplit(block.deltaText, seed: seed))
            }

            var canonicalArguments: String?
            for pieces in fragmentations {
                let (deltas, transcoder) = transcode(
                    pieces, format: format, toolSpecs: demoToolSpecs,
                    close: .toolCall(oracle)
                )
                let calls = reconstruct(deltas)
                #expect(calls.count == 1, "block: \(block.name)")
                guard let call = calls.first else { continue }

                #expect(call.name == oracle.function.name, "block: \(block.name)")
                #expect(call.id?.hasPrefix("call_") == true, "block: \(block.name)")

                if let canonical = canonicalArguments {
                    #expect(
                        call.arguments == canonical,
                        "fragmentation changed wire bytes — block: \(block.name)"
                    )
                } else {
                    canonicalArguments = call.arguments
                }

                let streamed = try #require(
                    decodeArguments(call.arguments),
                    "streamed arguments do not parse — block: \(block.name), args: \(call.arguments)"
                )
                #expect(
                    ArgumentTranscoder.equivalent(
                        .object(streamed), .object(oracle.function.arguments)),
                    "semantic mismatch — block: \(block.name), streamed: \(call.arguments)"
                )
                #expect(
                    transcoder.crossCheckMismatchCount == 0,
                    "cross-check flagged a mismatch — block: \(block.name)"
                )
            }
        }
    }

    /// No strict prefix of the accumulated arguments parses as JSON — the
    /// client finalizes on the first parseable accumulation.
    @Test(arguments: [ToolCallFormat.xmlFunction, .json])
    func neverParseableStrictPrefix(format: ToolCallFormat) throws {
        for block in corpus(for: format) {
            let oracle = try #require(
                referenceParse(block, format: format, toolSpecs: demoToolSpecs))
            let (deltas, _) = transcode(
                charSplit(block.deltaText), format: format, toolSpecs: demoToolSpecs,
                close: .toolCall(oracle)
            )
            let calls = reconstruct(deltas)
            guard let call = calls.first else {
                Issue.record("no wire call for block: \(block.name)")
                continue
            }
            let full = call.arguments
            #expect(parsesAsJSONObject(full), "block: \(block.name)")
            var prefix = ""
            for character in full.dropLast() {
                prefix.append(character)
                #expect(
                    !parsesAsJSONObject(prefix),
                    "strict prefix parses — block: \(block.name), prefix: \(prefix)"
                )
            }
        }
    }

    /// Truncation at every character offset, closed via `finish()`, still
    /// yields accumulated arguments that parse (Wire-Valid Close).
    @Test(arguments: [ToolCallFormat.xmlFunction, .json])
    func wireValidCloseAtEveryTruncation(format: ToolCallFormat) throws {
        for block in corpus(for: format) {
            let text = block.deltaText
            for offset in 0...text.count {
                let prefix = String(text.prefix(offset))
                var transcoder = makeTranscoder(format: format, toolSpecs: demoToolSpecs)
                var deltas: [OpenAI.ToolCall] = []
                for piece in randomSplit(prefix, seed: UInt64(offset) &+ 7) {
                    deltas += transcoder.ingest(
                        .toolCallDelta(name: nil, argumentsDelta: piece))
                }
                deltas += transcoder.finish()
                for call in reconstruct(deltas) where call.id != nil {
                    #expect(
                        parsesAsJSONObject(call.arguments),
                        "truncated close does not parse — block: \(block.name), offset: \(offset), args: \(call.arguments)"
                    )
                }
            }
        }
    }
}

// MARK: - Wire shape

struct ArgumentTranscoderWireShapeTests {

    @Test func engagementDeltaCarriesIdentityAndEmptyArguments() throws {
        let block = xmlCorpus[0]
        let (deltas, _) = transcode(
            [block.deltaText], format: .xmlFunction, toolSpecs: demoToolSpecs, close: nil)
        let first = try #require(deltas.first)
        #expect(first.id == "call_test_1")
        #expect(first.type == "function")
        #expect(first.function?.name == "demo")
        #expect(first.function?.arguments?.isEmpty == true)
        #expect(first.index == 0)
        for delta in deltas.dropFirst() {
            #expect(delta.id == nil)
            #expect(delta.function?.name == nil)
            #expect(delta.index == 0)
        }
    }

    @Test func multipleCallsIncrementIndexAndMintFreshIDs() throws {
        var transcoder = makeTranscoder(format: .xmlFunction, toolSpecs: demoToolSpecs)
        var deltas: [OpenAI.ToolCall] = []
        for block in [xmlCorpus[0], xmlCorpus[4]] {
            let oracle = try #require(
                referenceParse(block, format: .xmlFunction, toolSpecs: demoToolSpecs))
            for piece in randomSplit(block.deltaText, seed: 5) {
                deltas += transcoder.ingest(.toolCallDelta(name: nil, argumentsDelta: piece))
            }
            deltas += transcoder.ingest(.toolCall(oracle))
        }
        let calls = reconstruct(deltas)
        #expect(calls.count == 2)
        #expect(calls.map(\.index) == [0, 1])
        #expect(calls[0].id == "call_test_1")
        #expect(calls[1].id == "call_test_2")
        #expect(parsesAsJSONObject(calls[0].arguments))
        #expect(parsesAsJSONObject(calls[1].arguments))
        #expect(transcoder.crossCheckMismatchCount == 0)
    }

    /// A parse-failure chain: block one never gets a `.toolCall` (its
    /// `</function>` is missing), block two's open tag arrives in the same
    /// delta run. Block one closes wire-valid, block two engages with a fresh
    /// index. (Inside an open string value a `<tool_call>` literal is
    /// content, so the chain is only recognized between parameters.)
    @Test func backToBackBlocksWithoutCloseEventSplitCleanly() throws {
        let brokenBlock =
            "<tool_call>\n<function=demo>\n<parameter=text>\nvalue\n</parameter>\n"
        let secondBlock = xmlCorpus[0].deltaText
        var transcoder = makeTranscoder(format: .xmlFunction, toolSpecs: demoToolSpecs)
        var deltas: [OpenAI.ToolCall] = []
        for piece in randomSplit(brokenBlock + secondBlock, seed: 11) {
            deltas += transcoder.ingest(.toolCallDelta(name: nil, argumentsDelta: piece))
        }
        let oracle = try #require(
            referenceParse(xmlCorpus[0], format: .xmlFunction, toolSpecs: demoToolSpecs))
        deltas += transcoder.ingest(.toolCall(oracle))

        let calls = reconstruct(deltas)
        #expect(calls.count == 2)
        #expect(calls.map(\.index) == [0, 1])
        #expect(parsesAsJSONObject(calls[0].arguments))
        #expect(parsesAsJSONObject(calls[1].arguments))
    }

    @Test func malformedEventClosesEngagedCallWireValid() {
        let partial = "<tool_call>\n<function=demo>\n<parameter=text>\nhalf way"
        var transcoder = makeTranscoder(format: .xmlFunction, toolSpecs: demoToolSpecs)
        var deltas: [OpenAI.ToolCall] = []
        for piece in charSplit(partial) {
            deltas += transcoder.ingest(.toolCallDelta(name: nil, argumentsDelta: piece))
        }
        deltas += transcoder.ingest(.malformedToolCall(partial))
        let calls = reconstruct(deltas)
        #expect(calls.count == 1)
        #expect(parsesAsJSONObject(calls[0].arguments))
        #expect(transcoder.hasStreamedFragments)
    }

    @Test func malformedBeforeNameLockStreamsNothing() {
        let preName = "<tool_call>\n<funct"
        var transcoder = makeTranscoder(format: .xmlFunction, toolSpecs: demoToolSpecs)
        var deltas: [OpenAI.ToolCall] = []
        for piece in charSplit(preName) {
            deltas += transcoder.ingest(.toolCallDelta(name: nil, argumentsDelta: piece))
        }
        deltas += transcoder.ingest(.malformedToolCall(preName))
        deltas += transcoder.finish()
        #expect(deltas.isEmpty)
        #expect(!transcoder.hasStreamedFragments)
    }

    @Test func finishIsIdempotent() {
        let partial = "<tool_call>\n<function=demo>\n<parameter=text>\nhalf"
        var transcoder = makeTranscoder(format: .xmlFunction, toolSpecs: demoToolSpecs)
        var deltas: [OpenAI.ToolCall] = []
        for piece in charSplit(partial) {
            deltas += transcoder.ingest(.toolCallDelta(name: nil, argumentsDelta: piece))
        }
        deltas += transcoder.finish()
        let extra = transcoder.finish()
        #expect(extra.isEmpty)
        let calls = reconstruct(deltas)
        #expect(calls.count == 1)
        #expect(parsesAsJSONObject(calls[0].arguments))
    }

    @Test func textAndThinkingEventsProduceNothing() {
        var transcoder = makeTranscoder(format: .xmlFunction, toolSpecs: demoToolSpecs)
        #expect(transcoder.ingest(.text("hello")).isEmpty)
        #expect(transcoder.ingest(.thinking("hmm")).isEmpty)
        #expect(transcoder.ingest(.thinkStart).isEmpty)
        #expect(transcoder.ingest(.thinkEnd).isEmpty)
    }
}

// MARK: - Atomic fallback

struct ArgumentTranscoderAtomicFallbackTests {

    /// Non-transcodable formats keep the atomic two-delta emission and ignore
    /// deltas entirely.
    @Test func nonTranscodableFormatEmitsAtomically() throws {
        var transcoder = makeTranscoder(format: .glm4, toolSpecs: demoToolSpecs)
        #expect(
            transcoder.ingest(
                .toolCallDelta(name: nil, argumentsDelta: "demo<arg_key>text</arg_key>")
            ).isEmpty)

        let call = ToolCall(
            function: .init(name: "demo", arguments: ["text": JSONValue.string("hello")])
        )
        let deltas = transcoder.ingest(.toolCall(call))
        #expect(deltas.count == 2)
        #expect(deltas[0].id == "call_test_1")
        #expect(deltas[0].type == "function")
        #expect(deltas[0].function?.name == "demo")
        #expect(deltas[0].function?.arguments?.isEmpty == true)
        #expect(deltas[0].index == 0)
        #expect(deltas[1].id == nil)
        #expect(deltas[1].function?.arguments == #"{"text":"hello"}"#)
        #expect(deltas[1].index == 0)
        #expect(!transcoder.hasStreamedFragments)
        #expect(transcoder.finish().isEmpty)
    }

    /// A `.toolCall` with no preceding deltas (a bare-JSON block) falls back
    /// to atomic emission even on a transcodable format.
    @Test func parsedCallWithoutDeltasEmitsAtomically() {
        var transcoder = makeTranscoder(format: .json, toolSpecs: demoToolSpecs)
        let call = ToolCall(
            function: .init(name: "demo", arguments: ["count": JSONValue.int(3)])
        )
        let deltas = transcoder.ingest(.toolCall(call))
        #expect(deltas.count == 2)
        #expect(deltas[0].function?.name == "demo")
        #expect(deltas[1].function?.arguments == #"{"count":3}"#)
        #expect(!transcoder.hasStreamedFragments)
    }

    /// Atomic and transcoded emissions share the index counter.
    @Test func atomicAfterTranscodedIncrementsIndex() throws {
        var transcoder = makeTranscoder(format: .json, toolSpecs: demoToolSpecs)
        let block = jsonCorpus[0]
        let oracle = try #require(
            referenceParse(block, format: .json, toolSpecs: demoToolSpecs))
        var deltas: [OpenAI.ToolCall] = []
        for piece in randomSplit(block.deltaText, seed: 9) {
            deltas += transcoder.ingest(.toolCallDelta(name: nil, argumentsDelta: piece))
        }
        deltas += transcoder.ingest(.toolCall(oracle))
        let bare = ToolCall(
            function: .init(name: "demo", arguments: [:] as [String: JSONValue]))
        deltas += transcoder.ingest(.toolCall(bare))
        let calls = reconstruct(deltas)
        #expect(calls.map(\.index) == [0, 1])
    }
}

// MARK: - JSON wrapper specifics

struct ArgumentTranscoderJSONWrapperTests {

    /// A stringified `arguments` value streams nothing; the close emits the
    /// parser-normalized arguments as one whole fragment.
    @Test func stringifiedArgumentsEmitWholeAtClose() throws {
        let deltaText =
            "<tool_call>"
            + #"{"name": "demo", "arguments": "{\"text\": \"hi\"}"}"#
        let block = CorpusBlock(name: "stringified", deltaText: deltaText)
        let oracle = try #require(
            referenceParse(block, format: .json, toolSpecs: demoToolSpecs),
            "vendor parser should normalize stringified arguments"
        )

        let (deltas, transcoder) = transcode(
            charSplit(deltaText), format: .json, toolSpecs: demoToolSpecs,
            close: .toolCall(oracle)
        )
        let calls = reconstruct(deltas)
        #expect(calls.count == 1)
        let call = try #require(calls.first)
        #expect(call.name == "demo")
        let streamed = try #require(decodeArguments(call.arguments))
        #expect(
            ArgumentTranscoder.equivalent(
                .object(streamed), .object(oracle.function.arguments)))
        #expect(transcoder.crossCheckMismatchCount == 0)
    }

    /// The wrapper's `name` engages even when a nested object also contains a
    /// `"name"` key.
    @Test func nestedNameKeyDoesNotConfuseEngagement() throws {
        let deltaText =
            "<tool_call>"
            + #"{"arguments": {"config": {"name": "inner"}, "text": "hi"}, "name": "demo"}"#
        let block = CorpusBlock(name: "nested name", deltaText: deltaText)
        let oracle = try #require(
            referenceParse(block, format: .json, toolSpecs: demoToolSpecs))
        let (deltas, transcoder) = transcode(
            charSplit(deltaText), format: .json, toolSpecs: demoToolSpecs,
            close: .toolCall(oracle)
        )
        let calls = reconstruct(deltas)
        #expect(calls.count == 1)
        #expect(calls.first?.name == "demo")
        #expect(transcoder.crossCheckMismatchCount == 0)
    }

    @Test func crossCheckFlagsSemanticDivergence() throws {
        let block = jsonCorpus[0]
        var transcoder = makeTranscoder(format: .json, toolSpecs: demoToolSpecs)
        var deltas: [OpenAI.ToolCall] = []
        for piece in charSplit(block.deltaText) {
            deltas += transcoder.ingest(.toolCallDelta(name: nil, argumentsDelta: piece))
        }
        // Close with a parse that disagrees with the streamed bytes.
        let divergent = ToolCall(
            function: .init(name: "demo", arguments: ["text": JSONValue.string("other")]))
        deltas += transcoder.ingest(.toolCall(divergent))
        #expect(transcoder.crossCheckMismatchCount == 1)
        // The wire is never retro-corrected: the streamed bytes still parse.
        let calls = reconstruct(deltas)
        #expect(parsesAsJSONObject(calls[0].arguments))
    }
}

// MARK: - Equivalence

struct ArgumentTranscoderEquivalenceTests {

    @Test func numericBooleanTolerance() {
        #expect(ArgumentTranscoder.equivalent(.int(1), .bool(true)))
        #expect(ArgumentTranscoder.equivalent(.bool(false), .int(0)))
        #expect(ArgumentTranscoder.equivalent(.int(3), .double(3.0)))
        #expect(!ArgumentTranscoder.equivalent(.int(2), .bool(true)))
        #expect(!ArgumentTranscoder.equivalent(.string("1"), .int(1)))
        #expect(
            ArgumentTranscoder.equivalent(
                .object(["a": .array([.int(1), .null])]),
                .object(["a": .array([.bool(true), .null])])
            ))
        #expect(
            !ArgumentTranscoder.equivalent(
                .object(["a": .int(1)]), .object(["b": .int(1)])
            ))
    }
}
