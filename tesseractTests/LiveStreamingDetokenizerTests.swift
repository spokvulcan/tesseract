import Foundation
import MLXLMCommon
import Testing
import os

@testable import Tesseract_Agent

struct LiveStreamingDetokenizerTests {
    @Test func loadedByteLevelTokenizerReleasesWithoutWaitingForNewline() async throws {
        let tokenizer = try await ByteLevelTokenizerFixture.load()
        var detokenizer = LinearStreamingDetokenizer(tokenizer: tokenizer, delivery: .live)
        #expect(detokenizer.decodesFromBytes)
        for byte in "hello".utf8 {
            let chunks = detokenizer.append(token: Int(byte))
            #expect(chunks.map { Array($0.utf8) } == [[byte]])
        }
        #expect(detokenizer.finish().isEmpty)
    }

    @Test func addedTokensPreserveLiteralBoundariesAndReleaseSteps() async throws {
        let tokenizer = try await ByteLevelTokenizerFixture.load(
            addedTokens: ["Ġ", "", "😀", "\n", "\u{fffd}", "\u{600}", "\u{600}\u{fffd}"])
        let cases = [
            [0xE2, 256, 0x82, 0xAC, 120],
            [0xE2, 257, 0x82, 0xAC, 120],
            [256, 257, 258, 258, 259, 97, 259, 0xF0, 0x9F],
            [97, 260, 120, 261, 260, 120],
            [97, -1, 9999, 0xE2, -1, 0x82, 0xAC, 120],
            [262, 262, 120, 260, 262, 259],
        ]
        for tokens in cases {
            expectLiveDetokenizationParity(tokens, tokenizer: tokenizer)
        }
    }

    @Test func longMalformedRunRemainsResponsiveUntilTextCompletes() async throws {
        let tokenizer = try await ByteLevelTokenizerFixture.load()
        var live = LinearStreamingDetokenizer(tokenizer: tokenizer, delivery: .live)
        let start = ContinuousClock.now
        for _ in 0..<12_000 {
            #expect(live.append(token: 0xFF).isEmpty)
        }
        let chunks = live.append(token: 120)
        let elapsed = start.duration(to: .now)
        #expect(
            chunks.map { Array($0.utf8) } == [
                Array((String(repeating: "\u{fffd}", count: 12_000) + "x").utf8)
            ])
        #expect(elapsed < .seconds(1), "\(elapsed)")
    }

    @Test func byteShapesMatchTheReferenceAtEveryToken() async throws {
        let tokenizer = try await ByteLevelTokenizerFixture.load(addedTokens: ["", "<special>"])
        let samples = [
            "First paragraph.\n\nSecond paragraph without a trailing newline.",
            #"<tool_call>{"name":"write","arguments":{"text":"hello"}}</tool_call>"#,
            "日本語 café € — 😀 🏳️‍🌈 🇺🇸 e\u{301} '️",
            "func f() {\n\treturn 42\n}\r\n",
            String(repeating: "abcdefgh", count: 128),
        ]
        for sample in samples {
            expectLiveDetokenizationParity(Array(sample.utf8).map(Int.init), tokenizer: tokenizer)
        }
        var random = SeededGenerator(seed: 0x487)
        for _ in 0..<150 {
            let tokens = (0..<60).map { _ in Int.random(in: 0...257, using: &random) }
            expectLiveDetokenizationParity(tokens, tokenizer: tokenizer)
        }
    }

    @Test func unsupportedDecodersAndCleanupKeepNaiveLiveStreaming() async throws {
        for cleanup in [true, nil] as [Bool?] {
            let tokenizer = try await ByteLevelTokenizerFixture.load(cleanup: cleanup)
            expectLiveDetokenizationParity(
                Array("a  b , c . don't\n".utf8).map(Int.init), tokenizer: tokenizer,
                viaBytes: false)
        }
        let sequence = try await ByteLevelTokenizerFixture.load(
            decoder: ["type": "Sequence", "decoders": [["type": "ByteLevel"]]])
        expectLiveDetokenizationParity(
            Array("café 😀 tail".utf8).map(Int.init), tokenizer: sequence, viaBytes: false)
        expectLiveDetokenizationParity(
            [97, 0xE2, 0x82, 120], tokenizer: FakeChatMLTokenizer(), viaBytes: false)
    }

    @Test func loadingPreservesTemplateRenderingAndSpecialTokenDecodeModes() async throws {
        let tokenizer = try await ByteLevelTokenizerFixture.load(addedTokens: ["<special>"])
        let rendering = try #require(tokenizer as? any ChatTemplateRendering)
        let messages: [[String: any Sendable]] = [["role": "user", "content": "hello"]]
        let rendered = try rendering.renderChatTemplate(
            messages: messages, tools: nil, additionalContext: nil)
        #expect(rendered == "hello")
        #expect(
            try tokenizer.applyChatTemplate(messages: messages)
                == tokenizer.encode(text: rendered, addSpecialTokens: false))
        #expect(tokenizer.decode(tokenIds: [256, 97]) == "<special>a")
        #expect(tokenizer.decode(tokenIds: [256, 97], skipSpecialTokens: true) == "a")
    }

    @Test func increasingNewlineFreeRunsDoNotRedecodePrefixes() async throws {
        let loaded = try await ByteLevelTokenizerFixture.load()
        for length in [1_000, 2_000, 4_000, 8_000] {
            let tokenizer = DecodeWorkTokenizer(loaded)
            var live = LinearStreamingDetokenizer(tokenizer: tokenizer, delivery: .live)
            let text = String(repeating: "abcdefgh", count: length / 8)
            var chunks: [String] = []
            for byte in text.utf8 { chunks += live.append(token: Int(byte)) }
            chunks += live.finish()
            #expect(chunks.count == length)
            #expect(Array(chunks.joined().utf8) == Array(text.utf8))
            #expect(tokenizer.decodedTokens == 0)
            #expect(tokenizer.spellings == 8)
            print(
                "live detok work: \(length) tokens, \(tokenizer.decodedTokens) decoded token entries, \(tokenizer.spellings) spelling lookups"
            )
        }
    }

}

func expectLiveDetokenizationParity(
    _ tokens: [Int], tokenizer: any MLXLMCommon.Tokenizer, viaBytes: Bool = true
) {
    var reference = NaiveStreamingDetokenizer(tokenizer: tokenizer)
    var live = LinearStreamingDetokenizer(tokenizer: tokenizer, delivery: .live)
    #expect(live.decodesFromBytes == viaBytes)
    for (step, token) in tokens.enumerated() {
        reference.append(token: token)
        let expected = reference.next().map { [Array($0.utf8)] } ?? []
        let actual = live.append(token: token).map { Array($0.utf8) }
        #expect(actual == expected, "step \(step), token \(token)")
    }
    #expect(live.finish().isEmpty)
}

/// A real swift-transformers BPE tokenizer, with one token per byte and no
/// merges. Loading it exercises the same configuration boundary as a model,
/// without weights or a network request.
enum ByteLevelTokenizerFixture {
    static func load(
        addedTokens: [String] = [], cleanup: Bool? = false,
        decoder: [String: Any] = ["type": "ByteLevel"]
    ) async throws -> any MLXLMCommon.Tokenizer {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("live-detokenizer-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        var vocabulary: [String: Int] = [:]
        for (scalar, byte) in LinearStreamingDetokenizer.byteLevelAlphabet {
            vocabulary[String(scalar)] = Int(byte)
        }
        let data: [String: Any] = [
            "model": ["type": "BPE", "vocab": vocabulary, "merges": []],
            "decoder": decoder,
            "added_tokens": addedTokens.enumerated().map { index, content in
                ["id": 256 + index, "content": content, "special": true] as [String: Any]
            },
        ]
        var config: [String: Any] = [
            "tokenizer_class": "PreTrainedTokenizer",
            "chat_template": "{{ messages[0]['content'] }}",
        ]
        if let cleanup { config["clean_up_tokenization_spaces"] = cleanup }
        try JSONSerialization.data(withJSONObject: data)
            .write(to: directory.appendingPathComponent("tokenizer.json"))
        try JSONSerialization.data(withJSONObject: config)
            .write(to: directory.appendingPathComponent("tokenizer_config.json"))
        return try await AppTokenizerLoader().load(from: directory)
    }
}

/// Observes work at the external tokenizer port. It preserves the capability
/// of the real loaded tokenizer; it cannot grant one to an unknown decoder.
struct DecodeWorkTokenizer: ByteLevelTokenizing {
    let upstream: any MLXLMCommon.Tokenizer
    let byteLevelDecoding: ByteLevelDecoding?
    private let work = OSAllocatedUnfairLock(initialState: (decodedTokens: 0, spellings: 0))

    init(_ upstream: any MLXLMCommon.Tokenizer) {
        self.upstream = upstream
        byteLevelDecoding = (upstream as? any ByteLevelTokenizing)?.byteLevelDecoding
    }

    var decodedTokens: Int { work.withLock { $0.decodedTokens } }
    var spellings: Int { work.withLock { $0.spellings } }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        work.withLock { $0.decodedTokens += tokenIds.count }
        return upstream.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }
    func convertTokenToId(_ token: String) -> Int? { upstream.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? {
        work.withLock { $0.spellings += 1 }
        return upstream.convertIdToToken(id)
    }
    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try upstream.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: additionalContext)
    }
}
