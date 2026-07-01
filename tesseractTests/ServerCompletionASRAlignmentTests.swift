import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

struct ServerCompletionASRAlignmentTests {

    @Test func bearingAlignmentDropsSingleTrailingEOSBeforeExcision() {
        let tokenizer = SingleTokenEOSTokenizer()
        let storedRender =
            [1]
            + tokenizer.encode(text: "<think>", addSpecialTokens: false)
            + [2, 3]
            + tokenizer.encode(text: "</think>", addSpecialTokens: false)
            + tokenizer.encode(text: "\n\n", addSpecialTokens: false)
            + [4, tokenizer.eosID]

        let alignment = ServerCompletion.asrBearingTokenAlignment(
            storedRenderTokens: storedRender,
            bearingOffset: storedRender.count - 1,
            tokenizer: tokenizer
        )

        #expect(alignment?.tokens == Array(storedRender.dropLast()))
        #expect(alignment?.ignoredTrailingTokenCount == 1)

        // The aligned bearing tokens feed Render-Diff Excision against the
        // canonical future render (think block dropped, next turn appended).
        let futureRender = [1, 4, 50, 51]
        let excision = AsymmetricStateRestore.renderDiffExcision(
            bearingTokens: alignment?.tokens ?? [],
            admitPath: futureRender
        )

        #expect(excision.spans == [AsymmetricStateRestore.ExcisionSpan(start: 1, end: 6)])
        #expect(excision.alignedDepth == 2)
        #expect(!excision.seamCut)
        #expect(
            AsymmetricStateRestore.strippedTokens(
                in: alignment?.tokens ?? [], spans: excision.spans
            ) == [1, 4])
    }

    @Test func bearingAlignmentAcceptsModelConfigStopToken() {
        let tokenizer = ConfigStopOnlyTokenizer()
        let storedRender = [1, 2, tokenizer.stopID]

        #expect(
            ServerCompletion.asrBearingTokenAlignment(
                storedRenderTokens: storedRender,
                bearingOffset: 2,
                tokenizer: tokenizer
            ) == nil)

        let alignment = ServerCompletion.asrBearingTokenAlignment(
            storedRenderTokens: storedRender,
            bearingOffset: 2,
            tokenizer: tokenizer,
            stopTokenIDs: [tokenizer.stopID]
        )

        #expect(alignment?.tokens == [1, 2])
        #expect(alignment?.ignoredTrailingTokenCount == 1)
    }

    @Test func bearingAlignmentDropsSingleTemplateNewlineAfterEOS() {
        let tokenizer = SingleTokenEOSTokenizer()
        let newline = tokenizer.encode(text: "\n", addSpecialTokens: false).first!
        let storedRender = [1, tokenizer.eosID, newline]

        let alignment = ServerCompletion.asrBearingTokenAlignment(
            storedRenderTokens: storedRender,
            bearingOffset: 2,
            tokenizer: tokenizer
        )

        #expect(alignment?.tokens == [1, tokenizer.eosID])
        #expect(alignment?.ignoredTrailingTokenCount == 1)
    }

    @Test func bearingAlignmentDropsMergedDoubleNewlineAfterEOS() {
        // The real Ornith/Qwen3.5 final-answer shape: template emits
        // `<|im_end|>\n` then a trailing newline, BPE-merging to a single
        // `\n\n` token after the EOS. The separator tolerance must recognize
        // the merged token, not only a single "\n".
        let tokenizer = SingleTokenEOSTokenizer()
        let doubleNewline = tokenizer.encode(text: "\n\n", addSpecialTokens: false).first!
        let storedRender = [1, tokenizer.eosID, doubleNewline]

        let alignment = ServerCompletion.asrBearingTokenAlignment(
            storedRenderTokens: storedRender,
            bearingOffset: 2,
            tokenizer: tokenizer
        )

        #expect(alignment?.tokens == [1, tokenizer.eosID])
        #expect(alignment?.ignoredTrailingTokenCount == 1)
    }

    @Test func bearingAlignmentRejectsTemplateNewlineWithoutPrecedingEOS() {
        let tokenizer = SingleTokenEOSTokenizer()
        let newline = tokenizer.encode(text: "\n", addSpecialTokens: false).first!

        #expect(
            ServerCompletion.asrBearingTokenAlignment(
                storedRenderTokens: [1, 2, newline],
                bearingOffset: 2,
                tokenizer: tokenizer
            ) == nil)
    }

    @Test func bearingAlignmentKeepsExactLengthRender() {
        let tokenizer = SingleTokenEOSTokenizer()
        let storedRender = [1, 2, 3]

        let alignment = ServerCompletion.asrBearingTokenAlignment(
            storedRenderTokens: storedRender,
            bearingOffset: storedRender.count,
            tokenizer: tokenizer
        )

        #expect(alignment?.tokens == storedRender)
        #expect(alignment?.ignoredTrailingTokenCount == 0)
    }

    @Test func bearingAlignmentRejectsNonEOSTrailingToken() {
        let tokenizer = SingleTokenEOSTokenizer()

        #expect(
            ServerCompletion.asrBearingTokenAlignment(
                storedRenderTokens: [1, 2, 12345],
                bearingOffset: 2,
                tokenizer: tokenizer
            ) == nil)
    }

    @Test func bearingAlignmentRejectsLargerLengthDrift() {
        let tokenizer = SingleTokenEOSTokenizer()

        #expect(
            ServerCompletion.asrBearingTokenAlignment(
                storedRenderTokens: [1, 2, tokenizer.eosID, tokenizer.eosID],
                bearingOffset: 2,
                tokenizer: tokenizer
            ) == nil)
    }
}

private struct SingleTokenEOSTokenizer: Tokenizer {
    let eosID = 999

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        switch text {
        case "<think>": [100]
        case "</think>": [101]
        case "\n\n": [102]
        case "\n": [10]
        case "<|im_end|>": [eosID]
        default: Array(text.utf8).map(Int.init)
        }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(bytes: tokenIds.compactMap { UInt8(exactly: $0) }, encoding: .utf8) ?? ""
    }

    func tokenize(text: String) -> [String] { [] }
    func convertTokenToId(_ token: String) -> Int? { token == "<|im_end|>" ? eosID : nil }
    func convertIdToToken(_ id: Int) -> String? { id == eosID ? "<|im_end|>" : nil }

    var bosToken: String? { nil }
    var bosTokenId: Int? { nil }
    var eosToken: String? { "<|im_end|>" }
    var eosTokenId: Int? { eosID }
    var unknownToken: String? { nil }
    var unknownTokenId: Int? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        []
    }
}

private struct ConfigStopOnlyTokenizer: Tokenizer {
    let stopID = 248_046

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        Array(text.utf8).map(Int.init)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(bytes: tokenIds.compactMap { UInt8(exactly: $0) }, encoding: .utf8) ?? ""
    }

    func tokenize(text: String) -> [String] { [] }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }

    var bosToken: String? { nil }
    var bosTokenId: Int? { nil }
    var eosToken: String? { nil }
    var eosTokenId: Int? { nil }
    var unknownToken: String? { nil }
    var unknownTokenId: Int? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        []
    }
}
