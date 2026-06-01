import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import Tesseract_Agent

/// Behavior of the **Prefill Planner**: the boundary detection it owns and the
/// pure restore/filter decisions it folds into a **Prefill Plan**. Driven by a
/// byte-level fake tokenizer and constructed lookup values — no model, no actor.
@Suite struct PrefillPlannerTests {

    // MARK: - Fake tokenizer

    /// A byte-level tokenizer (one UTF-8 byte ⇒ one token) whose
    /// `applyChatTemplate` reproduces the `<|im_start|>role\ncontent<|im_end|>\n`
    /// envelope and, unless `add_generation_prompt` is false, appends the
    /// generation prompt — think or non-think per `promptStartsThinking`.
    private struct FakeTokenizer: Tokenizer {
        var promptStartsThinking = true

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            Array(text.utf8).map(Int.init)
        }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            String(decoding: tokenIds.compactMap { UInt8(exactly: $0) }, as: UTF8.self)
        }
        func tokenize(text: String) -> [String] { [] }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }

        var bosToken: String? { nil }
        var bosTokenId: Int? { nil }
        var eosToken: String? { "<|im_end|>" }
        var eosTokenId: Int? { nil }
        var unknownToken: String? { nil }
        var unknownTokenId: Int? { nil }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            var rendered = ""
            for message in messages {
                let role = message["role"] as? String ?? ""
                let content = message["content"] as? String ?? ""
                rendered += "<|im_start|>\(role)\n\(content)<|im_end|>\n"
            }
            if let tools, !tools.isEmpty {
                rendered += "<tools>\(tools.count)</tools>\n"
            }
            let addGenerationPrompt = (additionalContext?["add_generation_prompt"] as? Bool) ?? true
            if addGenerationPrompt {
                rendered += Self.generationPrompt(thinking: promptStartsThinking)
            }
            return encode(text: rendered, addSpecialTokens: false)
        }

        static func generationPrompt(thinking: Bool) -> String {
            thinking ? "<|im_start|>assistant\n<think>\n" : "<|im_start|>assistant\n"
        }
    }

    // MARK: - Fixtures

    private let key = CachePartitionKey(modelID: "test-model", kvBits: nil, kvGroupSize: 64)

    private static let system = "You are a helpful assistant."

    private func conversation(
        systemPrompt: String = PrefillPlannerTests.system,
        messages: [HTTPPrefixCacheMessage]
    ) -> HTTPPrefixCacheConversation {
        HTTPPrefixCacheConversation(
            systemPrompt: systemPrompt,
            messages: messages,
            toolDefinitionsDigest: "tools",
            templateContextDigest: "ctx"
        )
    }

    private func fullTokens(
        _ conversation: HTTPPrefixCacheConversation,
        tokenizer: FakeTokenizer
    ) throws -> [Int] {
        try tokenizer.applyChatTemplate(
            messages: conversation.promptMessages,
            tools: nil,
            additionalContext: nil
        )
    }

    private func snapshot(offset: Int, type: HybridCacheSnapshot.CheckpointType = .system) -> HybridCacheSnapshot {
        let kv = KVCacheSimple()
        kv.state = [
            MLXArray.zeros([1, 1, max(offset, 1), 64]),
            MLXArray.zeros([1, 1, max(offset, 1), 64]),
        ]
        return HybridCacheSnapshot.capture(cache: [kv], offset: offset, type: type)!
    }

    private func hit(_ snapshot: HybridCacheSnapshot, promptTokenCount: Int) -> PrefixCacheManager.LookupResult {
        PrefixCacheManager.LookupResult(
            snapshot: snapshot,
            partitionKey: key,
            snapshotTokenOffset: snapshot.tokenOffset,
            sharedPrefixLength: snapshot.tokenOffset,
            reason: .hit(snapshotOffset: snapshot.tokenOffset, totalTokens: promptTokenCount, type: snapshot.checkpointType)
        )
    }

    private var miss: PrefixCacheManager.LookupResult {
        PrefixCacheManager.LookupResult(
            snapshot: nil,
            partitionKey: nil,
            snapshotTokenOffset: 0,
            sharedPrefixLength: 0,
            reason: .missNoSnapshotInPrefix
        )
    }

    private func offsets(_ plan: [(offset: Int, type: HybridCacheSnapshot.CheckpointType)]) -> [Int] {
        plan.map(\.offset)
    }

    // MARK: - plan(): restore vs cold

    @Test func missPlansAColdPrefillCoveringNothing() {
        let plan = PrefillPlanner.plan(
            boundaries: PrefillBoundaries(stablePrefixOffset: 8, lastMessageOffset: nil, lastUserOffset: nil),
            lookupResult: miss,
            checkpointPlan: [(offset: 8, type: .system), (offset: 20, type: .branchPoint)],
            promptTokenCount: 40,
            partitionKey: key
        )
        guard case .cold = plan.restore else { Issue.record("expected cold"); return }
        #expect(plan.prefillBaseOffset == 0)
        // Cold prefill keeps the whole checkpoint plan — nothing is already cached.
        #expect(offsets(plan.checkpointsToCapture) == [8, 20])
    }

    @Test func hitInsideThePromptPlansASuffixRestore() {
        let snap = snapshot(offset: 12)
        let plan = PrefillPlanner.plan(
            boundaries: PrefillBoundaries(stablePrefixOffset: 8, lastMessageOffset: nil, lastUserOffset: nil),
            lookupResult: hit(snap, promptTokenCount: 40),
            checkpointPlan: [(offset: 8, type: .system)],
            promptTokenCount: 40,
            partitionKey: key
        )
        guard case .restore(_, let cacheOffset) = plan.restore else { Issue.record("expected restore"); return }
        #expect(cacheOffset == 12)
        // prefillBaseOffset collapses the old skippedTokens / checkpointBaseOffset pair.
        #expect(plan.prefillBaseOffset == 12)
    }

    @Test func hitAtOffsetZeroFallsBackToCold() {
        let plan = PrefillPlanner.plan(
            boundaries: PrefillBoundaries(stablePrefixOffset: nil, lastMessageOffset: nil, lastUserOffset: nil),
            lookupResult: hit(snapshot(offset: 0), promptTokenCount: 40),
            checkpointPlan: [],
            promptTokenCount: 40,
            partitionKey: key
        )
        guard case .cold = plan.restore else { Issue.record("expected cold for offset 0"); return }
        #expect(plan.prefillBaseOffset == 0)
    }

    @Test func completeHitAtPromptLengthFallsBackToCold() {
        // A snapshot covering the entire prompt has no suffix to prefill; the
        // existing behavior runs cold rather than restoring a zero-length tail.
        let plan = PrefillPlanner.plan(
            boundaries: PrefillBoundaries(stablePrefixOffset: nil, lastMessageOffset: nil, lastUserOffset: nil),
            lookupResult: hit(snapshot(offset: 40), promptTokenCount: 40),
            checkpointPlan: [],
            promptTokenCount: 40,
            partitionKey: key
        )
        guard case .cold = plan.restore else { Issue.record("expected cold at prompt length"); return }
    }

    @Test func suffixFilterDropsCheckpointsAlreadyCoveredByTheRestore() {
        let plan = PrefillPlanner.plan(
            boundaries: PrefillBoundaries(stablePrefixOffset: 8, lastMessageOffset: nil, lastUserOffset: nil),
            lookupResult: hit(snapshot(offset: 12), promptTokenCount: 40),
            checkpointPlan: [(offset: 8, type: .system), (offset: 12, type: .branchPoint), (offset: 28, type: .branchPoint)],
            promptTokenCount: 40,
            partitionKey: key
        )
        // Only checkpoints strictly past the restored offset survive.
        #expect(offsets(plan.checkpointsToCapture) == [28])
    }

    // MARK: - plan(): transient checkpoint offsets

    @Test func transientOffsetsSurviveWhenPastBaseInsidePromptAndNotAlreadyPlanned() {
        let plan = PrefillPlanner.plan(
            boundaries: PrefillBoundaries(stablePrefixOffset: 8, lastMessageOffset: 30, lastUserOffset: 18),
            lookupResult: miss,
            checkpointPlan: [(offset: 8, type: .system)],
            promptTokenCount: 40,
            partitionKey: key
        )
        #expect(plan.transientCheckpointOffsets == [18, 30])
    }

    @Test func transientOffsetIsDroppedWhenItDuplicatesAPlannedCheckpoint() {
        let plan = PrefillPlanner.plan(
            boundaries: PrefillBoundaries(stablePrefixOffset: 8, lastMessageOffset: 20, lastUserOffset: nil),
            lookupResult: miss,
            checkpointPlan: [(offset: 8, type: .system), (offset: 20, type: .branchPoint)],
            promptTokenCount: 40,
            partitionKey: key
        )
        #expect(plan.transientCheckpointOffsets.isEmpty)
    }

    @Test func transientOffsetsOutsideTheSuffixWindowAreDropped() {
        // lastUser at/under the restore base is already cached; lastMessage at
        // or past the prompt length is out of range.
        let plan = PrefillPlanner.plan(
            boundaries: PrefillBoundaries(stablePrefixOffset: 8, lastMessageOffset: 40, lastUserOffset: 12),
            lookupResult: hit(snapshot(offset: 12), promptTokenCount: 40),
            checkpointPlan: [],
            promptTokenCount: 40,
            partitionKey: key
        )
        #expect(plan.transientCheckpointOffsets.isEmpty)
    }

    // MARK: - detectBoundaries(): stable prefix

    @Test func detectsStablePrefixAtTheSystemPlusToolsBoundary() throws {
        let tokenizer = FakeTokenizer()
        let conv = conversation(messages: [HTTPPrefixCacheMessage(role: .user, content: "hello there")])
        let tokens = try fullTokens(conv, tokenizer: tokenizer)

        let boundaries = try PrefillPlanner.detectBoundaries(
            conversation: conv,
            toolSpecs: nil,
            fullTokens: tokens,
            promptStartsThinking: true,
            tokenizer: tokenizer
        )

        // The two probes diverge at the user content, so the boundary is the
        // byte length of the rendered system block plus the user envelope head.
        let expected = "<|im_start|>system\n\(Self.system)<|im_end|>\n<|im_start|>user\n".utf8.count
        #expect(boundaries.stablePrefixOffset == expected)
    }

    @Test func emptySystemPromptHasNoStablePrefix() throws {
        let tokenizer = FakeTokenizer()
        let conv = conversation(systemPrompt: "", messages: [HTTPPrefixCacheMessage(role: .user, content: "hi")])
        let tokens = try fullTokens(conv, tokenizer: tokenizer)

        let boundaries = try PrefillPlanner.detectBoundaries(
            conversation: conv,
            toolSpecs: nil,
            fullTokens: tokens,
            promptStartsThinking: true,
            tokenizer: tokenizer
        )
        #expect(boundaries.stablePrefixOffset == nil)
    }

    // MARK: - detectBoundaries(): last-message (generation-prompt subtraction)

    @Test func lastMessageOffsetSubtractsTheThinkingGenerationPrompt() throws {
        var tokenizer = FakeTokenizer()
        tokenizer.promptStartsThinking = true
        let conv = conversation(messages: [HTTPPrefixCacheMessage(role: .user, content: "question")])
        let tokens = try fullTokens(conv, tokenizer: tokenizer)

        let boundaries = try PrefillPlanner.detectBoundaries(
            conversation: conv,
            toolSpecs: nil,
            fullTokens: tokens,
            promptStartsThinking: true,
            tokenizer: tokenizer
        )

        let genPromptBytes = FakeTokenizer.generationPrompt(thinking: true).utf8.count
        #expect(boundaries.lastMessageOffset == tokens.count - genPromptBytes)
    }

    @Test func lastMessageOffsetUsesTheNonThinkingPromptWhenNotThinking() throws {
        var tokenizer = FakeTokenizer()
        tokenizer.promptStartsThinking = false
        let conv = conversation(messages: [HTTPPrefixCacheMessage(role: .user, content: "question")])
        let tokens = try fullTokens(conv, tokenizer: tokenizer)

        let boundaries = try PrefillPlanner.detectBoundaries(
            conversation: conv,
            toolSpecs: nil,
            fullTokens: tokens,
            promptStartsThinking: false,
            tokenizer: tokenizer
        )

        let genPromptBytes = FakeTokenizer.generationPrompt(thinking: false).utf8.count
        #expect(boundaries.lastMessageOffset == tokens.count - genPromptBytes)
    }

    @Test func lastMessageOffsetIsNilWhenTheGenerationPromptSuffixDoesNotMatch() throws {
        // Tokens rendered with the thinking prompt, but detection told the turn
        // is non-thinking ⇒ the suffix won't match ⇒ no last-message boundary.
        var tokenizer = FakeTokenizer()
        tokenizer.promptStartsThinking = true
        let conv = conversation(messages: [HTTPPrefixCacheMessage(role: .user, content: "question")])
        let tokens = try fullTokens(conv, tokenizer: tokenizer)

        let boundaries = try PrefillPlanner.detectBoundaries(
            conversation: conv,
            toolSpecs: nil,
            fullTokens: tokens,
            promptStartsThinking: false,
            tokenizer: tokenizer
        )
        #expect(boundaries.lastMessageOffset == nil)
    }

    // MARK: - detectBoundaries(): last-user re-render

    @Test func lastUserOffsetReRendersUpToTheLastUserDroppingTrailingAssistant() throws {
        let tokenizer = FakeTokenizer()
        let conv = conversation(messages: [
            HTTPPrefixCacheMessage(role: .user, content: "first question"),
            HTTPPrefixCacheMessage(role: .assistant, content: "partial answer"),
        ])
        let tokens = try fullTokens(conv, tokenizer: tokenizer)

        let boundaries = try PrefillPlanner.detectBoundaries(
            conversation: conv,
            toolSpecs: nil,
            fullTokens: tokens,
            promptStartsThinking: true,
            tokenizer: tokenizer
        )

        // The re-render stops after the user turn — it excludes the trailing
        // assistant message that the full tokenization includes.
        let upToUser = "<|im_start|>system\n\(Self.system)<|im_end|>\n"
            + "<|im_start|>user\nfirst question<|im_end|>\n"
        #expect(boundaries.lastUserOffset == upToUser.utf8.count)
        let lastMessage = try #require(boundaries.lastMessageOffset)
        #expect(boundaries.lastUserOffset! < lastMessage)
    }

    @Test func lastUserOffsetIsNilWithoutAUserMessage() throws {
        let tokenizer = FakeTokenizer()
        let conv = conversation(messages: [HTTPPrefixCacheMessage(role: .assistant, content: "system-initiated")])
        let tokens = try fullTokens(conv, tokenizer: tokenizer)

        let boundaries = try PrefillPlanner.detectBoundaries(
            conversation: conv,
            toolSpecs: nil,
            fullTokens: tokens,
            promptStartsThinking: true,
            tokenizer: tokenizer
        )
        #expect(boundaries.lastUserOffset == nil)
    }
}
