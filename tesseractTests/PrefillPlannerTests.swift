import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import Tesseract_Agent

/// Behavior of the **Prefill Planner**: the boundary detection it owns and the
/// pure restore/filter decisions it folds into a **Prefill Plan**. Driven by a
/// byte-level fake tokenizer and constructed lookup values — no model, no actor.
@Suite struct PrefillPlannerTests {

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
        tokenizer: FakeChatMLTokenizer
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
            promptTokenCount: 40
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
            promptTokenCount: 40
        )
        guard case .restore(let cacheOffset) = plan.restore else { Issue.record("expected restore"); return }
        #expect(cacheOffset == 12)
        // prefillBaseOffset collapses the old skippedTokens / checkpointBaseOffset pair.
        #expect(plan.prefillBaseOffset == 12)
    }

    @Test func hitAtOffsetZeroFallsBackToCold() {
        let plan = PrefillPlanner.plan(
            boundaries: PrefillBoundaries(stablePrefixOffset: nil, lastMessageOffset: nil, lastUserOffset: nil),
            lookupResult: hit(snapshot(offset: 0), promptTokenCount: 40),
            checkpointPlan: [],
            promptTokenCount: 40
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
            promptTokenCount: 40
        )
        guard case .cold = plan.restore else { Issue.record("expected cold at prompt length"); return }
    }

    @Test func suffixFilterDropsCheckpointsAlreadyCoveredByTheRestore() {
        let plan = PrefillPlanner.plan(
            boundaries: PrefillBoundaries(stablePrefixOffset: 8, lastMessageOffset: nil, lastUserOffset: nil),
            lookupResult: hit(snapshot(offset: 12), promptTokenCount: 40),
            checkpointPlan: [(offset: 8, type: .system), (offset: 12, type: .branchPoint), (offset: 28, type: .branchPoint)],
            promptTokenCount: 40
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
            promptTokenCount: 40
        )
        #expect(plan.transientCheckpointOffsets == [18, 30])
    }

    @Test func transientOffsetIsDroppedWhenItDuplicatesAPlannedCheckpoint() {
        let plan = PrefillPlanner.plan(
            boundaries: PrefillBoundaries(stablePrefixOffset: 8, lastMessageOffset: 20, lastUserOffset: nil),
            lookupResult: miss,
            checkpointPlan: [(offset: 8, type: .system), (offset: 20, type: .branchPoint)],
            promptTokenCount: 40
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
            promptTokenCount: 40
        )
        #expect(plan.transientCheckpointOffsets.isEmpty)
    }

    // MARK: - detectBoundaries(): stable prefix

    @Test func detectsStablePrefixAtTheSystemPlusToolsBoundary() throws {
        let tokenizer = FakeChatMLTokenizer()
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
        let tokenizer = FakeChatMLTokenizer()
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
        var tokenizer = FakeChatMLTokenizer()
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

        let genPromptBytes = FakeChatMLTokenizer.generationPrompt(thinking: true).utf8.count
        #expect(boundaries.lastMessageOffset == tokens.count - genPromptBytes)
    }

    @Test func lastMessageOffsetUsesTheNonThinkingPromptWhenNotThinking() throws {
        var tokenizer = FakeChatMLTokenizer()
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

        let genPromptBytes = FakeChatMLTokenizer.generationPrompt(thinking: false).utf8.count
        #expect(boundaries.lastMessageOffset == tokens.count - genPromptBytes)
    }

    @Test func lastMessageOffsetIsNilWhenTheGenerationPromptSuffixDoesNotMatch() throws {
        // Tokens rendered with the thinking prompt, but detection told the turn
        // is non-thinking ⇒ the suffix won't match ⇒ no last-message boundary.
        var tokenizer = FakeChatMLTokenizer()
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
        let tokenizer = FakeChatMLTokenizer()
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
        let tokenizer = FakeChatMLTokenizer()
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
