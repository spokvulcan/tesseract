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

    private func snapshot(offset: Int, type: HybridCacheSnapshot.CheckpointType = .system)
        -> HybridCacheSnapshot
    {
        let kv = KVCacheSimple()
        kv.state = [
            MLXArray.zeros([1, 1, max(offset, 1), 64]),
            MLXArray.zeros([1, 1, max(offset, 1), 64]),
        ]
        return HybridCacheSnapshot.capture(cache: [kv], offset: offset, type: type)!
    }

    private func hit(_ snapshot: HybridCacheSnapshot, promptTokenCount: Int)
        -> PrefixCacheManager.LookupResult
    {
        PrefixCacheManager.LookupResult(
            snapshot: snapshot,
            partitionKey: key,
            snapshotTokenOffset: snapshot.tokenOffset,
            sharedPrefixLength: snapshot.tokenOffset,
            reason: .hit(
                snapshotOffset: snapshot.tokenOffset, totalTokens: promptTokenCount,
                type: snapshot.checkpointType)
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

    private func offsets(_ plan: [(offset: Int, type: HybridCacheSnapshot.CheckpointType)]) -> [Int]
    {
        plan.map(\.offset)
    }

    // MARK: - plan(): restore vs cold

    @Test func missPlansAColdPrefillCoveringNothing() {
        let plan = PrefillPlanner.plan(
            boundaries: PrefillBoundaries(
                stablePrefixOffset: 8, lastMessageOffset: nil, lastUserOffset: nil),
            lookupResult: miss,
            checkpointPlan: [(offset: 8, type: .system), (offset: 20, type: .branchPoint)],
            promptTokenCount: 40,
            keySpace: .identity()
        )
        guard case .cold = plan.restore else { Issue.record("expected cold"); return }
        #expect(plan.prefillBaseOffset == 0)
        // Cold prefill keeps the whole checkpoint plan — nothing is already cached.
        #expect(offsets(plan.checkpointsToCapture) == [8, 20])
    }

    @Test func hitInsideThePromptPlansASuffixRestore() {
        let snap = snapshot(offset: 12)
        let plan = PrefillPlanner.plan(
            boundaries: PrefillBoundaries(
                stablePrefixOffset: 8, lastMessageOffset: nil, lastUserOffset: nil),
            lookupResult: hit(snap, promptTokenCount: 40),
            checkpointPlan: [(offset: 8, type: .system)],
            promptTokenCount: 40,
            keySpace: .identity()
        )
        guard case .restore(let cacheOffset, _) = plan.restore else {
            Issue.record("expected restore"); return
        }
        #expect(cacheOffset == 12)
        // prefillBaseOffset collapses the old skippedTokens / checkpointBaseOffset pair.
        #expect(plan.prefillBaseOffset == 12)
    }

    @Test func hitAtOffsetZeroFallsBackToCold() {
        let plan = PrefillPlanner.plan(
            boundaries: PrefillBoundaries(
                stablePrefixOffset: nil, lastMessageOffset: nil, lastUserOffset: nil),
            lookupResult: hit(snapshot(offset: 0), promptTokenCount: 40),
            checkpointPlan: [],
            promptTokenCount: 40,
            keySpace: .identity()
        )
        guard case .cold = plan.restore else { Issue.record("expected cold for offset 0"); return }
        #expect(plan.prefillBaseOffset == 0)
    }

    @Test func completeHitAtPromptLengthFallsBackToCold() {
        // A snapshot covering the entire prompt has no suffix to prefill; the
        // existing behavior runs cold rather than restoring a zero-length tail.
        let plan = PrefillPlanner.plan(
            boundaries: PrefillBoundaries(
                stablePrefixOffset: nil, lastMessageOffset: nil, lastUserOffset: nil),
            lookupResult: hit(snapshot(offset: 40), promptTokenCount: 40),
            checkpointPlan: [],
            promptTokenCount: 40,
            keySpace: .identity()
        )
        guard case .cold = plan.restore else {
            Issue.record("expected cold at prompt length"); return
        }
    }

    @Test func suffixFilterDropsCheckpointsAlreadyCoveredByTheRestore() {
        let plan = PrefillPlanner.plan(
            boundaries: PrefillBoundaries(
                stablePrefixOffset: 8, lastMessageOffset: nil, lastUserOffset: nil),
            lookupResult: hit(snapshot(offset: 12), promptTokenCount: 40),
            checkpointPlan: [
                (offset: 8, type: .system), (offset: 12, type: .branchPoint),
                (offset: 28, type: .branchPoint),
            ],
            promptTokenCount: 40,
            keySpace: .identity()
        )
        // Only checkpoints strictly past the restored offset survive.
        #expect(offsets(plan.checkpointsToCapture) == [28])
    }

    // MARK: - plan(): image-bearing key space clamps

    /// A key space whose single image occupies `runRange` of the prepared
    /// sequence — the minimal fixture for the planner's warm-offset clamp.
    private func imageKeySpace(
        runRange: Range<Int>,
        positionSpan: Int,
        totalTokens: Int
    ) throws -> CacheKeySpace {
        let pad = 999
        var prepared = Array(repeating: 7, count: totalTokens)
        for index in runRange { prepared[index] = pad }
        return try CacheKeySpace.make(
            preparedTokens: prepared,
            images: [
                CacheKeySpace.RequestImage(
                    digest: #require(ImageDigest(rawDigest: Data(repeating: 0xAB, count: 32))),
                    positionSpan: positionSpan
                )
            ],
            placeholderIdentity: ImagePlaceholderIdentity(imagePadTokenId: pad)
        ).get()
    }

    @Test func hitSplittingTheImageRunDegradesToCold() throws {
        // Image run [10, 26): a snapshot at 12 splits the run — a corrupt
        // boundary (`positionAnchorDelta` is nil there), so the plan still runs
        // cold under phase 2 (the run-split guard, not the warm-offset clamp).
        let keySpace = try imageKeySpace(runRange: 10..<26, positionSpan: 8, totalTokens: 60)
        let plan = PrefillPlanner.plan(
            boundaries: PrefillBoundaries(
                stablePrefixOffset: 8, lastMessageOffset: nil, lastUserOffset: nil),
            lookupResult: hit(snapshot(offset: 12), promptTokenCount: 60),
            checkpointPlan: [(offset: 8, type: .system), (offset: 40, type: .branchPoint)],
            promptTokenCount: 60,
            keySpace: keySpace
        )
        guard case .cold = plan.restore else {
            Issue.record("expected cold for a run-splitting offset"); return
        }
        // The cold image plan also drops checkpoints inside the image prefix
        // [0, 26) — they are uncapturable there.
        #expect(offsets(plan.checkpointsToCapture) == [40])
        #expect(plan.minimumWarmOffset == 26)
    }

    @Test func hitBelowTheImageRunRestoresAndContinuesThroughIt() throws {
        // Phase 2 (ADR-0007): a snapshot at 8 — below the image run [10, 26) and
        // at a clean boundary — now restores and continues *through* the image,
        // instead of degrading to cold under the old `>= minimumWarmOffset`
        // clamp. The anchor delta is 0 (no image fully cached before 8).
        let keySpace = try imageKeySpace(runRange: 10..<26, positionSpan: 8, totalTokens: 60)
        let plan = PrefillPlanner.plan(
            boundaries: PrefillBoundaries(
                stablePrefixOffset: 8, lastMessageOffset: nil, lastUserOffset: nil),
            lookupResult: hit(snapshot(offset: 8), promptTokenCount: 60),
            checkpointPlan: [
                (offset: 8, type: .system),
                (offset: 20, type: .branchPoint),
                (offset: 40, type: .branchPoint),
            ],
            promptTokenCount: 60,
            keySpace: keySpace
        )
        guard case .restore(let cacheOffset, let anchorDelta) = plan.restore else {
            Issue.record("expected restore below the image run"); return
        }
        #expect(cacheOffset == 8)
        #expect(anchorDelta == 0)
        // Checkpoints inside the continued image span [8, 26) are uncapturable
        // (8 is the restore base, 20 is mid-span); only the text tail past
        // minimumWarmOffset survives.
        #expect(offsets(plan.checkpointsToCapture) == [40])
    }

    @Test func hitAtOrPastTheImageWarmOffsetRestoresWithItsAnchorDelta() throws {
        // positionSpan 8 over a 16-token run: delta = 8 − 16 = −8.
        let keySpace = try imageKeySpace(runRange: 10..<26, positionSpan: 8, totalTokens: 60)
        let plan = PrefillPlanner.plan(
            boundaries: PrefillBoundaries(
                stablePrefixOffset: 8, lastMessageOffset: nil, lastUserOffset: nil),
            lookupResult: hit(snapshot(offset: 30), promptTokenCount: 60),
            checkpointPlan: [],
            promptTokenCount: 60,
            keySpace: keySpace
        )
        guard case .restore(let cacheOffset, let anchorDelta) = plan.restore else {
            Issue.record("expected restore"); return
        }
        #expect(cacheOffset == 30)
        #expect(anchorDelta == -8)
    }

    @Test func transientOffsetsInsideTheImagePrefixAreDropped() throws {
        // lastUser inside the image prefix is uncapturable on the cold image
        // plan; lastMessage past the warm offset survives.
        let keySpace = try imageKeySpace(runRange: 10..<26, positionSpan: 8, totalTokens: 60)
        let plan = PrefillPlanner.plan(
            boundaries: PrefillBoundaries(
                stablePrefixOffset: nil, lastMessageOffset: 30, lastUserOffset: 8),
            lookupResult: miss,
            checkpointPlan: [],
            promptTokenCount: 60,
            keySpace: keySpace
        )
        #expect(plan.transientCheckpointOffsets == [30])
    }

    // MARK: - plan(): transient checkpoint offsets

    @Test func transientOffsetsSurviveWhenPastBaseInsidePromptAndNotAlreadyPlanned() {
        let plan = PrefillPlanner.plan(
            boundaries: PrefillBoundaries(
                stablePrefixOffset: 8, lastMessageOffset: 30, lastUserOffset: 18),
            lookupResult: miss,
            checkpointPlan: [(offset: 8, type: .system)],
            promptTokenCount: 40,
            keySpace: .identity()
        )
        #expect(plan.transientCheckpointOffsets == [18, 30])
    }

    @Test func transientOffsetIsDroppedWhenItDuplicatesAPlannedCheckpoint() {
        let plan = PrefillPlanner.plan(
            boundaries: PrefillBoundaries(
                stablePrefixOffset: 8, lastMessageOffset: 20, lastUserOffset: nil),
            lookupResult: miss,
            checkpointPlan: [(offset: 8, type: .system), (offset: 20, type: .branchPoint)],
            promptTokenCount: 40,
            keySpace: .identity()
        )
        #expect(plan.transientCheckpointOffsets.isEmpty)
    }

    @Test func transientOffsetsOutsideTheSuffixWindowAreDropped() {
        // lastUser at/under the restore base is already cached; lastMessage at
        // or past the prompt length is out of range.
        let plan = PrefillPlanner.plan(
            boundaries: PrefillBoundaries(
                stablePrefixOffset: 8, lastMessageOffset: 40, lastUserOffset: 12),
            lookupResult: hit(snapshot(offset: 12), promptTokenCount: 40),
            checkpointPlan: [],
            promptTokenCount: 40,
            keySpace: .identity()
        )
        #expect(plan.transientCheckpointOffsets.isEmpty)
    }

    // MARK: - detectBoundaries(): stable prefix

    @Test func detectsStablePrefixAtTheSystemPlusToolsBoundary() throws {
        let tokenizer = FakeChatMLTokenizer()
        let conv = conversation(messages: [
            HTTPPrefixCacheMessage(role: .user, content: "hello there")
        ])
        let tokens = try fullTokens(conv, tokenizer: tokenizer)

        let boundaries = try PrefillPlanner.detectBoundaries(
            conversation: conv,
            toolSpecs: nil,
            promptStartsThinking: true,
            tokenizer: tokenizer,
            keySpace: .identity(keyPath: tokens)
        )

        // The two probes diverge at the user content, so the boundary is the
        // byte length of the rendered system block plus the user envelope head.
        let expected = "<|im_start|>system\n\(Self.system)<|im_end|>\n<|im_start|>user\n".utf8.count
        #expect(boundaries.stablePrefixOffset == expected)
    }

    @Test func emptySystemPromptHasNoStablePrefix() throws {
        let tokenizer = FakeChatMLTokenizer()
        let conv = conversation(
            systemPrompt: "", messages: [HTTPPrefixCacheMessage(role: .user, content: "hi")])
        let tokens = try fullTokens(conv, tokenizer: tokenizer)

        let boundaries = try PrefillPlanner.detectBoundaries(
            conversation: conv,
            toolSpecs: nil,
            promptStartsThinking: true,
            tokenizer: tokenizer,
            keySpace: .identity(keyPath: tokens)
        )
        #expect(boundaries.stablePrefixOffset == nil)
    }

    // MARK: - detectBoundaries(): last-message (generation-prompt subtraction)

    @Test func lastMessageOffsetSubtractsTheThinkingGenerationPrompt() throws {
        var tokenizer = FakeChatMLTokenizer()
        tokenizer.promptStartsThinking = true
        let conv = conversation(messages: [HTTPPrefixCacheMessage(role: .user, content: "question")]
        )
        let tokens = try fullTokens(conv, tokenizer: tokenizer)

        let boundaries = try PrefillPlanner.detectBoundaries(
            conversation: conv,
            toolSpecs: nil,
            promptStartsThinking: true,
            tokenizer: tokenizer,
            keySpace: .identity(keyPath: tokens)
        )

        let genPromptBytes = FakeChatMLTokenizer.generationPrompt(thinking: true).utf8.count
        #expect(boundaries.lastMessageOffset == tokens.count - genPromptBytes)
    }

    @Test func lastMessageOffsetUsesTheNonThinkingPromptWhenNotThinking() throws {
        var tokenizer = FakeChatMLTokenizer()
        tokenizer.promptStartsThinking = false
        let conv = conversation(messages: [HTTPPrefixCacheMessage(role: .user, content: "question")]
        )
        let tokens = try fullTokens(conv, tokenizer: tokenizer)

        let boundaries = try PrefillPlanner.detectBoundaries(
            conversation: conv,
            toolSpecs: nil,
            promptStartsThinking: false,
            tokenizer: tokenizer,
            keySpace: .identity(keyPath: tokens)
        )

        let genPromptBytes = FakeChatMLTokenizer.generationPrompt(thinking: false).utf8.count
        #expect(boundaries.lastMessageOffset == tokens.count - genPromptBytes)
    }

    @Test func lastMessageOffsetIsNilWhenTheGenerationPromptSuffixDoesNotMatch() throws {
        // Tokens rendered with the thinking prompt, but detection told the turn
        // is non-thinking ⇒ the suffix won't match ⇒ no last-message boundary.
        var tokenizer = FakeChatMLTokenizer()
        tokenizer.promptStartsThinking = true
        let conv = conversation(messages: [HTTPPrefixCacheMessage(role: .user, content: "question")]
        )
        let tokens = try fullTokens(conv, tokenizer: tokenizer)

        let boundaries = try PrefillPlanner.detectBoundaries(
            conversation: conv,
            toolSpecs: nil,
            promptStartsThinking: false,
            tokenizer: tokenizer,
            keySpace: .identity(keyPath: tokens)
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
            promptStartsThinking: true,
            tokenizer: tokenizer,
            keySpace: .identity(keyPath: tokens)
        )

        // The re-render stops after the user turn — it excludes the trailing
        // assistant message that the full tokenization includes.
        let upToUser =
            "<|im_start|>system\n\(Self.system)<|im_end|>\n"
            + "<|im_start|>user\nfirst question<|im_end|>\n"
        #expect(boundaries.lastUserOffset == upToUser.utf8.count)
        let lastMessage = try #require(boundaries.lastMessageOffset)
        #expect(boundaries.lastUserOffset! < lastMessage)
    }

    @Test func lastUserOffsetIsNilWithoutAUserMessage() throws {
        let tokenizer = FakeChatMLTokenizer()
        let conv = conversation(messages: [
            HTTPPrefixCacheMessage(role: .assistant, content: "system-initiated")
        ])
        let tokens = try fullTokens(conv, tokenizer: tokenizer)

        let boundaries = try PrefillPlanner.detectBoundaries(
            conversation: conv,
            toolSpecs: nil,
            promptStartsThinking: true,
            tokenizer: tokenizer,
            keySpace: .identity(keyPath: tokens)
        )
        #expect(boundaries.lastUserOffset == nil)
    }

    // MARK: - detectBoundaries(): key-space translation

    @Test func lastUserOffsetIsTranslatedThroughTheKeySpaceForImageRenders() throws {
        let tokenizer = FakeChatMLTokenizer()
        let userMessage = HTTPPrefixCacheMessage(
            role: .user,
            content: "what is in this image",
            images: [HTTPPrefixCacheImage(data: Data([0x01, 0x02]))]
        )
        let conv = conversation(messages: [
            userMessage,
            HTTPPrefixCacheMessage(role: .assistant, content: "a cat"),
        ])
        let keySpace = try FakeChatMLTokenizer.keySpace(for: conv, runLengths: [4])

        let boundaries = try PrefillPlanner.detectBoundaries(
            conversation: conv,
            toolSpecs: nil,
            promptStartsThinking: true,
            tokenizer: tokenizer,
            keySpace: keySpace
        )

        // The re-render up to the user turn carries one single-pad placeholder;
        // in key space that pad becomes the image's 4-token pseudo-expansion,
        // so the boundary lands 3 tokens further than the render length.
        let renderUpToUser = try tokenizer.applyChatTemplate(
            messages: conversation(messages: [userMessage]).promptMessages,
            tools: nil,
            additionalContext: ["add_generation_prompt": false]
        )
        #expect(boundaries.lastUserOffset == renderUpToUser.count + 3)
        #expect(boundaries.lastUserTranslationFailure == nil)
    }

    @Test func lastUserTranslationFailureSkipsOnlyTheLastUserBoundary() throws {
        let tokenizer = FakeChatMLTokenizer()
        let first = HTTPPrefixCacheMessage(
            role: .user, content: "first", images: [HTTPPrefixCacheImage(data: Data([0x01]))]
        )
        let second = HTTPPrefixCacheMessage(
            role: .user, content: "second", images: [HTTPPrefixCacheImage(data: Data([0x02]))]
        )
        let conv = conversation(messages: [first, second])
        // A one-image key space against a two-pad re-render: the request's
        // prepared sequence carried one image run (the second pad position
        // arrived render-only — the shape of a pad token typed as user text),
        // so the re-render's second pad has no image-table entry and
        // translation fails typed.
        let render = try fullTokens(conv, tokenizer: tokenizer)
        let pad = FakeChatMLTokenizer.imagePadTokenId
        var prepared: [Int] = []
        var expandedFirstPad = false
        for token in render {
            guard token == pad else {
                prepared.append(token)
                continue
            }
            if expandedFirstPad {
                prepared.append(3)
            } else {
                prepared += Array(repeating: pad, count: 4)
                expandedFirstPad = true
            }
        }
        let keySpace = try CacheKeySpace.make(
            preparedTokens: prepared,
            images: [CacheKeySpace.RequestImage(digest: first.images[0].digest, positionSpan: 4)],
            placeholderIdentity: ImagePlaceholderIdentity(
                imagePadTokenId: FakeChatMLTokenizer.imagePadTokenId
            )
        ).get()

        let boundaries = try PrefillPlanner.detectBoundaries(
            conversation: conv,
            toolSpecs: nil,
            promptStartsThinking: true,
            tokenizer: tokenizer,
            keySpace: keySpace
        )

        #expect(boundaries.lastUserOffset == nil)
        #expect(
            boundaries.lastUserTranslationFailure
                == .placeholderOccurrencesExceedImages(occurrences: 2, images: 1))
        // Only the last-user boundary drops — the others keep working.
        #expect(boundaries.stablePrefixOffset != nil)
        #expect(boundaries.lastMessageOffset != nil)
    }
}
