import Foundation
import MLXLMCommon

/// The tokenizer-affine boundary offsets a single request's prefill needs,
/// detected once in one tested place.
///
/// All three are token offsets into the full tokenized conversation:
/// - `stablePrefixOffset` — system + tools boundary (via `StablePrefixDetector`).
/// - `lastMessageOffset` — where the final history message ends, right before
///   the assistant-generation prompt (e.g. `<|im_start|>assistant\n<think>\n`).
/// - `lastUserOffset` — where the conversation ends when re-rendered up to and
///   including the last user message (no generation prompt). Stable across
///   think-block rewriting of older assistant turns.
nonisolated struct PrefillBoundaries: Sendable, Equatable {
    let stablePrefixOffset: Int?
    let lastMessageOffset: Int?
    let lastUserOffset: Int?
}

/// The pre-prefill decisions for one request, as a value: whether to restore a
/// cached prefix or run cold, which checkpoints to capture in the suffix, and
/// the boundary offsets the actor later lifts snapshots from. Carries offsets,
/// not post-prefill artifacts.
///
/// The read-side counterpart to **Snapshot Admission**: this is the plan the
/// orchestrator executes (slice + restore + iterate), where Snapshot Admission
/// is the write-side value it commits afterward.
nonisolated struct PrefillPlan: Sendable {
    /// The restore-vs-cold decision. `.restore` carries the offset its KV
    /// state covers (the prefill base).
    enum Restore: Sendable {
        case cold
        case restore(cacheOffset: Int)
    }

    let restore: Restore
    /// Checkpoints to capture during this prefill, already filtered to the
    /// suffix the restored snapshot does not already cover.
    let checkpointsToCapture: [(offset: Int, type: HybridCacheSnapshot.CheckpointType)]
    /// The raw last-message / last-user boundary offsets, kept individually so
    /// the orchestrator can lift each transient boundary snapshot by name.
    let transientBoundaries: (lastMessage: Int?, lastUser: Int?)
    /// Token count of the full conversation; bounds the restore and transient
    /// decisions.
    let promptTokenCount: Int

    /// The number of leading tokens already covered by the restored cache —
    /// zero on a cold prefill. Single source for what were two values that
    /// could drift (`skippedTokens` and `checkpointBaseOffset`).
    var prefillBaseOffset: Int {
        if case .restore(let cacheOffset) = restore { return cacheOffset }
        return 0
    }

    /// The transient boundary offsets that survive: past the prefill base,
    /// inside the prompt, and not already a planned checkpoint.
    var transientCheckpointOffsets: Set<Int> {
        var offsets: Set<Int> = []
        for offset in [transientBoundaries.lastMessage, transientBoundaries.lastUser].compactMap({ $0 })
        where offset > prefillBaseOffset
            && offset < promptTokenCount
            && !checkpointsToCapture.contains(where: { $0.offset == offset }) {
            offsets.insert(offset)
        }
        return offsets
    }
}

/// Produces the **Prefill Plan** for one request. Tokenizer-affine boundary
/// detection plus the pure restore/filter arithmetic — no GPU model, no live
/// KV cache. The orchestrator runs the resulting plan (slice → restore →
/// iterate → capture).
nonisolated enum PrefillPlanner {
    /// Detect the three prefill boundaries for one request. Depends only on a
    /// `Tokenizer`; runs the same two-probe stable-prefix detection plus the
    /// generation-prompt suffix subtraction and the last-user re-render.
    ///
    /// The MLXLMCommon `Tokenizer` protocol doesn't expose `addGenerationPrompt`,
    /// so the last-message boundary is found by encoding the known generation
    /// prompt string and subtracting it from the full token suffix.
    static func detectBoundaries(
        conversation: HTTPPrefixCacheConversation,
        toolSpecs: [ToolSpec]?,
        fullTokens: [Int],
        promptStartsThinking: Bool,
        tokenizer: any Tokenizer
    ) throws -> PrefillBoundaries {
        let stablePrefixOffset = try StablePrefixDetector.detect(
            systemPrompt: conversation.systemPrompt,
            toolSpecs: toolSpecs,
            fullTokens: fullTokens,
            tokenizer: tokenizer
        )

        let genPromptStr = promptStartsThinking
            ? "<|im_start|>assistant\n<think>\n"
            : "<|im_start|>assistant\n"
        let genPromptTokens = tokenizer.encode(text: genPromptStr, addSpecialTokens: false)
        let lastMessageOffset: Int?
        if genPromptTokens.count > 0,
           fullTokens.count > genPromptTokens.count,
           Array(fullTokens.suffix(genPromptTokens.count)).elementsEqual(genPromptTokens) {
            lastMessageOffset = fullTokens.count - genPromptTokens.count
        } else {
            lastMessageOffset = nil
        }

        let lastUserOffset: Int?
        if let lastUserIndex = conversation.messages.lastIndex(where: { $0.role == .user }) {
            let userPrefixConversation = HTTPPrefixCacheConversation(
                systemPrompt: conversation.systemPrompt,
                messages: Array(conversation.messages[...lastUserIndex]),
                toolDefinitionsDigest: conversation.toolDefinitionsDigest,
                templateContextDigest: conversation.templateContextDigest
            )
            lastUserOffset = try tokenizer.applyChatTemplate(
                messages: userPrefixConversation.promptMessages,
                tools: toolSpecs,
                additionalContext: ["add_generation_prompt": false]
            ).count
        } else {
            lastUserOffset = nil
        }

        return PrefillBoundaries(
            stablePrefixOffset: stablePrefixOffset,
            lastMessageOffset: lastMessageOffset,
            lastUserOffset: lastUserOffset
        )
    }

    /// Fold the resolved lookup, the checkpoint plan, and the detected
    /// boundaries into the **Prefill Plan**. Pure: a hit inside the prompt
    /// becomes a suffix restore; everything else runs cold.
    static func plan(
        boundaries: PrefillBoundaries,
        lookupResult: PrefixCacheManager.LookupResult,
        checkpointPlan: [(offset: Int, type: HybridCacheSnapshot.CheckpointType)],
        promptTokenCount: Int
    ) -> PrefillPlan {
        let restore: PrefillPlan.Restore
        let checkpointsToCapture: [(offset: Int, type: HybridCacheSnapshot.CheckpointType)]
        if let snapshot = lookupResult.snapshot,
           snapshot.tokenOffset > 0,
           snapshot.tokenOffset < promptTokenCount {
            let cacheOffset = snapshot.tokenOffset
            restore = .restore(cacheOffset: cacheOffset)
            // Only capture checkpoints in the SUFFIX the snapshot doesn't cover.
            checkpointsToCapture = checkpointPlan.filter { $0.offset > cacheOffset }
        } else {
            restore = .cold
            checkpointsToCapture = checkpointPlan
        }

        return PrefillPlan(
            restore: restore,
            checkpointsToCapture: checkpointsToCapture,
            transientBoundaries: (boundaries.lastMessageOffset, boundaries.lastUserOffset),
            promptTokenCount: promptTokenCount
        )
    }
}
