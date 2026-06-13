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
    /// Typed feature-level skip: the last-user re-render could not be
    /// translated into key space, so only that boundary is dropped (the
    /// request, its lookup, and the other boundaries keep working).
    let lastUserTranslationFailure: CacheKeySpace.TranslationFailure?

    init(
        stablePrefixOffset: Int?,
        lastMessageOffset: Int?,
        lastUserOffset: Int?,
        lastUserTranslationFailure: CacheKeySpace.TranslationFailure? = nil
    ) {
        self.stablePrefixOffset = stablePrefixOffset
        self.lastMessageOffset = lastMessageOffset
        self.lastUserOffset = lastUserOffset
        self.lastUserTranslationFailure = lastUserTranslationFailure
    }
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
    /// state covers (the prefill base) and the **Position Anchor** rope delta
    /// for that offset (zero for image-free prefixes) — a cold plan cannot
    /// carry a delta at all.
    enum Restore: Sendable {
        case cold
        case restore(cacheOffset: Int, anchorDelta: Int)
    }

    let restore: Restore
    /// Checkpoints to capture during this prefill, already filtered to the
    /// suffix the restored snapshot does not already cover — and, on an
    /// image-bearing request, to offsets past the vendor-prepared image
    /// prefix (nothing inside `[0, minimumWarmOffset)` is capturable there).
    let checkpointsToCapture: [(offset: Int, type: HybridCacheSnapshot.CheckpointType)]
    /// The raw last-message / last-user boundary offsets, kept individually so
    /// the orchestrator can lift each transient boundary snapshot by name.
    let transientBoundaries: (lastMessage: Int?, lastUser: Int?)
    /// Token count of the full conversation; bounds the restore and transient
    /// decisions.
    let promptTokenCount: Int
    /// The request's smallest capturable/warm offset — the end of its last
    /// image run (`CacheKeySpace.minimumWarmOffset`), zero for text-only.
    let minimumWarmOffset: Int

    init(
        restore: Restore,
        checkpointsToCapture: [(offset: Int, type: HybridCacheSnapshot.CheckpointType)],
        transientBoundaries: (lastMessage: Int?, lastUser: Int?),
        promptTokenCount: Int,
        minimumWarmOffset: Int = 0
    ) {
        self.restore = restore
        self.checkpointsToCapture = checkpointsToCapture
        self.transientBoundaries = transientBoundaries
        self.promptTokenCount = promptTokenCount
        self.minimumWarmOffset = minimumWarmOffset
    }

    /// The number of leading tokens already covered by the restored cache —
    /// zero on a cold prefill. Single source for what were two values that
    /// could drift (`skippedTokens` and `checkpointBaseOffset`).
    var prefillBaseOffset: Int {
        if case .restore(let cacheOffset, _) = restore { return cacheOffset }
        return 0
    }

    /// The transient boundary offsets that survive: past the prefill base,
    /// past the request's image prefix, inside the prompt, and not already a
    /// planned checkpoint.
    var transientCheckpointOffsets: Set<Int> {
        var offsets: Set<Int> = []
        for offset in [transientBoundaries.lastMessage, transientBoundaries.lastUser].compactMap({ $0 })
        where offset > prefillBaseOffset
            && offset >= minimumWarmOffset
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
    /// All offsets are detected against `keySpace.keyPath` — the one token
    /// path the radix tree is driven with — so they cannot land in the wrong
    /// space on an image-bearing request.
    ///
    /// The MLXLMCommon `Tokenizer` protocol doesn't expose `addGenerationPrompt`,
    /// so the last-message boundary is found by encoding the known generation
    /// prompt string and subtracting it from the full token suffix.
    static func detectBoundaries(
        conversation: HTTPPrefixCacheConversation,
        toolSpecs: [ToolSpec]?,
        promptStartsThinking: Bool,
        tokenizer: any Tokenizer,
        keySpace: CacheKeySpace,
        renderContext: TemplateRenderContext = .canonical
    ) throws -> PrefillBoundaries {
        let fullTokens = keySpace.keyPath
        let stablePrefixOffset = try StablePrefixDetector.detect(
            systemPrompt: conversation.systemPrompt,
            toolSpecs: toolSpecs,
            additionalContext: renderContext.additionalContext(),
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

        // The last-user re-render is *render space* — one placeholder token
        // per image, unexpanded. Its length is only a valid offset into the
        // key path after translation through the **Cache Key Space**.
        // Translation failure skips just this boundary, typed.
        let lastUserOffset: Int?
        var lastUserTranslationFailure: CacheKeySpace.TranslationFailure?
        if let lastUserIndex = conversation.messages.lastIndex(where: { $0.role == .user }) {
            let userPrefixConversation = HTTPPrefixCacheConversation(
                systemPrompt: conversation.systemPrompt,
                messages: Array(conversation.messages[...lastUserIndex]),
                toolDefinitionsDigest: conversation.toolDefinitionsDigest,
                templateContextDigest: conversation.templateContextDigest
            )
            let renderTokens = try tokenizer.applyChatTemplate(
                messages: userPrefixConversation.promptMessages,
                tools: toolSpecs,
                additionalContext: renderContext.additionalContext(
                    merging: ["add_generation_prompt": false]
                )
            )
            switch keySpace.translatedLength(renderTokens: renderTokens) {
            case .success(let keyLength):
                lastUserOffset = keyLength
            case .failure(let failure):
                lastUserOffset = nil
                lastUserTranslationFailure = failure
            }
        } else {
            lastUserOffset = nil
        }

        return PrefillBoundaries(
            stablePrefixOffset: stablePrefixOffset,
            lastMessageOffset: lastMessageOffset,
            lastUserOffset: lastUserOffset,
            lastUserTranslationFailure: lastUserTranslationFailure
        )
    }

    /// Fold the resolved lookup, the checkpoint plan, and the detected
    /// boundaries into the **Prefill Plan**. Pure: a hit inside the prompt
    /// becomes a suffix restore; everything else runs cold.
    ///
    /// On an image-bearing request the **Cache Key Space** shapes the plan
    /// (ADR-0007 phase 2 — warm image-remainder continuation):
    /// - a hit *below* the last image run is now usable: the deepest valid
    ///   snapshot is restored and the remainder is continued *through* the
    ///   image, chunked. The `minimumWarmOffset` clamp survives only as the
    ///   no-valid-restore fallback (the cold `else`);
    /// - the sole restore guard is `positionAnchorDelta(upTo:) != nil`, which
    ///   rejects an offset that splits a placeholder run (a corrupt boundary);
    /// - a usable restore carries its **Position Anchor** rope delta;
    /// - checkpoints inside the continued image span
    ///   `[cacheOffset, minimumWarmOffset)` (and, on a cold plan, the
    ///   image prefix `[0, minimumWarmOffset)`) stay uncapturable — the span is
    ///   forwarded atomically and Mamba state cannot be rewound — so they are
    ///   dropped here; only the span's end boundary and the text tail capture.
    static func plan(
        boundaries: PrefillBoundaries,
        lookupResult: PrefixCacheManager.LookupResult,
        checkpointPlan: [(offset: Int, type: HybridCacheSnapshot.CheckpointType)],
        promptTokenCount: Int,
        keySpace: CacheKeySpace
    ) -> PrefillPlan {
        let minimumWarmOffset = keySpace.minimumWarmOffset
        let restore: PrefillPlan.Restore
        let checkpointsToCapture: [(offset: Int, type: HybridCacheSnapshot.CheckpointType)]
        if let snapshot = lookupResult.snapshot,
           snapshot.tokenOffset > 0,
           snapshot.tokenOffset < promptTokenCount,
           let anchorDelta = keySpace.positionAnchorDelta(upTo: snapshot.tokenOffset) {
            let cacheOffset = snapshot.tokenOffset
            restore = .restore(cacheOffset: cacheOffset, anchorDelta: anchorDelta)
            // Capture in the suffix the snapshot doesn't cover, minus the
            // continued image span: `> cacheOffset` (past the restore) and
            // `>= minimumWarmOffset` (past the atomically-forwarded image).
            // For a text-only or image-already-cached restore the second clause
            // is implied, so this matches the old `> cacheOffset` behavior.
            checkpointsToCapture = checkpointPlan.filter {
                $0.offset > cacheOffset && $0.offset >= minimumWarmOffset
            }
        } else {
            restore = .cold
            checkpointsToCapture = checkpointPlan.filter { $0.offset >= minimumWarmOffset }
        }

        return PrefillPlan(
            restore: restore,
            checkpointsToCapture: checkpointsToCapture,
            transientBoundaries: (boundaries.lastMessageOffset, boundaries.lastUserOffset),
            promptTokenCount: promptTokenCount,
            minimumWarmOffset: minimumWarmOffset
        )
    }
}
