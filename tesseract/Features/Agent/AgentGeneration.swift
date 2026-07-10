import Foundation
import MLXLMCommon

/// Human-readable label for `GenerateStopReason`, used in diagnostic logs so
/// operators can distinguish natural EOS from length-limit from cancellation
/// without decoding enum raw-values.
nonisolated func describeStopReason(_ reason: GenerateStopReason) -> String {
    switch reason {
    case .stop: return "stop(eos)"
    case .length: return "length(maxTokens)"
    case .cancelled: return "cancelled"
    }
}

/// Parameters controlling text generation behavior.
///
/// Defaults: temperature=0.6, top_p=0.95, repeat_penalty disabled, max_tokens=262144
/// (Qwen3.5 native context window — `max_position_embeddings` on every current
/// agent-catalog checkpoint).
struct AgentGenerateParameters: Sendable, Codable {
    var maxTokens: Int = 262_144
    var temperature: Float = 0.6
    var topP: Float = 0.95
    var topK: Int = 0
    var minP: Float = 0.0
    var repetitionPenalty: Float?
    var repetitionContextSize: Int = 20
    var presencePenalty: Float?
    var presenceContextSize: Int = 20
    var frequencyPenalty: Float?
    var frequencyContextSize: Int = 20

    /// Thinking-loop safeguard. See ``ThinkingRepetitionDetector/Config``.
    /// Applies only when the model uses a `<think>` chat template (Qwen3/3.5 thinking).
    var thinkingSafeguard: ThinkingRepetitionDetector.Config = .init()

    /// Number of bits for KV cache quantization (4 or 8). nil disables quantization.
    ///
    /// **Default is `nil` (unquantized).** `kvBits = 8` was the default until #252
    /// measured what it actually buys, out to the owner's real 200K regime:
    ///
    /// - **It saves zero *peak* memory, at every context.** Run peaks are identical
    ///   to the byte — 21.58 / 26.76 / 30.57 GB at 32K / 128K / 200K — because the
    ///   high-water mark is set during *prefill*, where the cache is still fp16.
    ///   `maybeQuantizeKVCache` (`KVCache.swift:1859`) converts only *after* the
    ///   step-0 forward, so quantization arrives after the peak has happened.
    /// - **It costs decode, and the cost grows with context**: −11.6% at 32K,
    ///   −40.1% at 128K, −38.2% at 200K.
    /// - **It is the numerically fragile option** (#233): 4× the chunk-shape noise
    ///   floor at 32K, and the only config where a benign prefill-chunk-size change
    ///   flips a greedy prediction.
    ///
    /// It does halve the cache *itself*, which peak memory hides. Only 10 of the 40
    /// layers carry a KV cache (the other 30 are GatedDeltaNet, with a fixed-size
    /// recurrent state), so fp16 KV costs 20 KiB/token: 0.67 / 2.68 / 4.10 GB at
    /// 32K / 128K / 200K, against 0.36 / 1.43 / 2.18 GB at 8 bits. Within a request
    /// that is never binding — 4.10 GB of KV sits far under the 30.57 GB prefill
    /// peak. **Across** requests it may be: `HybridCacheSnapshot` stores whatever
    /// cache type is live, so dense snapshots are ~1.9× larger and a fixed budget
    /// retains ~half as many prefixes. Snapshots partition on `kvBits`
    /// (`SnapshotManifest.partitionDigest`), so flipping this is always safe — it
    /// just strands the previous partition's snapshots. Decoupling the live dtype
    /// from the stored dtype is #259.
    var kvBits: Int?
    /// Group size for KV cache quantization.
    var kvGroupSize: Int = 64
    /// Token chunk size for prompt prefill.
    ///
    /// **1024, re-confirmed by measurement in #253** (the earlier "Phase 3.2"
    /// rationale predates the fixed harness). On PARO 35B at 32K, full prefill:
    /// **1006 tok/s at 1024**, 864 at 2048, 857 at 4096 — and peak memory rises
    /// 21.58 → 23.24 → 26.08 GB, because the unfused attention path materializes
    /// a `[1, 16, Lq, Lk]` score matrix.
    ///
    /// Counter-intuitively the *chunk loop itself* prefers larger chunks (#254:
    /// each MoE expert's `gather_qmm` gets `Lq × topK / numExperts` rows — only
    /// **32 rows at 1024**, running at 43% of peak GEMM). What kills the raise is
    /// `LLMModel.prepare`'s **tail**: it loops `while size > prefillStepSize`, so
    /// the `TokenIterator` swallows `promptTokens mod prefillStepSize` tokens in
    /// one un-pipelined forward — 141 tokens at 1024, but 3,213 at 4096. The tail
    /// grows with the step and erases the loop's gain. See #258.
    ///
    /// And at long context a raise is not a knob with a memory price — it is a
    /// cliff. At 128K, `prefillStepSize = 2048` measured **155.53 tok/s against
    /// 1024's 431.27** (2.8× slower) with peak 31.60 GB, because a single chunk's
    /// score matrix is 8.36 GB and the machine starts swapping. At 200K, 4096
    /// projects to ~59 GB on a 48 GB machine.
    ///
    /// Do not raise this before #258 lands, and never without re-measuring peak
    /// memory at 128K–200K.
    var prefillStepSize: Int = 1024

    static let `default` = AgentGenerateParameters()

    /// Qwen3-4B-Instruct-2507 recommended parameters for non-thinking mode.
    /// See: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
    static let qwen3: AgentGenerateParameters = {
        var p = AgentGenerateParameters(
            temperature: 0.7,
            topP: 0.8,
            topK: 20,
            presencePenalty: 1.5
        )
        p.thinkingSafeguard.enabled = false  // no `<think>` block on instruct variant
        return p
    }()

    /// Qwen3-4B-Thinking-2507 recommended parameters for thinking mode.
    /// No repetition penalty — it causes premature EOS in think blocks,
    /// making think-loops worse (model stops mid-think instead of transitioning to tool call).
    /// See: https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507
    static let qwen3Thinking = AgentGenerateParameters(
        temperature: 0.6,
        topP: 0.95,
        topK: 20
    )

    /// Qwen3-4B distilled from Claude 4.5 Opus — no repetition penalty needed.
    /// The distilled model was trained to be well-behaved without guardrails.
    static let qwen3OpusDistill = AgentGenerateParameters(
        temperature: 0.6,
        topP: 0.95
    )

    /// Qwen3.5-4B recommended parameters.
    /// See: https://huggingface.co/Qwen/Qwen3.5-4B
    static let qwen35 = AgentGenerateParameters(
        temperature: 1.0,
        topP: 0.95,
        topK: 20,
        presencePenalty: 1.5
    )

    /// Qwen3.6 thinking-mode defaults. No presencePenalty — inside `<think>`
    /// it drives paraphrase-with-changing-identifiers loops rather than
    /// preventing repetition.
    static let qwen36Thinking = AgentGenerateParameters(
        temperature: 0.6,
        topP: 0.95,
        topK: 20
    )

    /// Ornith 1.0 9B (DeepReinforce; Qwen3.5-dense `qwen3_5`, text-only).
    /// Vendor-recommended sampling. Values coincide with `.qwen36Thinking`;
    /// kept as its own preset so Ornith is isolated from future Qwen-preset
    /// changes. The thinking-loop safeguard stays at its default — it is inert
    /// unless the shipped chat template opens a `<think>` block.
    static let ornith9b = AgentGenerateParameters(
        temperature: 0.6,
        topP: 0.95,
        topK: 20
    )

    /// Ornith 1.0 35B (DeepReinforce; Qwen3.5-A3B MoE `qwen3_5_moe`,
    /// vision-capable). The vendor's Terminal-Bench recipe, taken verbatim.
    /// NOTE: this model ships the Qwen3.5 hybrid thinking template (it opens
    /// `<think>` by default), and the recipe sets `repetitionPenalty = 1.05` —
    /// exactly what ``qwen3Thinking``'s comment warns causes premature EOS in
    /// think blocks. Honored here per an explicit decision; the thinking-loop
    /// safeguard stays armed as the backstop. Revisit if generation truncates
    /// mid-think.
    static let ornith35b = AgentGenerateParameters(
        temperature: 1.0,
        topP: 1.0,
        topK: 40,
        minP: 0.01,
        repetitionPenalty: 1.05
    )

    /// Returns the recommended parameters for a given model ID.
    static func forModel(_ modelID: String) -> AgentGenerateParameters {
        if modelID.contains("opus-distill") { return .qwen3OpusDistill }
        if modelID.contains("thinking") { return .qwen3Thinking }
        if modelID.hasPrefix("ornith-9b") { return .ornith9b }
        if modelID.hasPrefix("ornith-35b") { return .ornith35b }
        if modelID.hasPrefix("qwen3.5") { return .qwen35 }
        if modelID.hasPrefix("qwen3.6") { return .qwen36Thinking }
        if modelID.hasPrefix("qwen3") { return .qwen3 }
        return .default
    }

    /// Emit a warning when sampling configuration is known to elevate
    /// thinking-loop risk on the active preset. Called from both HTTP
    /// prefix-cache and fallback paths right before starting generation.
    nonisolated func warnIfThinkingLoopRiskElevated(startsThinking: Bool) {
        guard thinkingSafeguard.enabled, startsThinking, temperature < 0.3 else {
            return
        }
        Log.agent.warning(
            "temperature=\(temperature) on thinking-capable model — Qwen docs "
                + "advise against greedy decoding in thinking mode; loop risk elevated."
        )
    }
}

/// Events emitted during streaming text generation.
nonisolated enum AgentGeneration: Sendable {
    /// A chunk of decoded text from the model.
    case text(String)

    /// A parsed tool call extracted from `<tool_call>` tags.
    case toolCall(ToolCall)

    /// A `<tool_call>` tag was found but contained malformed JSON.
    /// The associated string is the raw content between the tags.
    case malformedToolCall(String)

    /// In-flight chunk of tool-call body text observed inside
    /// `<tool_call>…</tool_call>` before the closing tag. Consumers that
    /// want to surface live tool-call arguments (the in-app Requests log)
    /// append `argumentsDelta` to the most recent "building" span for this
    /// tool call. The authoritative `.toolCall` / `.malformedToolCall`
    /// event still fires once on close with the parsed payload and should
    /// replace the building span with the finalized one.
    /// - Parameter name: non-nil once the producer's name-lock fires
    ///   (`ToolCallNameLock`: the first complete name literal, JSON or XML
    ///   dialect). `nil` before that point.
    /// - Parameter argumentsDelta: append-only raw text added to the parser
    ///   buffer on this chunk. Not parsed JSON.
    case toolCallDelta(name: String?, argumentsDelta: String)

    /// The model started a `<think>` block.
    case thinkStart
    /// A streaming chunk of thinking content.
    case thinking(String)
    /// The model finished its `<think>` block.
    case thinkEnd
    /// Generation ended without `</think>` — reclassify thinking content as text.
    case thinkReclassify

    /// Thinking-loop safeguard fired: discard accumulated thinking and treat
    /// `safePrefix` as the canonical reasoning for this turn. Consumers that buffer
    /// `.thinking` chunks (CompletionHandler, Path-A `handle()`) must reset their
    /// accumulator to exactly `safePrefix` on receipt. Emitted only by the safeguard
    /// intervention flow, never by `ToolCallParser`.
    case thinkTruncate(safePrefix: String)

    /// Completion metrics emitted once generation finishes.
    case info(Info)

    nonisolated struct Info: Sendable {
        let promptTokenCount: Int
        let generationTokenCount: Int
        let promptTime: TimeInterval
        let generateTime: TimeInterval
        let stopReason: GenerateStopReason

        var tokensPerSecond: Double {
            guard generateTime > 0 else { return 0 }
            return Double(generationTokenCount) / generateTime
        }
    }

    /// Bridge from ``ToolCallParser/Event`` to ``AgentGeneration``.
    nonisolated init(parserEvent: ToolCallParser.Event) {
        switch parserEvent {
        case .text(let text): self = .text(text)
        case .toolCall(let call): self = .toolCall(call)
        case .malformedToolCall(let raw): self = .malformedToolCall(raw)
        case .thinkStart: self = .thinkStart
        case .thinking(let text): self = .thinking(text)
        case .thinkEnd: self = .thinkEnd
        case .thinkReclassify: self = .thinkReclassify
        case .toolCallDelta(let name, let argumentsDelta):
            self = .toolCallDelta(name: name, argumentsDelta: argumentsDelta)
        }
    }
}
