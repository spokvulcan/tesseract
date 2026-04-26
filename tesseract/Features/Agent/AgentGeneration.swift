import Foundation
import MLXLMCommon

/// Produced by `SettingsManager.makeDFlashLoadConfig` and consumed by
/// `LLMActor.loadModel`. `nil` at any step means inference falls back to
/// the standard `TokenIterator`.
struct DFlashLoadConfig: Sendable {
    let draftDirectory: URL
    let blockSize: Int
}

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
    var repetitionPenalty: Float? = nil
    var repetitionContextSize: Int = 20
    var presencePenalty: Float? = nil
    var presenceContextSize: Int = 20
    var frequencyPenalty: Float? = nil
    var frequencyContextSize: Int = 20

    /// Thinking-loop safeguard. See ``ThinkingRepetitionDetector/Config``.
    /// Applies only when the model uses a `<think>` chat template (Qwen3/3.5 thinking).
    var thinkingSafeguard: ThinkingRepetitionDetector.Config = .init()

    /// Number of bits for KV cache quantization (4 or 8). nil disables quantization.
    var kvBits: Int? = 8
    /// Group size for KV cache quantization.
    var kvGroupSize: Int = 64
    /// Token chunk size for prompt prefill. Larger values improve throughput at
    /// the cost of higher per-step peak memory. Our Phase 3.2 benchmark on the
    /// production prefix-cache path found 1024 to be the fastest cold-prefill
    /// default that still keeps peak memory well below the larger 2048/4096
    /// settings, so that is the production default.
    var prefillStepSize: Int = 1024
    var triAttention: TriAttentionConfiguration = .v1Disabled

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

    /// Returns the recommended parameters for a given model ID.
    static func forModel(_ modelID: String) -> AgentGenerateParameters {
        if modelID.contains("opus-distill") { return .qwen3OpusDistill }
        if modelID.contains("thinking") { return .qwen3Thinking }
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
    /// - Parameter name: non-nil once the parser has scanned past the
    ///   first `"name":"X"` literal. `nil` before that point.
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
