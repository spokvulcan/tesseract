//
//  SamplingPreset.swift
//  tesseract
//

import Foundation

/// User-selectable sampling parameter preset for the agent.
///
/// Presets override only the six sampling fields that differ across
/// Qwen3.5/3.6 recommendations (temperature, topP, topK, minP,
/// presencePenalty, repetitionPenalty). Everything else — thinkingSafeguard,
/// kvBits, prefillStepSize, triAttention, maxTokens — continues to come from
/// the model-derived base returned by `AgentGenerateParameters.forModel`.
///
/// See: https://huggingface.co/Qwen/Qwen3.5-4B (Chat Completions usage).
enum SamplingPreset: String, CaseIterable, Identifiable, Sendable {
    /// Use the defaults bundled with the selected model (current behavior).
    case automatic
    /// Qwen3.5/3.6 thinking mode — general tasks.
    case qwenThinkingGeneral
    /// Qwen3.5/3.6 thinking mode — precise coding / web development.
    case qwenThinkingCoding
    /// Qwen3.5/3.6 instruct (non-thinking) mode — general chat.
    case qwenInstructGeneral
    /// Qwen3.5/3.6 instruct (non-thinking) mode — reasoning tasks.
    case qwenInstructReasoning

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .automatic: "Automatic (from model)"
        case .qwenThinkingGeneral: "Thinking – General"
        case .qwenThinkingCoding: "Thinking – Coding (WebDev)"
        case .qwenInstructGeneral: "Instruct – General"
        case .qwenInstructReasoning: "Instruct – Reasoning"
        }
    }

    var description: String {
        switch self {
        case .automatic:
            "Uses defaults bundled with the selected model."
        case .qwenThinkingGeneral:
            "temp 1.0, top_p 0.95, top_k 20, presence 1.5 — Qwen3.5/3.6 recommendation for thinking-mode chat."
        case .qwenThinkingCoding:
            "temp 0.6, top_p 0.95, top_k 20, presence 0.0 — precise coding and web development."
        case .qwenInstructGeneral:
            "temp 0.7, top_p 0.80, top_k 20, presence 1.5 — general chat in non-thinking mode."
        case .qwenInstructReasoning:
            "temp 1.0, top_p 0.95, top_k 20, presence 1.5 — reasoning tasks in non-thinking mode."
        }
    }

    /// Returns `base` unchanged for `.automatic`; otherwise overrides the six
    /// sampling fields (temperature, topP, topK, minP, presencePenalty,
    /// repetitionPenalty) and preserves everything else from `base`.
    func apply(to base: AgentGenerateParameters) -> AgentGenerateParameters {
        guard let overrides else { return base }
        var p = base
        p.temperature = overrides.temperature
        p.topP = overrides.topP
        p.topK = overrides.topK
        p.minP = overrides.minP
        p.presencePenalty = overrides.presencePenalty
        p.repetitionPenalty = overrides.repetitionPenalty
        return p
    }

    private struct Overrides {
        let temperature: Float
        let topP: Float
        let topK: Int
        let minP: Float
        let presencePenalty: Float?
        /// `nil` encodes "no penalty" (equivalent to the spec's rep=1.0).
        /// The HTTP path's `makeGenerateParameters` normalizes 1.0 → nil.
        let repetitionPenalty: Float?
    }

    private var overrides: Overrides? {
        switch self {
        case .automatic:
            return nil
        case .qwenThinkingGeneral:
            return Overrides(
                temperature: 1.0, topP: 0.95, topK: 20, minP: 0.0,
                presencePenalty: 1.5, repetitionPenalty: nil
            )
        case .qwenThinkingCoding:
            return Overrides(
                temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
                presencePenalty: 0.0, repetitionPenalty: nil
            )
        case .qwenInstructGeneral:
            return Overrides(
                temperature: 0.7, topP: 0.8, topK: 20, minP: 0.0,
                presencePenalty: 1.5, repetitionPenalty: nil
            )
        case .qwenInstructReasoning:
            return Overrides(
                temperature: 1.0, topP: 0.95, topK: 20, minP: 0.0,
                presencePenalty: 1.5, repetitionPenalty: nil
            )
        }
    }
}
