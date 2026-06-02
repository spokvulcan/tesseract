//
//  AgentTestFixtures.swift
//  tesseractTests
//
//  Shared fixtures for coordinator/agent tests. Consolidates the no-op `Agent`
//  that several suites built identically (only the model id differed), so a
//  change to the `Agent` / `AgentLoopConfig` init touches one place, not three.
//

import Foundation
import MLXLMCommon

@testable import Tesseract_Agent

/// A no-op `Agent` for coordinator tests: every config hook returns empty and
/// `generate` immediately finishes the stream — no model load, no I/O. Only the
/// model id varies between suites, so it is the sole parameter.
@MainActor
func makeNoOpAgent(modelID: String) -> Agent {
    let config = AgentLoopConfig(
        model: AgentModelRef(id: modelID),
        convertToLlm: { _ in [] },
        contextTransform: nil,
        getSteeringMessages: nil,
        getFollowUpMessages: nil
    )
    return Agent(
        config: config,
        systemPrompt: "test",
        tools: [],
        generate: { _, _, _, _ in AsyncThrowingStream { $0.finish() } }
    )
}

/// Shared generation-event fixtures: a terminal `AgentGeneration.Info` and a
/// minimal `ToolCall`. Consolidates the builders the accumulator and projection
/// suites had each copied, so a change to `AgentGeneration.Info`'s init (or
/// `ToolCall`) touches one place, not three.
enum GenerationFixtures {
    static func info(
        promptTokenCount: Int = 10,
        generationTokenCount: Int = 1,
        promptTime: TimeInterval = 0.1,
        generateTime: TimeInterval = 0.2,
        stopReason: GenerateStopReason = .stop
    ) -> AgentGeneration.Info {
        AgentGeneration.Info(
            promptTokenCount: promptTokenCount,
            generationTokenCount: generationTokenCount,
            promptTime: promptTime,
            generateTime: generateTime,
            stopReason: stopReason
        )
    }

    static func toolCall(name: String) -> ToolCall {
        ToolCall(function: .init(name: name, arguments: [:]))
    }
}
