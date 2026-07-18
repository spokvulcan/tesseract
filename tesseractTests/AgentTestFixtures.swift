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
import Testing

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

/// A `ChatSession` over the no-op agent and hermetic fixtures. Consolidates
/// the construction the chat-session and Mission Control suites built
/// identically, so a `ChatSession.init` change touches one place. The store
/// is the widened protocol seam: suites pass the in-memory fixture or the
/// real store over a temp directory as the test demands.
@MainActor
func makeChatSession(
    agent: Agent? = nil,
    store: any AgentConversationStoring = InMemoryAgentConversationStore(),
    arbiter: InMemoryInferenceArbiter = InMemoryInferenceArbiter(),
    restoreComposerDraft: @MainActor @escaping (String, [ImageAttachment]) -> Void = { _, _ in }
) -> Tesseract_Agent.ChatSession {
    Tesseract_Agent.ChatSession(
        agent: agent ?? makeNoOpAgent(modelID: "test-model"),
        conversationStore: store,
        arbiter: arbiter,
        restoreComposerDraft: restoreComposerDraft,
        liveMarkdownThrottle: .zero
    )
}

/// Runs a tool and returns its joined text result — the invocation shape
/// every tool suite needs. Consolidates the helper three companion suites
/// had each copied, so a `tool.execute` signature change touches one place.
func toolText(
    _ tool: AgentToolDefinition, _ args: [String: JSONValue]
) async throws -> String {
    let result = try await tool.execute("test-call", args, nil, nil)
    let texts = result.content.compactMap { block -> String? in
        if case .text(let text) = block { return text }
        return nil
    }
    guard !texts.isEmpty else {
        Issue.record("expected a text result")
        return ""
    }
    return texts.joined(separator: "\n")
}

/// A unique scratch directory — the defence against the scheme's parallel
/// twin runners colliding on shared paths. Callers own cleanup (or lean on
/// the OS reaping the temp directory).
func makeTempDir(_ label: String = "scratch") -> URL {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("\(label)-\(UUID().uuidString)", isDirectory: true)
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}

/// A hermetic `MemoryStore` over its own scratch directory. Consolidates the
/// helper the companion suites each built identically, so a
/// `MemoryStore(directory:)` change touches one place, not three.
func scratchStore() throws -> MemoryStore {
    try MemoryStore(directory: makeTempDir("scratch-memory"))
}

/// A hermetic flight recorder over its own scratch directory.
func scratchRecorder() -> CompanionFlightRecorder {
    CompanionFlightRecorder(directory: makeTempDir("scratch-flight"))
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

    static func toolCall(name: String, arguments: [String: JSONValue] = [:]) -> ToolCall {
        ToolCall(function: .init(name: name, arguments: arguments))
    }
}
