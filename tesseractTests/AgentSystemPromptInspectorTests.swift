//
//  AgentSystemPromptInspectorTests.swift
//  tesseractTests
//
//  Tests the **System Prompt Inspector** at its own seam — no `Agent`. The
//  prompt source and the raw-prompt formatter are injected closures, so the
//  published trio and supersede-in-flight behaviour are exercised directly.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct AgentSystemPromptInspectorTests {

    /// A formatter whose calls suspend on a continuation, so a test can hold a
    /// fetch in-flight and resolve specific calls in any order.
    @MainActor
    private final class ControllableFormatter {
        private(set) var calls: [String] = []
        private var continuations: [CheckedContinuation<(text: String, tokenCount: Int), Error>] =
            []
        var pendingCount: Int { continuations.count }

        func format(_ systemPrompt: String, _ tools: [AgentToolDefinition]?) async throws -> (
            text: String, tokenCount: Int
        ) {
            calls.append(systemPrompt)
            return try await withCheckedThrowingContinuation { continuations.append($0) }
        }

        func resolve(call index: Int, text: String, tokenCount: Int) {
            continuations[index].resume(returning: (text, tokenCount))
        }
    }

    private func source(_ prompt: String)
        -> @MainActor () -> (systemPrompt: String, tools: [AgentToolDefinition])
    {
        { (prompt, []) }
    }

    // MARK: - Init

    @Test func initSeedsAssembledPromptFromSource() {
        let inspector = AgentSystemPromptInspector(promptSource: source("SYS"))
        #expect(inspector.assembledSystemPrompt == "SYS")
        #expect(inspector.rawChatMLPrompt == nil)
        #expect(inspector.systemPromptTokenCount == nil)
    }

    // MARK: - Published trio

    @Test func fetchPublishesRawPromptAndTokenCount() async throws {
        let formatter = ControllableFormatter()
        let inspector = AgentSystemPromptInspector(
            promptSource: source("SYS"),
            formatRawPrompt: { try await formatter.format($0, $1) }
        )

        inspector.fetchRawSystemPrompt()
        for _ in 0..<500 where formatter.pendingCount == 0 { await Task.yield() }

        formatter.resolve(call: 0, text: "RAW", tokenCount: 42)
        for _ in 0..<500 where inspector.rawChatMLPrompt == nil { await Task.yield() }

        #expect(inspector.rawChatMLPrompt == "RAW")
        #expect(inspector.systemPromptTokenCount == 42)
    }

    // MARK: - Supersede in-flight

    /// A second fetch supersedes the first: the first fetch's late completion is
    /// dropped, and the published trio reflects only the second.
    @Test func secondFetchSupersedesFirstInFlight() async throws {
        let formatter = ControllableFormatter()
        let inspector = AgentSystemPromptInspector(
            promptSource: source("SYS"),
            formatRawPrompt: { try await formatter.format($0, $1) }
        )

        inspector.fetchRawSystemPrompt()
        for _ in 0..<500 where formatter.pendingCount < 1 { await Task.yield() }

        // Supersede while the first fetch is still in-flight.
        inspector.fetchRawSystemPrompt()
        for _ in 0..<500 where formatter.pendingCount < 2 { await Task.yield() }

        // Resolve the SECOND fetch first — it wins.
        formatter.resolve(call: 1, text: "SECOND", tokenCount: 2)
        for _ in 0..<500 where inspector.rawChatMLPrompt == nil { await Task.yield() }
        #expect(inspector.rawChatMLPrompt == "SECOND")
        #expect(inspector.systemPromptTokenCount == 2)

        // The first fetch completes late — its result must be dropped.
        formatter.resolve(call: 0, text: "FIRST", tokenCount: 1)
        for _ in 0..<500 { await Task.yield() }
        #expect(inspector.rawChatMLPrompt == "SECOND")
        #expect(inspector.systemPromptTokenCount == 2)
    }

    // MARK: - Reset

    @Test func resetClearsRenderedPrompt() async throws {
        let formatter = ControllableFormatter()
        let inspector = AgentSystemPromptInspector(
            promptSource: source("SYS"),
            formatRawPrompt: { try await formatter.format($0, $1) }
        )

        inspector.fetchRawSystemPrompt()
        for _ in 0..<500 where formatter.pendingCount == 0 { await Task.yield() }
        formatter.resolve(call: 0, text: "RAW", tokenCount: 9)
        for _ in 0..<500 where inspector.rawChatMLPrompt == nil { await Task.yield() }

        inspector.reset()
        #expect(inspector.rawChatMLPrompt == nil)
        #expect(inspector.systemPromptTokenCount == nil)
    }
}
