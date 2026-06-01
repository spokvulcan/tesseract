//
//  AgentSystemPromptInspector.swift
//  tesseract
//
//  The **System Prompt Inspector** module: the cancellable raw-prompt /
//  token-count transparency panel, carved out of `AgentCoordinator`. It owns the
//  published `assembledSystemPrompt` / `rawChatMLPrompt` / `systemPromptTokenCount`
//  trio and the async fetch that renders the assembled prompt into raw ChatML +
//  a token count, superseding any in-flight fetch.
//
//  Dependencies are injected closures — a read-only `promptSource` for the
//  current prompt + tools, and the `formatRawPrompt` formatter — so it tests with
//  no `Agent`. View-triggered, never event-driven: zero event-spine coupling.
//

import Foundation
import Observation
import os

@Observable @MainActor
final class AgentSystemPromptInspector {

    // MARK: - Observable State

    private(set) var assembledSystemPrompt: String = ""
    private(set) var rawChatMLPrompt: String?
    private(set) var systemPromptTokenCount: Int?

    // MARK: - Dependencies

    /// Read-only source of the current assembled prompt + active tools.
    private let promptSource: @MainActor () -> (systemPrompt: String, tools: [AgentToolDefinition])
    private let formatRawPrompt: (@MainActor (String, [AgentToolDefinition]?) async throws -> (text: String, tokenCount: Int))?

    @ObservationIgnored private var fetchTask: Task<Void, Never>?

    // MARK: - Init

    init(
        promptSource: @escaping @MainActor () -> (systemPrompt: String, tools: [AgentToolDefinition]),
        formatRawPrompt: (@MainActor (String, [AgentToolDefinition]?) async throws -> (text: String, tokenCount: Int))? = nil
    ) {
        self.promptSource = promptSource
        self.formatRawPrompt = formatRawPrompt
        self.assembledSystemPrompt = promptSource().systemPrompt
    }

    // MARK: - Fetch

    /// Refresh the assembled prompt and (re)render the raw ChatML + token count.
    /// Supersedes any in-flight fetch — a stale completion is dropped.
    func fetchRawSystemPrompt() {
        let source = promptSource()
        assembledSystemPrompt = source.systemPrompt
        Log.agent.info("fetchRawSystemPrompt — prompt length=\(self.assembledSystemPrompt.count)")

        guard let formatRawPrompt else {
            Log.agent.warning("fetchRawSystemPrompt — formatRawPrompt closure is nil")
            return
        }

        Log.agent.info("fetchRawSystemPrompt — calling closure with \(source.tools.count) tools")
        fetchTask?.cancel()
        fetchTask = Task {
            do {
                let result = try await formatRawPrompt(source.systemPrompt, source.tools)
                guard !Task.isCancelled else { return }
                Log.agent.info("fetchRawSystemPrompt — success, raw length=\(result.text.count), tokens=\(result.tokenCount)")
                rawChatMLPrompt = result.text
                systemPromptTokenCount = result.tokenCount
            } catch is CancellationError {
                Log.agent.debug("fetchRawSystemPrompt — cancelled")
            } catch {
                Log.agent.error("fetchRawSystemPrompt — error: \(error)")
            }
        }
    }

    /// Clear the rendered raw prompt + token count (e.g. on new conversation).
    func reset() {
        fetchTask?.cancel()
        rawChatMLPrompt = nil
        systemPromptTokenCount = nil
    }
}
