import Foundation

// MARK: - ExtensionRunner

/// Central event dispatcher that fires events through all registered extensions.
/// Iterates extensions in registration order, catches and logs handler errors,
/// and aggregates results per the dispatch rules in spec E.4/E.5.
///
/// All handlers for an event always run (no early-exit). Result aggregation
/// happens after dispatch: first `.block` wins for tool calls, last
/// `.modifyToolResult` wins for tool results, `.modifyContext` transforms
/// are applied sequentially so each handler sees the previous handler's output.
@MainActor
final class ExtensionRunner {
    private let host: ExtensionHost
    private let contextFactory: @Sendable () -> any ExtensionContext

    init(host: ExtensionHost, contextFactory: @escaping @Sendable () -> any ExtensionContext) {
        self.host = host
        self.contextFactory = contextFactory
    }

    // MARK: - General dispatch

    /// Fire an event through all extensions, collecting results.
    /// For `.context` events, modifications are applied sequentially — each handler
    /// receives the payload updated with the previous handler's `.modifyContext` output.
    func fire(
        event: ExtensionEventType,
        payload: ExtensionEventPayload
    ) async -> [ExtensionEventResult] {
        var results: [ExtensionEventResult] = []
        var currentPayload = payload
        let context = contextFactory()

        for ext in host.registeredExtensions {
            guard let handlers = ext.handlers[event] else { continue }
            for handler in handlers {
                do {
                    if let result = try await handler.handle(currentPayload, context) {
                        results.append(result)

                        // Thread context modifications through so subsequent handlers
                        // see the updated message list, not the original.
                        // Only applies to .context events — other event types must not
                        // have their payload shape replaced by a buggy handler.
                        if event == .context, case .modifyContext(let updated) = result {
                            currentPayload = .context(messages: updated)
                        }
                    }
                } catch {
                    Log.agent.error(
                        "[ExtensionRunner] Handler error in '\(ext.path)' for \(event.rawValue): \(error)"
                    )
                }
            }
        }

        return results
    }

    // MARK: - Tool call hook (before execution)

    /// Fire a `tool_call` event before tool execution.
    /// All handlers run (audit/logging extensions always observe the event).
    /// Returns the first `.block` result if any extension blocks the call, otherwise nil.
    func fireToolCall(
        toolCallId: String,
        toolName: String,
        argsJSON: String
    ) async -> ExtensionEventResult? {
        let results = await fire(
            event: .toolCall,
            payload: .toolCall(toolCallId: toolCallId, toolName: toolName, argsJSON: argsJSON)
        )

        for result in results {
            if case .block = result {
                return result
            }
        }

        return nil
    }

    // MARK: - Tool result hook (after execution)

    /// Fire a `tool_result` event after tool execution.
    /// Returns the (possibly modified) tool result. Last `.modifyToolResult` wins.
    func fireToolResult(
        toolCallId: String,
        toolName: String,
        result: AgentToolResult
    ) async -> AgentToolResult {
        let results = await fire(
            event: .toolResult,
            payload: .toolResult(toolCallId: toolCallId, toolName: toolName, result: result)
        )

        var finalResult = result
        for r in results {
            if case .modifyToolResult(let modified) = r {
                finalResult = modified
            }
        }

        return finalResult
    }
}
