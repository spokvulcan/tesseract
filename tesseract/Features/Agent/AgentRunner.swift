import Foundation
import MLXLMCommon
import os

/// Events emitted by ``AgentRunner`` during a multi-round agent loop.
enum AgentRunnerEvent: Sendable {
    /// A text chunk for streaming display.
    case text(String)
    /// The model started a `<think>` block.
    case thinkStart
    /// A streaming chunk of thinking content.
    case thinking(String)
    /// The model finished its `<think>` block.
    case thinkEnd
    /// A tool is about to execute.
    case toolStart(name: String, arguments: [String: JSONValue])
    /// A tool finished executing.
    case toolResult(name: String, result: String)
    /// A malformed tool call was detected.
    case toolError(String)
    /// Generation metrics for one round.
    case info(AgentGeneration.Info)
    /// Raw ChatML prompt for a generation round (for benchmark transcripts).
    case roundStart(round: Int, rawPrompt: String, messageCount: Int)
    /// All new messages to append to history (emitted once at the end).
    case completed([AgentChatMessage])
}

/// Orchestrates the generate → execute tools → re-generate loop.
///
/// Sits between ``AgentCoordinator`` (UI state) and ``AgentEngine`` (inference).
/// Pure logic — takes messages in, yields ``AgentRunnerEvent``s out.
@MainActor
final class AgentRunner {

    private let engine: AgentEngine
    private let toolRegistry: ToolRegistry
    let maxToolRounds: Int

    private var runTask: Task<Void, Never>?

    init(engine: AgentEngine, toolRegistry: ToolRegistry, maxToolRounds: Int = 5) {
        self.engine = engine
        self.toolRegistry = toolRegistry
        self.maxToolRounds = maxToolRounds
    }

    /// Runs the agent loop: generate, optionally execute tool calls, re-generate.
    ///
    /// - Parameters:
    ///   - messages: Full conversation history (including system prompt).
    ///   - parameters: Generation parameters forwarded to the engine.
    /// - Returns: An async stream of ``AgentRunnerEvent``s.
    func run(
        messages: [AgentChatMessage],
        parameters: AgentGenerateParameters = .default,
        emitRawPrompts: Bool = false
    ) throws -> AsyncThrowingStream<AgentRunnerEvent, Error> {
        let (stream, continuation) = AsyncThrowingStream.makeStream(of: AgentRunnerEvent.self)

        let engine = self.engine
        let registry = self.toolRegistry
        let maxRounds = self.maxToolRounds
        let toolSpecs = registry.toolSpecs

        let task = Task { @MainActor [weak self] in
            do {
                var workingMessages = messages
                var newMessages: [AgentChatMessage] = []
                var executedKeys: Set<String> = []
                var consecutiveEmptyRounds = 0

                for round in 0..<maxRounds {
                    try Task.checkCancellation()

                    let stats = await engine.memoryStats()
                    Log.agent.info("Round \(round + 1)/\(maxRounds) — MLX active: \(String(format: "%.0f", stats.activeMB))MB, peak: \(String(format: "%.0f", stats.peakMB))MB")

                    if emitRawPrompts {
                        let rawPrompt = try await engine.formatRawPrompt(
                            messages: workingMessages, tools: toolSpecs
                        )
                        continuation.yield(.roundStart(
                            round: round + 1,
                            rawPrompt: rawPrompt,
                            messageCount: workingMessages.count
                        ))
                    }

                    let genStream = try engine.generate(
                        messages: workingMessages,
                        tools: toolSpecs,
                        parameters: parameters
                    )

                    var responseText = ""
                    var thinkingText = ""
                    var toolCalls: [ToolCall] = []
                    var malformedCalls: [String] = []
                    var hadThinkStart = false

                    for try await event in genStream {
                        switch event {
                        case .text(let chunk):
                            responseText += chunk
                            continuation.yield(.text(chunk))
                        case .thinkStart:
                            hadThinkStart = true
                            continuation.yield(.thinkStart)
                        case .thinking(let chunk):
                            thinkingText += chunk
                            continuation.yield(.thinking(chunk))
                        case .thinkEnd:
                            if !hadThinkStart {
                                // Model omitted <think> but included </think> —
                                // text accumulated so far is actually thinking
                                thinkingText = responseText
                                responseText = ""
                            }
                            continuation.yield(.thinkEnd)
                        case .toolCall(let call):
                            toolCalls.append(call)
                        case .malformedToolCall(let raw):
                            malformedCalls.append(raw)
                        case .info(let info):
                            continuation.yield(.info(info))
                        }
                    }

                    // Free stale MLX buffers from this generation round
                    await engine.clearMemoryCache()

                    let thinking = thinkingText.isEmpty ? nil : thinkingText

                    // No tool calls and no malformed calls — final response
                    if toolCalls.isEmpty && malformedCalls.isEmpty {
                        // Model produced no visible text or tools — stalled.
                        if responseText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                        {
                            consecutiveEmptyRounds += 1
                            Log.agent.info("Empty response round \(round + 1) (consecutive: \(consecutiveEmptyRounds)) — no text or tools, retrying")

                            // Scan thinking for a tool name the model wanted to call
                            let mentionedTool = registry.toolNames.first {
                                thinkingText.localizedCaseInsensitiveContains($0)
                            }

                            // Stage 2: If this is a repeated failure and the model mentioned
                            // a zero-arg tool, bypass generation entirely — construct and
                            // execute the tool call synthetically.
                            if consecutiveEmptyRounds >= 2,
                               let toolName = mentionedTool,
                               registry.hasNoRequiredParameters(toolName)
                            {
                                Log.agent.info("Synthetic injection: calling \(toolName) (zero-arg, \(consecutiveEmptyRounds) consecutive empties)")

                                let syntheticCall = ToolCall(
                                    function: .init(name: toolName, arguments: [:] as [String: any Sendable])
                                )

                                // Build assistant message with the synthetic tool call
                                let assistantContent = Self.reconstructAssistantMessage(
                                    text: "", toolCalls: [syntheticCall], thinking: thinking
                                )
                                workingMessages.append(.assistant(assistantContent))
                                newMessages.append(.assistant("", thinking: thinking, toolCalls: [syntheticCall]))

                                continuation.yield(.toolStart(name: toolName, arguments: [:]))

                                let result: String
                                do {
                                    result = try await registry.execute(call: syntheticCall)
                                } catch {
                                    result = "Error: \(error.localizedDescription)"
                                }

                                let callKey = Self.canonicalKey(syntheticCall)
                                executedKeys.insert(callKey)

                                continuation.yield(.toolResult(name: toolName, result: result))
                                Log.agent.info("Synthetic \(toolName) → \(result.prefix(200))")

                                let toolMsg = AgentChatMessage.tool(result)
                                workingMessages.append(toolMsg)
                                newMessages.append(toolMsg)

                                consecutiveEmptyRounds = 0
                                continue
                            }

                            // Stage 1: Insert a placeholder assistant message before the
                            // nudge to maintain proper ChatML alternation (user/assistant/user).
                            workingMessages.append(.assistant("I need to call a tool."))

                            if let tool = mentionedTool {
                                workingMessages.append(
                                    .user("You did not produce a tool call. You MUST respond with a <tool_call> block now. Call \(tool). Example format:\n<tool_call>\n{\"name\": \"\(tool)\", \"arguments\": {}}\n</tool_call>\nDo not deliberate — output the <tool_call> immediately.")
                                )
                            } else {
                                workingMessages.append(
                                    .user("You did not produce any output. You MUST either call a tool using <tool_call> tags or write a text response. Do it now.")
                                )
                            }
                            continue
                        }
                        consecutiveEmptyRounds = 0
                        newMessages.append(.assistant(responseText, thinking: thinking, toolCalls: toolCalls))
                        continuation.yield(.completed(newMessages))
                        continuation.finish()
                        return
                    }

                    // Process valid tool calls
                    if !toolCalls.isEmpty {
                        consecutiveEmptyRounds = 0
                        // Check for `respond` — the "final answer" tool.
                        // If present, extract its text as the response and end the loop.
                        // Other tools in the same response are still executed first.
                        let respondCall = toolCalls.first { $0.function.name == "respond" }
                        let dataToolCalls = toolCalls.filter { $0.function.name != "respond" }

                        // Reconstruct message with ALL tool calls (including respond) for history
                        let workingContent = Self.reconstructAssistantMessage(
                            text: responseText, toolCalls: toolCalls, thinking: thinking
                        )
                        workingMessages.append(.assistant(workingContent))
                        newMessages.append(.assistant(responseText, thinking: thinking, toolCalls: toolCalls))

                        for call in dataToolCalls {
                            try Task.checkCancellation()

                            // Within-turn dedup: skip if same tool+args already executed
                            let callKey = Self.canonicalKey(call)
                            if executedKeys.contains(callKey) {
                                Log.agent.info("Skipping duplicate: \(call.function.name)")
                                let toolMsg = AgentChatMessage.tool("[Already called — see result above]")
                                workingMessages.append(toolMsg)
                                newMessages.append(toolMsg)
                                continue
                            }

                            continuation.yield(.toolStart(name: call.function.name, arguments: call.function.arguments))

                            let result: String
                            do {
                                result = try await registry.execute(call: call)
                            } catch {
                                result = "Error: \(error.localizedDescription)"
                            }

                            executedKeys.insert(callKey)

                            continuation.yield(.toolResult(name: call.function.name, result: result))
                            Log.agent.info("Tool \(call.function.name) → \(result.prefix(200))")

                            let toolMsg = AgentChatMessage.tool(result)
                            workingMessages.append(toolMsg)
                            newMessages.append(toolMsg)
                        }

                        // If `respond` was called, use its text as the final response
                        if let respondCall {
                            let respondText = respondCall.function.arguments.string(for: "text") ?? responseText
                            if !respondText.isEmpty {
                                // Replace the last assistant message with the respond text
                                if let lastIdx = newMessages.lastIndex(where: { $0.role == .assistant }) {
                                    newMessages[lastIdx] = .assistant(respondText, thinking: thinking, toolCalls: toolCalls)
                                }
                            }
                            Log.agent.info("Respond tool → ending loop (\(respondText.count) chars)")
                            continuation.yield(.completed(newMessages))
                            continuation.finish()
                            return
                        }

                        continue
                    }

                    // Malformed tool calls — send error back so model can retry
                    if !malformedCalls.isEmpty {
                        workingMessages.append(.assistant(responseText))
                        newMessages.append(.assistant(responseText, thinking: thinking))

                        for raw in malformedCalls {
                            continuation.yield(.toolError(raw))
                            Log.agent.warning("Malformed tool call: \(raw.prefix(200))")

                            let errorMsg = AgentChatMessage.tool(
                                "Error: malformed tool call JSON. Please retry with valid JSON inside <tool_call> tags."
                            )
                            workingMessages.append(errorMsg)
                            newMessages.append(errorMsg)
                        }

                        continue
                    }
                }

                // Exhausted all rounds — emit what we have
                Log.agent.warning("Agent loop exhausted \(maxRounds) rounds")
                continuation.yield(.completed(newMessages))
                continuation.finish()
            } catch is CancellationError {
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }

            self?.runTask = nil
        }

        runTask = task
        continuation.onTermination = { _ in task.cancel() }

        return stream
    }

    /// Cancels any in-progress agent loop and underlying generation.
    func cancelGeneration() {
        runTask?.cancel()
        runTask = nil
        engine.cancelGeneration()
    }

    // MARK: - Private

    /// Produces a canonical string key for a tool call (name + sorted args) for dedup.
    private static func canonicalKey(_ call: ToolCall) -> String {
        let sortedArgs = call.function.arguments.sorted { $0.key < $1.key }
            .map { "\($0.key)=\(argString($0.value))" }
            .joined(separator: ",")
        return "\(call.function.name)(\(sortedArgs))"
    }

    private static func argString(_ value: JSONValue) -> String {
        switch value {
        case .string(let s): return s.lowercased()
        case .int(let i): return String(i)
        case .double(let d): return String(d)
        case .bool(let b): return String(b)
        case .null: return "null"
        default: return "?"
        }
    }

    /// Reconstructs the assistant message content with tool_call tags for conversation history.
    ///
    /// Thinking is intentionally **omitted** from working messages — the model only needs to see
    /// its prior actions (tool calls + text responses), not its prior reasoning. This prevents
    /// context bloat and removes the "example" of long deliberation that the model tends to copy.
    private static func reconstructAssistantMessage(
        text: String, toolCalls: [ToolCall], thinking: String? = nil
    ) -> String {
        var content = text
        for call in toolCalls {
            if let data = try? JSONEncoder().encode(call.function),
               let json = String(data: data, encoding: .utf8)
            {
                content += "\n<tool_call>\n\(json)\n</tool_call>"
            }
        }
        return content
    }
}
