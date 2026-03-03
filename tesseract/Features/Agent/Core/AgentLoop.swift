import Foundation
import MLXLMCommon
import os

// MARK: - AgentContext

/// Authoritative conversation state threaded through the agent loop.
struct AgentContext: Sendable {
    var systemPrompt: String
    var messages: [any AgentMessageProtocol & Sendable]
    var tools: [AgentToolDefinition]?
}

// MARK: - LLMGenerateFunction

/// Closure type for LLM generation. Connected to AgentEngine in Epic 6.
typealias LLMGenerateFunction = @Sendable (
    _ systemPrompt: String,
    _ messages: [LLMMessage],
    _ tools: [AgentToolDefinition]?,
    _ signal: CancellationToken?
) -> AsyncThrowingStream<AgentGeneration, Error>

// MARK: - StopReason

/// Why the assistant stopped generating.
private enum StopReason: Sendable {
    case endOfTurn
    case cancelled
    case error(Error)
}

// MARK: - Entry Points

/// Start a new agent loop with the given prompts.
func agentLoop(
    prompts: [any AgentMessageProtocol & Sendable],
    context: inout AgentContext,
    config: AgentLoopConfig,
    generate: @escaping LLMGenerateFunction,
    signal: CancellationToken?,
    emit: @escaping @Sendable (AgentEvent) -> Void
) async {
    // 1. Push prompts into context
    for prompt in prompts {
        context.messages.append(prompt)
    }

    // 2. Emit agent start
    emit(.agentStart)

    // 3. Emit message lifecycle for each prompt
    for prompt in prompts {
        emit(.messageStart(message: prompt))
        emit(.messageEnd(message: prompt))
    }

    // 4. Run the double-loop
    await runLoop(
        context: &context,
        pendingMessages: [],
        config: config,
        generate: generate,
        signal: signal,
        emit: emit
    )
}

/// Continue from existing context without new prompts (retry/resume).
func agentLoopContinue(
    context: inout AgentContext,
    config: AgentLoopConfig,
    generate: @escaping LLMGenerateFunction,
    signal: CancellationToken?,
    emit: @escaping @Sendable (AgentEvent) -> Void
) async {
    emit(.agentStart)

    await runLoop(
        context: &context,
        pendingMessages: [],
        config: config,
        generate: generate,
        signal: signal,
        emit: emit
    )
}

// MARK: - Double Loop

/// The Pi double-loop: outer loop handles follow-ups, inner loop handles
/// tool calls and steering messages.
private func runLoop(
    context: inout AgentContext,
    pendingMessages initialPending: [any AgentMessageProtocol & Sendable],
    config: AgentLoopConfig,
    generate: @escaping LLMGenerateFunction,
    signal: CancellationToken?,
    emit: @escaping @Sendable (AgentEvent) -> Void
) async {
    var pendingMessages = initialPending
    let allNewMessages = MessageAccumulator()
    var isFirstTurn = true

    // OUTER LOOP: handles follow-ups
    outerLoop: while true {
        // INNER LOOP: handles tool calls and steering
        innerLoop: while true {
            if signal?.isCancelled == true { break outerLoop }

            // 1. Emit turnStart (skip for first turn if no pending)
            if !isFirstTurn || !pendingMessages.isEmpty {
                emit(.turnStart)
            }
            isFirstTurn = false

            // 2. Push pending messages into context
            if !pendingMessages.isEmpty {
                for msg in pendingMessages {
                    context.messages.append(msg)
                    allNewMessages.append(msg)
                    emit(.messageStart(message: msg))
                    emit(.messageEnd(message: msg))
                }
                pendingMessages.removeAll()
            }

            // 3. Stream assistant response
            let streamResult = await streamAssistantResponse(
                context: &context,
                config: config,
                generate: generate,
                signal: signal,
                emit: emit
            )

            let assistantMessage: AssistantMessage
            let stopReason: StopReason

            switch streamResult {
            case .success(let msg, let reason):
                assistantMessage = msg
                stopReason = reason
            }

            // Only persist the assistant message if it has actual content.
            // Zero-output failures (model errors before any tokens) should not
            // pollute conversation history with blank turns.
            let hasContent = !assistantMessage.content.isEmpty
                || (assistantMessage.thinking?.isEmpty == false)
                || !assistantMessage.toolCalls.isEmpty

            if hasContent {
                context.messages.append(assistantMessage)
                allNewMessages.append(assistantMessage)
            }

            // 4. Check for cancellation or error
            switch stopReason {
            case .cancelled:
                emit(.turnEnd(message: assistantMessage, toolResults: []))
                emit(.agentEnd(messages: allNewMessages.snapshot()))
                return
            case .error(let error):
                emit(.turnEnd(message: assistantMessage, toolResults: []))
                emit(.agentEnd(messages: allNewMessages.snapshot()))
                Log.agent.error("Generation error: \(error)")
                return
            case .endOfTurn:
                break
            }

            // 5. Extract tool calls
            let toolCalls = assistantMessage.toolCalls
            guard !toolCalls.isEmpty else {
                // No tool calls — end the inner loop
                emit(.turnEnd(message: assistantMessage, toolResults: []))
                break innerLoop
            }

            // 6. Execute tool calls
            let (toolResults, steeringMessages) = await executeToolCalls(
                toolCalls: toolCalls,
                tools: context.tools ?? [],
                context: &context,
                allNewMessages: allNewMessages,
                signal: signal,
                getSteeringMessages: config.getSteeringMessages,
                emit: emit
            )

            emit(.turnEnd(message: assistantMessage, toolResults: toolResults))

            // 8. Set pending = steering
            pendingMessages = steeringMessages

            // 9. Also poll for new steering
            if let getSteering = config.getSteeringMessages {
                let moreSteering = await getSteering()
                if !moreSteering.isEmpty {
                    pendingMessages.append(contentsOf: moreSteering)
                }
            }

            // Inner loop continues if there are tool calls or pending messages
            if toolCalls.isEmpty && pendingMessages.isEmpty {
                break innerLoop
            }
        }

        // 10. Check for follow-ups
        if let getFollowUps = config.getFollowUpMessages {
            let followUps = await getFollowUps()
            if !followUps.isEmpty {
                pendingMessages = followUps
                continue outerLoop
            }
        }

        // 11. No follow-ups — done
        break outerLoop
    }

    emit(.agentEnd(messages: allNewMessages.snapshot()))
}

// MARK: - Stream Assistant Response

/// Result of streaming an assistant response.
/// Always returns the (possibly partial) assistant message so it can be preserved.
private enum StreamResult: Sendable {
    case success(AssistantMessage, StopReason)
}

/// Calls the LLM with the current context, streams chunks, and builds the final
/// assistant message.
private func streamAssistantResponse(
    context: inout AgentContext,
    config: AgentLoopConfig,
    generate: @escaping LLMGenerateFunction,
    signal: CancellationToken?,
    emit: @escaping @Sendable (AgentEvent) -> Void
) async -> StreamResult {
    // a. Run context transform if configured
    if let ct = config.contextTransform {
        emit(.contextTransformStart(reason: ct.reason))
        let result = await ct.transform(context.messages, signal)
        if result.didMutate {
            context.messages = result.messages
        }
        emit(.contextTransformEnd(
            reason: ct.reason,
            didMutate: result.didMutate,
            messages: result.didMutate ? context.messages : nil
        ))
    }

    // b. Convert to LLM messages
    let llmMessages = config.convertToLlm(context.messages)

    // c. Start streaming generation
    let stream = generate(context.systemPrompt, llmMessages, context.tools, signal)

    // d-g. Process stream chunks and build assistant message
    var textContent = ""
    var thinkingContent: String?
    var toolCalls: [ToolCallInfo] = []
    var inThinking = false

    // Emit messageStart with a placeholder
    let placeholderMessage = AssistantMessage.create(content: "")
    emit(.messageStart(message: placeholderMessage))

    do {
        for try await generation in stream {
            if signal?.isCancelled == true {
                let msg = AssistantMessage.fromStream(
                    content: textContent, thinking: thinkingContent, toolCalls: toolCalls
                )
                emit(.messageEnd(message: msg))
                return .success(msg, .cancelled)
            }

            switch generation {
            case .text(let text):
                textContent += text
                let current = AssistantMessage.fromStream(
                    content: textContent, thinking: thinkingContent, toolCalls: toolCalls
                )
                emit(.messageUpdate(
                    message: current,
                    streamDelta: AssistantStreamDelta(
                        textDelta: text, thinkingDelta: nil, toolCallDelta: nil
                    )
                ))

            case .thinkStart:
                inThinking = true
                if thinkingContent == nil { thinkingContent = "" }

            case .thinking(let text):
                thinkingContent = (thinkingContent ?? "") + text
                let current = AssistantMessage.fromStream(
                    content: textContent, thinking: thinkingContent, toolCalls: toolCalls
                )
                emit(.messageUpdate(
                    message: current,
                    streamDelta: AssistantStreamDelta(
                        textDelta: nil, thinkingDelta: text, toolCallDelta: nil
                    )
                ))

            case .thinkEnd:
                inThinking = false

            case .toolCall(let call):
                let info = ToolCallInfo(
                    id: UUID().uuidString,
                    name: call.function.name,
                    argumentsJSON: encodeArguments(call.function.arguments)
                )
                toolCalls.append(info)
                let current = AssistantMessage.fromStream(
                    content: textContent, thinking: thinkingContent, toolCalls: toolCalls
                )
                emit(.messageUpdate(
                    message: current,
                    streamDelta: AssistantStreamDelta(
                        textDelta: nil, thinkingDelta: nil,
                        toolCallDelta: ToolCallDelta(
                            toolCallId: info.id, name: info.name, argumentsDelta: nil
                        )
                    )
                ))

            case .malformedToolCall(let raw):
                Log.agent.warning("Malformed tool call ignored: \(raw)")

            case .info:
                // Metrics — useful for logging but don't affect the message
                break
            }
        }
    } catch {
        let msg = AssistantMessage.fromStream(
            content: textContent, thinking: thinkingContent, toolCalls: toolCalls
        )
        emit(.messageEnd(message: msg))
        if signal?.isCancelled == true {
            return .success(msg, .cancelled)
        }
        // Preserve the partial message — runLoop will append it to context
        return .success(msg, .error(error))
    }

    let finalMessage = AssistantMessage.fromStream(
        content: textContent, thinking: thinkingContent, toolCalls: toolCalls
    )
    emit(.messageEnd(message: finalMessage))
    return .success(finalMessage, .endOfTurn)
}

// MARK: - Tool Execution

/// Execute tool calls sequentially. Steering messages can interrupt remaining tools.
private func executeToolCalls(
    toolCalls: [ToolCallInfo],
    tools: [AgentToolDefinition],
    context: inout AgentContext,
    allNewMessages: MessageAccumulator,
    signal: CancellationToken?,
    getSteeringMessages: (@Sendable () async -> [any AgentMessageProtocol])?,
    emit: @escaping @Sendable (AgentEvent) -> Void
) async -> (results: [ToolResultMessage], steering: [any AgentMessageProtocol & Sendable]) {
    var results: [ToolResultMessage] = []
    var steeringMessages: [any AgentMessageProtocol & Sendable] = []

    for (index, call) in toolCalls.enumerated() {
        // Check cancellation
        if signal?.isCancelled == true { break }

        // Look up tool
        guard let tool = tools.first(where: { $0.name == call.name }) else {
            let errorResult = ToolResultMessage.skipped(
                toolCallId: call.id, toolName: call.name,
                reason: "Unknown tool: \(call.name)"
            )
            results.append(errorResult)
            context.messages.append(errorResult)
            allNewMessages.append(errorResult)
            emit(.messageStart(message: errorResult))
            emit(.messageEnd(message: errorResult))
            continue
        }

        // Parse arguments
        let args: [String: JSONValue]
        if call.argumentsJSON.isEmpty || call.argumentsJSON == "{}" {
            args = [:]
        } else if let data = call.argumentsJSON.data(using: .utf8),
                  let parsed = try? JSONDecoder().decode([String: JSONValue].self, from: data)
        {
            args = parsed
        } else {
            let errorResult = ToolResultMessage.create(
                toolCallId: call.id, toolName: call.name,
                result: .error("Failed to parse tool arguments as JSON"),
                isError: true
            )
            results.append(errorResult)
            context.messages.append(errorResult)
            allNewMessages.append(errorResult)
            emit(.messageStart(message: errorResult))
            emit(.messageEnd(message: errorResult))
            continue
        }

        // Validate required parameters
        let missingParams = tool.parameterSchema.required.filter { args[$0] == nil }
        if !missingParams.isEmpty {
            let errorResult = ToolResultMessage.create(
                toolCallId: call.id, toolName: call.name,
                result: .error("Missing required parameters: \(missingParams.joined(separator: ", "))"),
                isError: true
            )
            results.append(errorResult)
            context.messages.append(errorResult)
            allNewMessages.append(errorResult)
            emit(.messageStart(message: errorResult))
            emit(.messageEnd(message: errorResult))
            continue
        }

        // Emit tool execution start
        emit(.toolExecutionStart(
            toolCallId: call.id, toolName: call.name,
            argsJSON: call.argumentsJSON
        ))

        // Create onUpdate callback
        let toolCallId = call.id
        let toolName = call.name
        let onUpdate: ToolProgressCallback = { progressResult in
            emit(.toolExecutionUpdate(
                toolCallId: toolCallId, toolName: toolName, result: progressResult
            ))
        }

        // Execute
        let toolResult: AgentToolResult
        let isError: Bool
        do {
            toolResult = try await tool.execute(call.id, args, signal, onUpdate)
            isError = false
        } catch {
            toolResult = .error("Tool execution failed: \(error.localizedDescription)")
            isError = true
        }

        // Emit tool execution end
        emit(.toolExecutionEnd(
            toolCallId: call.id, toolName: call.name,
            result: toolResult, isError: isError
        ))

        // Create result message
        let resultMessage = ToolResultMessage.create(
            toolCallId: call.id, toolName: call.name,
            result: toolResult, isError: isError
        )
        results.append(resultMessage)
        context.messages.append(resultMessage)
        allNewMessages.append(resultMessage)
        emit(.messageStart(message: resultMessage))
        emit(.messageEnd(message: resultMessage))

        // Check steering — skip remaining tools if steering arrived
        if let getSteering = getSteeringMessages {
            let steering = await getSteering()
            if !steering.isEmpty {
                steeringMessages = steering
                // Skip remaining tool calls
                for skippedCall in toolCalls[(index + 1)...] {
                    let skipped = ToolResultMessage.skipped(
                        toolCallId: skippedCall.id, toolName: skippedCall.name,
                        reason: "Skipped due to queued user message"
                    )
                    results.append(skipped)
                    context.messages.append(skipped)
                    allNewMessages.append(skipped)
                    emit(.messageStart(message: skipped))
                    emit(.messageEnd(message: skipped))
                }
                break
            }
        }
    }

    return (results, steeringMessages)
}

// MARK: - Helpers

/// Encode `[String: JSONValue]` arguments back to a JSON string.
private func encodeArguments(_ arguments: [String: JSONValue]) -> String {
    guard let data = try? JSONEncoder().encode(arguments),
          let json = String(data: data, encoding: .utf8)
    else {
        return "{}"
    }
    return json
}

/// Thread-safe accumulator for messages produced during a loop run.
/// Used to collect all new messages for the `.agentEnd` event.
private final class MessageAccumulator: @unchecked Sendable {
    private let lock = NSLock()
    private var messages: [any AgentMessageProtocol & Sendable] = []

    func append(_ message: any AgentMessageProtocol & Sendable) {
        lock.lock()
        messages.append(message)
        lock.unlock()
    }

    func snapshot() -> [any AgentMessageProtocol & Sendable] {
        lock.lock()
        defer { lock.unlock() }
        return messages
    }
}
