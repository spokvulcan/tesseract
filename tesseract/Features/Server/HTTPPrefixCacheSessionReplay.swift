import Foundation
import MLXLMCommon

nonisolated enum HTTPAssistantReasoningSource: String, Sendable {
    case client
    case sessionRecovered
    case missing
}

nonisolated struct HTTPAssistantReasoningRepair: Sendable {
    let messages: [OpenAI.ChatMessage]
    let clientCount: Int
    let sessionRecoveredCount: Int
    let missingCount: Int
}

actor HTTPPrefixCacheSessionReplayStore {
    private struct SessionEntry {
        var turnsBySignature: [HTTPPrefixCacheAssistantSignature: HTTPPrefixCacheMessage] = [:]
        var turnOrder: [HTTPPrefixCacheAssistantSignature] = []
    }

    private let maxSessions: Int
    private let maxTurnsPerSession: Int
    private var sessions: [String: SessionEntry] = [:]
    private var sessionOrder: [String] = []

    init(maxSessions: Int = 32, maxTurnsPerSession: Int = 256) {
        self.maxSessions = max(1, maxSessions)
        self.maxTurnsPerSession = max(1, maxTurnsPerSession)
    }

    func clear() {
        sessions.removeAll()
        sessionOrder.removeAll()
    }

    func repair(
        messages: [OpenAI.ChatMessage],
        sessionAffinity: String?
    ) -> HTTPAssistantReasoningRepair {
        let sessionAffinity = normalizedSessionAffinity(sessionAffinity)
        let storedTurns = sessionAffinity.flatMap { sessions[$0]?.turnsBySignature } ?? [:]

        var repairedMessages: [OpenAI.ChatMessage] = []
        repairedMessages.reserveCapacity(messages.count)

        var clientCount = 0
        var sessionRecoveredCount = 0
        var missingCount = 0

        for message in messages {
            guard message.role == .assistant else {
                repairedMessages.append(message)
                continue
            }

            if message.resolvedReasoningContent != nil {
                clientCount += 1
                repairedMessages.append(message)
                continue
            }

            if let signature = assistantSignature(for: message),
               let recovered = storedTurns[signature]?.reasoning {
                var repaired = message
                repaired.reasoning_content = recovered
                repaired.reasoning = nil
                sessionRecoveredCount += 1
                repairedMessages.append(repaired)
            } else {
                missingCount += 1
                repairedMessages.append(message)
            }
        }

        return HTTPAssistantReasoningRepair(
            messages: repairedMessages,
            clientCount: clientCount,
            sessionRecoveredCount: sessionRecoveredCount,
            missingCount: missingCount
        )
    }

    func record(
        sessionAffinity: String?,
        assistantMessage: HTTPPrefixCacheMessage
    ) {
        guard assistantMessage.role == .assistant,
              let signature = assistantMessage.assistantSignature,
              let sessionAffinity = normalizedSessionAffinity(sessionAffinity) else {
            return
        }

        var session = sessions[sessionAffinity] ?? SessionEntry()
        touchSession(sessionAffinity)

        session.turnsBySignature[signature] = assistantMessage
        session.turnOrder.removeAll { $0 == signature }
        session.turnOrder.append(signature)

        if session.turnOrder.count > maxTurnsPerSession {
            let overflow = session.turnOrder.count - maxTurnsPerSession
            let evicted = session.turnOrder.prefix(overflow)
            session.turnOrder.removeFirst(overflow)
            for signature in evicted {
                session.turnsBySignature.removeValue(forKey: signature)
            }
        }

        sessions[sessionAffinity] = session
    }

    private func assistantSignature(
        for message: OpenAI.ChatMessage
    ) -> HTTPPrefixCacheAssistantSignature? {
        guard message.role == .assistant else { return nil }

        let toolCalls = message.tool_calls?.compactMap { toolCall -> HTTPPrefixCacheToolCall? in
            guard let name = toolCall.function?.name else { return nil }
            return HTTPPrefixCacheToolCall(
                name: name,
                argumentsJSON: toolCall.function?.arguments ?? "{}"
            )
        } ?? []

        return HTTPPrefixCacheAssistantSignature(
            content: message.content?.textValue ?? "",
            toolCalls: toolCalls
        )
    }

    private func normalizedSessionAffinity(_ sessionAffinity: String?) -> String? {
        guard let sessionAffinity else { return nil }
        let trimmed = sessionAffinity.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private func touchSession(_ sessionAffinity: String) {
        sessionOrder.removeAll { $0 == sessionAffinity }
        sessionOrder.append(sessionAffinity)

        if sessionOrder.count > maxSessions {
            let overflow = sessionOrder.count - maxSessions
            let evicted = sessionOrder.prefix(overflow)
            sessionOrder.removeFirst(overflow)
            for session in evicted {
                sessions.removeValue(forKey: session)
            }
        }
    }
}
