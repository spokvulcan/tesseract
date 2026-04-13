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

    /// Partition key for reasoning recovery.
    ///
    /// Combines `sessionAffinity` (client-provided) with `modelID` + `visionMode`
    /// — the pair that identifies the physical LLM slot in
    /// `InferenceArbiter.LoadedLLMState`. This prevents recovered reasoning
    /// content from bleeding across two different physical containers sharing
    /// the same client session, which can happen if a client alternates
    /// models on the HTTP path or if the user toggles Vision Mode between
    /// requests on the same session.
    private struct ReplayKey: Hashable, Sendable {
        let sessionAffinity: String
        let modelID: String
        let visionMode: Bool
    }

    private let maxSessions: Int
    private let maxTurnsPerSession: Int
    private var sessions: [ReplayKey: SessionEntry] = [:]
    private var sessionOrder: [ReplayKey] = []

    // Default `maxSessions` is sized to absorb the partition multiplier from
    // `(sessionAffinity, modelID, visionMode)`. With ~3 agent models × 2
    // vision modes, the effective per-affinity slot count is 64 / 6 ≈ 10
    // unique sessions. Memory delta is bounded by `maxTurnsPerSession`.
    init(maxSessions: Int = 64, maxTurnsPerSession: Int = 256) {
        self.maxSessions = max(1, maxSessions)
        self.maxTurnsPerSession = max(1, maxTurnsPerSession)
    }

    func clear() {
        sessions.removeAll()
        sessionOrder.removeAll()
    }

    func repair(
        messages: [OpenAI.ChatMessage],
        sessionAffinity: String?,
        modelID: String,
        visionMode: Bool
    ) -> HTTPAssistantReasoningRepair {
        let key = replayKey(
            sessionAffinity: sessionAffinity,
            modelID: modelID,
            visionMode: visionMode
        )
        let storedTurns = key.flatMap { sessions[$0]?.turnsBySignature } ?? [:]

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
        modelID: String,
        visionMode: Bool,
        assistantMessage: HTTPPrefixCacheMessage
    ) {
        guard assistantMessage.role == .assistant,
              let signature = assistantMessage.assistantSignature,
              let key = replayKey(
                sessionAffinity: sessionAffinity,
                modelID: modelID,
                visionMode: visionMode
              ) else {
            return
        }

        var session = sessions[key] ?? SessionEntry()
        touchSession(key)

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

        sessions[key] = session
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

    private func replayKey(
        sessionAffinity: String?,
        modelID: String,
        visionMode: Bool
    ) -> ReplayKey? {
        guard let sessionAffinity else { return nil }
        let trimmedAffinity = sessionAffinity.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedModel = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedAffinity.isEmpty, !trimmedModel.isEmpty else { return nil }
        return ReplayKey(
            sessionAffinity: trimmedAffinity,
            modelID: trimmedModel,
            visionMode: visionMode
        )
    }

    private func touchSession(_ key: ReplayKey) {
        sessionOrder.removeAll { $0 == key }
        sessionOrder.append(key)

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
