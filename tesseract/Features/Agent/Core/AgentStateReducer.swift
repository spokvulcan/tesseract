//
//  AgentStateReducer.swift
//  tesseract
//
//  The **Agent State Reducer** — the single home for folding the `AgentEvent`
//  stream into the agent's committed message log. A total fold,
//  `reduce(_:into:)`, that mutates the `@Observable AgentState` **in place**
//  (ADR-0002: a value-type swap looks like every property changed and coarsens
//  Observation's per-property invalidation).
//
//  It owns the **message fold only**: run-presentation detail (live stream,
//  phase, pending tool calls) belongs to the **Chat Session**'s fold
//  (ADR-0024), and the run-lifecycle envelope (the busy bit) stays in
//  `Agent.beginRun`/`finishRun` — exactly as pi-mono splits `processEvents`
//  from `finishRun`. See `CONTEXT.md` → "Agent state reduction".
//

import Foundation

enum AgentStateReducer {

    /// Fold one `AgentEvent` into `state`, mutating in place. **Total** over
    /// `AgentEvent`: adding a case forces a decision here (the compiler flags
    /// the gap).
    @MainActor
    static func reduce(_ event: AgentEvent, into state: AgentState) {
        switch event {
        case .turnEnd(_, _, let contextMessages):
            // Authoritative sync — full replace from the loop's context snapshot,
            // which carries tool results the streaming path never emitted.
            state.messages = contextMessages.map { $0 as any AgentMessageProtocol }

        case .contextTransformEnd(_, let didMutate, let messages):
            if didMutate, let messages {
                state.messages = messages.map { $0 as any AgentMessageProtocol }
            }

        case .messageEnd(let message):
            // Commit on message_end; drop empty assistant turns from cancel/error
            // paths (`AssistantMessage.hasContent` — the same rule `runLoop`
            // folds against on persist).
            if let assistant = message as? AssistantMessage {
                guard assistant.hasContent else { break }
            }
            state.messages.append(message)

        case .agentStart, .agentEnd, .generationError, .turnStart,
            .contextTransformStart, .messageStart, .messageUpdate,
            .malformedToolCall, .toolExecutionStart, .toolExecutionUpdate,
            .toolExecutionEnd:
            // Run-presentation detail — the Chat Session's fold owns it
            // (ADR-0024); the agent's own state carries only the message log
            // and the envelope-owned busy bit.
            break
        }
    }
}
