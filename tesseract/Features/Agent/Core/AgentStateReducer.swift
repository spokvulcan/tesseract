//
//  AgentStateReducer.swift
//  tesseract
//
//  The **Agent State Reducer** — the single home for folding the `AgentEvent`
//  stream into run-level `AgentState`. A total fold, `reduce(_:into:)`, that
//  mutates the `@Observable AgentState` **in place** (ADR-0002: a value-type
//  swap looks like every property changed and coarsens Observation's
//  per-property invalidation).
//
//  It restores pi-mono's `processEvents` shape — *reduce all state, then notify
//  listeners* (the notify lives in `Agent.handleEvent`). It owns the **event
//  fold only**: the run-lifecycle envelope (the `.idle` transition and the
//  end-of-run clears of `streamMessage`/`pendingToolCalls`) stays in
//  `Agent.beginRun`/`finishRun`, exactly as pi-mono splits `processEvents` from
//  `finishRun`. See `CONTEXT.md` → "Agent state reduction".
//

import Foundation

enum AgentStateReducer {

    /// Fold one `AgentEvent` into `state`, mutating in place. **Total** over
    /// `AgentEvent`: adding a case forces a decision here (the compiler flags
    /// the gap).
    @MainActor
    static func reduce(_ event: AgentEvent, into state: AgentState) {
        switch event {
        case .agentStart:
            state.phase = .streaming

        case .agentEnd:
            // `finishRun` syncs `messages` from the final context; nothing here.
            break

        case .turnStart:
            break

        case .turnEnd(_, _, let contextMessages):
            // Authoritative sync — full replace from the loop's context snapshot,
            // which carries tool results the streaming path never emitted.
            state.messages = contextMessages.map { $0 as any AgentMessageProtocol }

        case .contextTransformStart(let reason):
            state.phase = .transformingContext(reason)

        case .contextTransformEnd(_, let didMutate, let messages):
            if didMutate, let messages {
                state.messages = messages.map { $0 as any AgentMessageProtocol }
            }
            // In-loop case: resume to streaming. Standalone `/compact` idle comes
            // from the run finishing under the lease, not from this event.
            state.phase = .streaming

        case .messageStart:
            break

        case .messageUpdate(let message, _):
            state.streamMessage = message

        case .messageEnd(let message):
            // pi-mono: clear the progressive stream on commit.
            state.streamMessage = nil
            // Commit on message_end; drop empty assistant turns from cancel/error
            // paths (`AssistantMessage.hasContent` — the same rule `runLoop`
            // folds against on persist).
            if let assistant = message as? AssistantMessage {
                guard assistant.hasContent else { break }
            }
            state.messages.append(message)

        case .malformedToolCall:
            break

        case .toolExecutionStart(let id, let name, _):
            state.pendingToolCalls.insert(id)
            state.phase = .executingTool(name)

        case .toolExecutionUpdate:
            break

        case .toolExecutionEnd(let id, _, _, _):
            state.pendingToolCalls.remove(id)
            if state.pendingToolCalls.isEmpty {
                state.phase = .streaming
            }
        }
    }
}
