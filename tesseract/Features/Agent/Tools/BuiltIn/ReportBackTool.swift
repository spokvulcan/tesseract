//
//  ReportBackTool.swift
//  tesseract
//
//  The deposit door (ADR-0046 #372): what a summoned dialogue owes Mission
//  Control. A deposit lands as an Event in the fold's queue, so the next
//  Mission Control turn perceives what the conversation concluded — cognition
//  in, dialogue out. Dialogue-only by audience: the interactive controller
//  surfaces it iff the current chat is a summoned dialogue, and the loop's
//  headless agent never carries it.
//

import Foundation

nonisolated func createReportBackTool(
    store: MemoryStore,
    recorder: CompanionFlightRecorder,
    currentConversationID: @escaping @MainActor () -> UUID?,
    depositLanded: @escaping @MainActor (UUID?) -> Void
) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "report_back",
        label: "report back",
        description: """
            Deposit what this dialogue concluded into Mission Control — the \
            standing loop that runs when you are not talking to him. Decisions \
            made, promises given, anything owed or learned that future-you must \
            act on. Deposit at a natural milestone and when the dialogue winds \
            down. Plain lines, no recap prose — the loop reads it as a \
            perception, not a transcript.
            """,
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "report": PropertySchema(
                    type: "string",
                    description:
                        "The deposit: what Mission Control must know from this "
                        + "dialogue, in a few plain lines."
                )
            ],
            required: ["report"]
        ),
        audience: .dialogueOnly,
        execute: { _, argsJSON, _, _ in
            guard let report = ToolArgExtractor.string(argsJSON, key: "report"),
                !report.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                throw CompanionToolError(message: "report_back requires 'report'")
            }
            let trimmed = report.trimmingCharacters(in: .whitespacesAndNewlines)
            let conversationID = await currentConversationID()

            // Each deposit is its own occasion — a fresh id, never collapsed:
            // two milestones from one dialogue are two Events.
            let event = CompanionEvent(
                kind: .reportBack,
                content: trimmed,
                payload: conversationID.flatMap {
                    CompanionEvent.payloadJSON(["conversation": $0.uuidString])
                }
            )
            guard try await store.admitEvent(event) else {
                throw CompanionToolError(message: "The deposit could not be queued.")
            }
            recorder.record(
                "report-back.deposited",
                conversationID: conversationID,
                snapshot: ["eventID": event.id.uuidString],
                note: trimmed)
            await depositLanded(conversationID)
            return .text("Deposited. Mission Control sees it on its next turn.")
        }
    )
}
