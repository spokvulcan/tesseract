//
//  FlightRecorderTools.swift
//  tesseract
//
//  The model's one hand on the flight recorder: `log_feedback`, the single
//  write door (#326, kept by the #369 palette review) — testimony stamped
//  `model-reported` with the conversation it came from, auditable against the
//  persisted transcript. The model adds testimony; it never edits history.
//
//  The read path (`flight_log`) died with ADR-0046: the standing conversation
//  is the entity's autobiographical record now, and the harness keeps the
//  trace mechanically — a tool to re-read it earned nothing.
//

import Foundation

nonisolated struct FlightRecorderToolError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

// MARK: - log_feedback (the one write door)

/// One list feeds both the schema's `enumValues` and the execute guard, so
/// the two can't drift.
private nonisolated let feedbackKinds = [
    "solicited", "spontaneous", "fabrication-flag", "annoyance", "dial-change",
]

nonisolated func createLogFeedbackTool(
    recorder: CompanionFlightRecorder,
    currentConversationID: @escaping @MainActor () -> UUID?
) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "log_feedback",
        label: "log feedback",
        description: """
            Record his reaction to something you did — verbatim, in his words. Use \
            it whenever he reacts to a beat, a callback, a promise, or your conduct: \
            the answer to a bookend solicitation ('kind' solicited), an unprompted \
            reaction ('spontaneous'), him flagging that you claimed something that \
            never happened ('fabrication-flag' — a defect, log it immediately), him \
            calling the pinging noise ('annoyance'), or him changing how firm he \
            wants you ('dial-change'). This is testimony added to your permanent \
            record; the weekly review replays it. Never paraphrase into something \
            kinder than what he said.
            """,
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "kind": PropertySchema(
                    type: "string",
                    description: "What kind of feedback this is.",
                    enumValues: feedbackKinds
                ),
                "verbatim": PropertySchema(
                    type: "string",
                    description: "His words, as close to verbatim as the transcript allows."
                ),
                "about": PropertySchema(
                    type: "string",
                    description:
                        "Optional: the wake/interaction id this reacts to, if you know it."
                ),
            ],
            required: ["kind", "verbatim"]
        ),
        execute: { _, argsJSON, _, _ in
            guard let kind = ToolArgExtractor.string(argsJSON, key: "kind"),
                feedbackKinds.contains(kind)
            else {
                throw FlightRecorderToolError(
                    message: "log_feedback requires kind: \(feedbackKinds.joined(separator: "|"))"
                )
            }
            guard let verbatim = ToolArgExtractor.string(argsJSON, key: "verbatim"),
                !verbatim.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                throw FlightRecorderToolError(
                    message: "log_feedback requires his words in 'verbatim'")
            }
            let about = ToolArgExtractor.string(argsJSON, key: "about")
            let conversationID = await currentConversationID()

            recorder.record(
                "feedback.\(kind)",
                source: .modelReported,
                wakeID: about.flatMap(UUID.init(uuidString:)),
                conversationID: conversationID,
                note: verbatim
            )
            return .text("Recorded (\(kind)): \"\(verbatim)\"")
        }
    )
}
