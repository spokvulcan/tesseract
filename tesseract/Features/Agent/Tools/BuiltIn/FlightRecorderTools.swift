//
//  FlightRecorderTools.swift
//  tesseract
//
//  The model's two hands on the flight recorder (#326): a read path (the
//  record is app-owned — file tools can't reach it, and shouldn't), and
//  `log_feedback`, the one write door — testimony stamped `model-reported`
//  with the conversation it came from, auditable against the persisted
//  transcript. The model adds testimony; it never edits history.
//

import Foundation

nonisolated struct FlightRecorderToolError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

// MARK: - flight_log (read)

nonisolated func createFlightLogTool(recorder: CompanionFlightRecorder) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "flight_log",
        label: "flight log",
        description: """
            Read your own flight recorder — the app-written log of every wake, \
            delivery, reaction, and feedback event. This is your record of what you \
            actually did and how he actually responded: consult it when composing \
            (did yesterday's pulse get dismissed?), when he refers to a past \
            interaction, or when grounding the weekly review. Autobiographical \
            claims must be record-backed: if it isn't here or in memory, you don't \
            claim it.
            """,
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "days": PropertySchema(
                    type: "integer",
                    description: "How many days back to read (default 7)."
                ),
                "filter": PropertySchema(
                    type: "string",
                    description:
                        "Optional event prefix, e.g. 'wake', 'delivery', 'reaction', "
                        + "'feedback', 'turn', 'beat'."
                ),
                "limit": PropertySchema(
                    type: "integer",
                    description: "Max lines returned, newest kept (default 60)."
                ),
            ],
            required: []
        ),
        execute: { _, argsJSON, _, _ in
            let days = min(max(ToolArgExtractor.int(argsJSON, key: "days") ?? 7, 1), 90)
            let limit = min(max(ToolArgExtractor.int(argsJSON, key: "limit") ?? 60, 1), 200)
            let filter = ToolArgExtractor.string(argsJSON, key: "filter")

            let since = Date().addingTimeInterval(-Double(days) * 86_400)
            var records = recorder.records(since: since)
            if let filter, !filter.isEmpty {
                records = records.filter { $0.event.hasPrefix(filter) }
            }
            guard !records.isEmpty else {
                return .text("No flight-recorder events in the last \(days) day(s).")
            }
            let shown = records.suffix(limit)
            let lines = shown.map { record -> String in
                let when = Date(timeIntervalSince1970: record.ts)
                    .formatted(date: .abbreviated, time: .shortened)
                var parts = ["\(when)  \(record.event)"]
                if let note = record.note, !note.isEmpty { parts.append("— \(note)") }
                if let snapshot = record.snapshot, !snapshot.isEmpty {
                    let detail = snapshot.sorted { $0.key < $1.key }
                        .map { "\($0.key)=\($0.value)" }.joined(separator: " ")
                    parts.append("[\(detail)]")
                }
                if record.source == CompanionTraceSource.modelReported.rawValue {
                    parts.append("(self-reported)")
                }
                return parts.joined(separator: " ")
            }
            let header =
                records.count > limit
                ? "Last \(limit) of \(records.count) events:\n" : ""
            return .text(header + lines.joined(separator: "\n"))
        }
    )
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
                        "Optional: the wake/interaction id this reacts to, from flight_log."
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
