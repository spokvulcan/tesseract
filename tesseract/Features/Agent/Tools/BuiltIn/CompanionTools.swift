//
//  CompanionTools.swift
//  tesseract
//
//  The entity's hands on its own future and its delivery rungs (ADR-0040).
//  `book_wake` is how Jarvis grants himself a future turn — code enforces the
//  visible budget and the persistence, never the judgment. The delivery tools
//  (notify, speak) are rungs of the escalation palette; choosing one is his
//  call under his standing instructions, and every use lands in the flight
//  recorder with the turn that made it.
//
//  These register on the Companion's headless agent only — the interactive
//  chat needs no delivery rungs; the owner is already looking at it.
//

import Foundation

nonisolated struct CompanionToolError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

/// The ~2/day discretionary promise budget (#309) — enforced visibly at the
/// tool layer, revisable with wear.
nonisolated enum CompanionWakeBudget {
    static let promisesPerDay = 2
}

// MARK: - book_wake

nonisolated func createBookWakeTool(
    store: MemoryStore,
    recorder: CompanionFlightRecorder,
    context: CompanionTurnContext
) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "book_wake",
        label: "book wake",
        description: """
            Book yourself a future turn — this is how you exist in time. Classes: \
            'rhythm' (your daily beats: morning planning, midday pulse, evening \
            journal — you book your own day, usually at morning planning for the \
            pulse and evening, and at evening for tomorrow's shape), 'promise' (a \
            specific touchpoint you're committing to — "I'll ask about the dentist \
            Thursday, sir"; announced or announceable, quiet delivery, budgeted), \
            'followup' (a cheap self-check — "wake me in 40 minutes to see if he \
            started"), 'resummons' (an escalation repeat while a summons is \
            unanswered). Content must be stateable in one line — it is what the \
            fallback notification says if you cannot run. To move an existing wake \
            ("do it in an hour"), pass its id in 'reschedule'. A booked wake WILL \
            fire — that is the harness's guarantee and your must-fire obligation.
            """,
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "content": PropertySchema(
                    type: "string",
                    description: "One announceable line: what future-you needs to know."
                ),
                "at": PropertySchema(
                    type: "string",
                    description:
                        "Local time 'yyyy-MM-dd HH:mm' (e.g. '2026-07-17 09:10'). "
                        + "Use this or in_minutes."
                ),
                "in_minutes": PropertySchema(
                    type: "integer",
                    description: "Minutes from now. Use this or 'at'."
                ),
                "class": PropertySchema(
                    type: "string",
                    description: "Wake class (default promise).",
                    enumValues: ["promise", "rhythm", "followup", "resummons"]
                ),
                "summons": PropertySchema(
                    type: "boolean",
                    description:
                        "Spoken-summons rights. Only when the owner granted 'wake me for "
                        + "this', or for rhythm beats per your instructions."
                ),
                "reschedule": PropertySchema(
                    type: "string",
                    description: "Id of an existing booked wake to move instead of booking new."
                ),
            ],
            required: ["content"]
        ),
        execute: { _, argsJSON, _, _ in
            guard let content = ToolArgExtractor.string(argsJSON, key: "content"),
                !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                throw CompanionToolError(message: "book_wake requires 'content'")
            }
            let due: Date
            if let atRaw = ToolArgExtractor.string(argsJSON, key: "at") {
                guard let parsed = CompanionWakeTime.parse(atRaw) else {
                    throw CompanionToolError(
                        message: "Could not parse 'at' — use local 'yyyy-MM-dd HH:mm'.")
                }
                due = parsed
            } else if let minutes = ToolArgExtractor.int(argsJSON, key: "in_minutes") {
                due = Date().addingTimeInterval(TimeInterval(max(1, minutes)) * 60)
            } else {
                throw CompanionToolError(message: "book_wake needs 'at' or 'in_minutes'")
            }
            guard due > Date() else {
                throw CompanionToolError(message: "That time is in the past.")
            }

            // Reschedule path: move, never lose.
            if let rescheduleID = ToolArgExtractor.string(argsJSON, key: "reschedule")
                .flatMap(UUID.init(uuidString:))
            {
                guard var wake = try await store.wake(id: rescheduleID),
                    wake.state == .booked || wake.state == .fired
                else {
                    throw CompanionToolError(
                        message: "No open wake with that id — check the briefing's list.")
                }
                let oldDue = wake.due
                wake.due = due
                wake.content = content
                wake.state = .booked
                try await store.upsertWake(wake)
                await recorder.record(
                    "wake.rescheduled",
                    wakeID: wake.id,
                    turnID: context.turnID,
                    conversationID: context.conversationID,
                    snapshot: [
                        "from": CompanionWakeTime.format(oldDue),
                        "to": CompanionWakeTime.format(due),
                    ],
                    note: content
                )
                return .text("Moved to \(CompanionWakeTime.format(due)): \(content)")
            }

            let wakeClass =
                CompanionWakeClass(
                    rawValue: ToolArgExtractor.string(argsJSON, key: "class") ?? "promise")
                ?? .promise

            // The visible budget (#309): refusal is a tool result the entity
            // sees, never a silent swallow. Keyed to the day the promise
            // *lands* — booking ahead draws on that day's budget, not today's.
            if wakeClass == .promise {
                let landingDay = TrackingDay.key(for: due)
                let booked = try await store.promisesBooked(onDay: landingDay)
                if booked >= CompanionWakeBudget.promisesPerDay {
                    return .text(
                        "Promise budget spent for \(landingDay) "
                            + "(\(booked)/\(CompanionWakeBudget.promisesPerDay)). Fold it into "
                            + "an existing beat's agenda instead, or land it another day.")
                }
            }

            let wake = await CompanionWake(
                content: content,
                due: due,
                wakeClass: wakeClass,
                summonsGrant: ToolArgExtractor.bool(argsJSON, key: "summons") ?? false,
                conversationID: context.conversationID
            )
            try await store.upsertWake(wake)
            await recorder.record(
                "wake.booked",
                wakeID: wake.id,
                turnID: context.turnID,
                conversationID: context.conversationID,
                snapshot: [
                    "class": wakeClass.rawValue, "due": CompanionWakeTime.format(due),
                    "summons": wake.summonsGrant ? "granted" : "no",
                ],
                note: content
            )
            return .text(
                "Booked [\(wakeClass.rawValue)] \(CompanionWakeTime.format(due)): \(content) "
                    + "(id \(wake.id.uuidString.prefix(8)))")
        }
    )
}

// MARK: - Delivery rungs

/// `notify` and `speak` — thin typed doors over the loop's delivery plumbing,
/// which owns the notifier, the reaction mapping, and the recorder events.
nonisolated func createNotifyTool(
    deliver: @escaping @MainActor (_ title: String, _ body: String) async -> Void
) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "notify",
        label: "notify",
        description: """
            Post a quiet notification banner — the standard rung for promises and \
            anything that should wait politely in Notification Center. He can \
            engage, reply inline, or dismiss; you will see the reaction in your \
            flight log. One banner per turn is almost always enough.
            """,
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "title": PropertySchema(
                    type: "string", description: "Short title (e.g. 'Jarvis')."),
                "body": PropertySchema(
                    type: "string", description: "The line itself — your register, brief."),
            ],
            required: ["body"]
        ),
        execute: { _, argsJSON, _, _ in
            guard let body = ToolArgExtractor.string(argsJSON, key: "body"), !body.isEmpty else {
                throw CompanionToolError(message: "notify requires 'body'")
            }
            let title = ToolArgExtractor.string(argsJSON, key: "title") ?? "Jarvis"
            await deliver(title, body)
            return .text("Notification posted.")
        }
    )
}

nonisolated func createSpeakTool(
    deliver: @escaping @MainActor (_ text: String) -> Void
) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "speak",
        label: "speak",
        description: """
            Say a line out loud — the summons rung. Reserved for contract beats \
            and wakes with summons rights; promises deliver quietly unless he \
            granted "wake me for this". English, your register, one or two \
            sentences. Never greet an empty room: check the situation block for \
            presence first.
            """,
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "text": PropertySchema(
                    type: "string", description: "The words to speak, exactly.")
            ],
            required: ["text"]
        ),
        execute: { _, argsJSON, _, _ in
            guard let text = ToolArgExtractor.string(argsJSON, key: "text"), !text.isEmpty else {
                throw CompanionToolError(message: "speak requires 'text'")
            }
            await deliver(text)
            return .text("Spoken.")
        }
    )
}

// MARK: - Time parsing

nonisolated enum CompanionWakeTime {

    static func parse(_ raw: String, now: Date = Date()) -> Date? {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = .current
        formatter.dateFormat = "yyyy-MM-dd HH:mm"
        if let date = formatter.date(from: raw) { return date }
        // Time-only ("21:30") books the next occurrence of that wall time.
        formatter.dateFormat = "HH:mm"
        if let time = formatter.date(from: raw) {
            let parts = Calendar.current.dateComponents([.hour, .minute], from: time)
            return Calendar.current.nextDate(
                after: now, matching: parts, matchingPolicy: .nextTime)
        }
        return nil
    }

    static func format(_ date: Date) -> String {
        date.formatted(date: .abbreviated, time: .shortened)
    }
}
