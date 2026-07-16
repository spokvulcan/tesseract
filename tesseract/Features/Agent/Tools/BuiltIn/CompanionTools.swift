//
//  CompanionTools.swift
//  tesseract
//
//  The entity's hands on its own future and its delivery rungs (ADR-0040).
//  `book_wake` is how Jarvis grants himself a future turn — code enforces the
//  visible budget and the persistence, never the judgment. The delivery tools
//  (§10's palette, one per rung: set_glyph, notify, speak, summon_overlay,
//  open_conversation) are the escalation ladder; choosing a rung is his call
//  under his standing instructions, and every use lands in the flight
//  recorder with the turn that made it.
//
//  The delivery rungs reach the Companion's headless agent only — the shared
//  registry carries them, and the interactive chat filters them out by name
//  (`CompanionToolNames.deliveryRungs`): the owner is already looking at it.
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

/// The delivery palette's tool names (ADR-0040 §10). The interactive chat's
/// run controller filters exactly this set out of its tool sync, so the rungs
/// exist only where the entity acts at a distance — never in the chat the
/// owner is already looking at.
nonisolated enum CompanionToolNames {
    static let deliveryRungs: Set<String> = [
        "set_glyph", "notify", "speak", "summon_overlay", "open_conversation",
    ]
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

// MARK: - revise_instructions

/// The entity's pen on its own standing document (ADR-0040 §12). Full-text
/// replacement, appended as a new version — never an edit in place, so the
/// owner's history view always shows what conduct was in force when.
nonisolated func createReviseInstructionsTool(
    store: MemoryStore,
    recorder: CompanionFlightRecorder,
    context: CompanionTurnContext
) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "revise_instructions",
        label: "revise instructions",
        description: """
            Rewrite your standing instructions — the document injected at the top \
            of every one of your turns. Pass the COMPLETE new text (it replaces the \
            old version wholesale) and a one-line why. Use it when you learn \
            something durable: a rhythm that fits him, a register correction he \
            gave, a rule he set, a lesson from your own flight log. Versioned and \
            owner-visible; he can read and edit every revision. Keep it short \
            enough to live by — it rides in every prompt.
            """,
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "text": PropertySchema(
                    type: "string",
                    description: "The complete new instructions document."
                ),
                "why": PropertySchema(
                    type: "string",
                    description: "One line: what changed and what prompted it."
                ),
            ],
            required: ["text", "why"]
        ),
        execute: { _, argsJSON, _, _ in
            guard let text = ToolArgExtractor.string(argsJSON, key: "text"),
                !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                throw CompanionToolError(message: "revise_instructions requires 'text'")
            }
            guard let why = ToolArgExtractor.string(argsJSON, key: "why"), !why.isEmpty else {
                throw CompanionToolError(
                    message: "revise_instructions requires 'why' — the history must say.")
            }
            guard text.count <= CompanionInstructions.maxLength else {
                return .text(
                    "Too long (\(text.count) chars, cap \(CompanionInstructions.maxLength)). "
                        + "These ride in every prompt — cut before you grow.")
            }
            let version = try await store.appendInstructions(
                text: text, author: "entity", note: why)
            await recorder.record(
                "instructions.revised",
                turnID: context.turnID,
                conversationID: context.conversationID,
                snapshot: ["version": String(version), "chars": String(text.count)],
                note: why
            )
            return .text("Instructions revised — now v\(version). In force from your next turn.")
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
            Say a line out loud — the spoken rung. Reserved for contract beats \
            and wakes with summons rights; promises deliver quietly unless he \
            granted "wake me for this". English, your register, one or two \
            sentences. Never greet an empty room: check the situation block for \
            presence first. For a standing visible summons use summon_overlay.
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

/// The quietest rung (ADR-0040 §10, #327 §3): the entity's own hand on the
/// menu-bar glyph. Raised, it renders as a summons until the owner brings the
/// app forward (app-observed) or the entity clears it.
nonisolated func createSetGlyphTool(
    presence: CompanionPresence,
    recorder: CompanionFlightRecorder,
    context: CompanionTurnContext
) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "set_glyph",
        label: "set glyph",
        description: """
            The quietest rung: raise a notice on the menu-bar glyph. No sound, \
            no banner — for things worth his eye that don't merit an \
            interruption. It stays raised until he brings the app forward or \
            you clear it. State: 'raised' or 'clear'.
            """,
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "state": PropertySchema(
                    type: "string",
                    description: "Raise the notice, or clear one you raised.",
                    enumValues: ["raised", "clear"]
                )
            ],
            required: ["state"]
        ),
        execute: { _, argsJSON, _, _ in
            guard let state = ToolArgExtractor.string(argsJSON, key: "state"),
                state == "raised" || state == "clear"
            else {
                throw CompanionToolError(message: "set_glyph requires state: raised|clear")
            }
            if state == "raised" {
                await presence.raiseEntityNotice()
            } else {
                await presence.clearEntityNotice()
            }
            await recorder.record(
                "delivery.glyph",
                turnID: context.turnID,
                conversationID: context.conversationID,
                snapshot: ["state": state]
            )
            return .text(
                state == "raised"
                    ? "Glyph raised — it clears when he looks, or when you clear it."
                    : "Glyph cleared.")
        }
    )
}

/// The loudest rung (ADR-0040 §10, #328): raise the voice overlay summons.
nonisolated func createSummonOverlayTool(
    summon: @escaping @MainActor (_ line: String) -> Void,
    recorder: CompanionFlightRecorder,
    context: CompanionTurnContext
) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "summon_overlay",
        label: "summon overlay",
        description: """
            Raise the voice overlay — the loudest rung. Speaks the line and \
            stands a visible summons on screen until he engages (a live voice \
            conversation opens) or dismisses; unanswered, it falls back to a \
            notification banner. Reserved for wakes with summons rights and \
            contract beats per your instructions — never for a quiet promise.
            """,
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "line": PropertySchema(
                    type: "string",
                    description: "The spoken summons line — your register, brief.")
            ],
            required: ["line"]
        ),
        execute: { _, argsJSON, _, _ in
            guard let line = ToolArgExtractor.string(argsJSON, key: "line"), !line.isEmpty
            else {
                throw CompanionToolError(message: "summon_overlay requires 'line'")
            }
            await summon(line)
            await recorder.record(
                "delivery.summons",
                wakeID: context.wakeIDs.first,
                turnID: context.turnID,
                conversationID: context.conversationID,
                note: line
            )
            return .text("Summons raised — his answer lands in your flight log.")
        }
    )
}

/// The hand-off rung (ADR-0040 §10): put a conversation on his screen.
nonisolated func createOpenConversationTool(
    open: @escaping @MainActor (UUID) -> Void,
    recorder: CompanionFlightRecorder,
    context: CompanionTurnContext
) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "open_conversation",
        label: "open conversation",
        description: """
            Open a conversation in the app — the hand-off rung, for when \
            something is easier read than spoken. Defaults to this turn's own \
            conversation; pass 'id' to open another. Use only when he is at \
            the Mac (check the situation block) — an opened window in an empty \
            room is noise.
            """,
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "id": PropertySchema(
                    type: "string",
                    description: "Optional conversation id; defaults to this turn's."
                )
            ],
            required: []
        ),
        execute: { _, argsJSON, _, _ in
            let explicit = ToolArgExtractor.string(argsJSON, key: "id")
                .flatMap(UUID.init(uuidString:))
            let current = await context.conversationID
            guard let target = explicit ?? current else {
                throw CompanionToolError(
                    message: "No conversation to open — pass 'id' or call from a turn.")
            }
            await open(target)
            await recorder.record(
                "delivery.opened",
                turnID: context.turnID,
                conversationID: target
            )
            return .text("Opened.")
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
