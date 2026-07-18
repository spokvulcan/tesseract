//
//  CompanionTools.swift
//  tesseract
//
//  The entity's hands on its own future and its delivery rungs (ADR-0040,
//  lean palette ADR-0046 #369). The wake palette is book / revise / cancel —
//  how Jarvis grants himself a future turn and keeps that future honest
//  (#354's triple-booking class): code enforces the visible budget and the
//  persistence, never the judgment. The delivery tools (§10's ladder:
//  set_glyph, notify, speak, summon_overlay) are the escalation rungs;
//  choosing one is his call under his standing instructions, and every use
//  lands in the flight recorder with the turn that made it.
//
//  `open_conversation` died at the #369 build review: engaging a delivery
//  already opens its conversation mechanically (the reaction path), and the
//  standing conversation sits in the chat list — the rung had no job left.
//
//  The delivery rungs reach the Companion's headless agent only — the shared
//  registry carries them, declared `audience: .companionOnly`, and the
//  interactive chat's tool sync drops them: the owner is already looking at it.
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
            fallback notification says if you cannot run. A booked wake WILL \
            fire — that is the harness's guarantee and your must-fire obligation. \
            Before booking a repeat of something, check the briefing's booked \
            list: an existing wake is moved with revise_wake, never re-booked.
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
                summonsGrant: try ToolArgExtractor.strictBool(argsJSON, key: "summons")
                    ?? false,
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

// MARK: - revise_wake

/// The move half of the palette (#354's triple-booking class): revision is
/// possible, so a follow-up is never re-booked beside its stale twin.
nonisolated func createReviseWakeTool(
    store: MemoryStore,
    recorder: CompanionFlightRecorder,
    context: CompanionTurnContext
) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "revise_wake",
        label: "revise wake",
        description: """
            Move or reword one of your booked wakes — "do it in an hour", a \
            better line for future-you — without losing it or minting a twin. \
            Pass its id (the briefing's booked list carries ids) and whatever \
            changes: a new time ('at' or 'in_minutes'), new 'content', or both.
            """,
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "id": PropertySchema(
                    type: "string", description: "The wake's id, from the briefing's list."),
                "content": PropertySchema(
                    type: "string", description: "New one-line content, if it changes."),
                "at": PropertySchema(
                    type: "string",
                    description: "New local time 'yyyy-MM-dd HH:mm'. Use this or in_minutes."
                ),
                "in_minutes": PropertySchema(
                    type: "integer", description: "New due time, minutes from now."),
            ],
            required: ["id"]
        ),
        execute: { _, argsJSON, _, _ in
            guard
                let id = ToolArgExtractor.string(argsJSON, key: "id")
                    .flatMap(UUID.init(uuidString:))
            else {
                throw CompanionToolError(message: "revise_wake requires the wake's 'id'.")
            }
            guard var wake = try await store.wake(id: id),
                wake.state == .booked || wake.state == .fired
            else {
                throw CompanionToolError(
                    message: "No open wake with that id — check the briefing's list.")
            }

            var newDue: Date?
            if let atRaw = ToolArgExtractor.string(argsJSON, key: "at") {
                guard let parsed = CompanionWakeTime.parse(atRaw) else {
                    throw CompanionToolError(
                        message: "Could not parse 'at' — use local 'yyyy-MM-dd HH:mm'.")
                }
                newDue = parsed
            } else if let minutes = ToolArgExtractor.int(argsJSON, key: "in_minutes") {
                newDue = Date().addingTimeInterval(TimeInterval(max(1, minutes)) * 60)
            }
            if let newDue, newDue <= Date() {
                throw CompanionToolError(message: "That time is in the past.")
            }
            let newContent = ToolArgExtractor.string(argsJSON, key: "content")
            guard newDue != nil || newContent != nil else {
                throw CompanionToolError(
                    message: "Nothing to revise — pass a new time, new content, or both.")
            }

            let oldDue = wake.due
            if let newDue { wake.due = newDue }
            if let newContent { wake.content = newContent }
            // A fired wake being revised is the turn moving its own occasion:
            // back to booked, so it fires at the new time instead of being
            // consumed by this turn's completion.
            wake.state = .booked
            try await store.upsertWake(wake)
            await recorder.record(
                "wake.revised",
                wakeID: wake.id,
                turnID: context.turnID,
                conversationID: context.conversationID,
                snapshot: [
                    "from": CompanionWakeTime.format(oldDue),
                    "to": CompanionWakeTime.format(wake.due),
                ],
                note: wake.content
            )
            return .text(
                "Revised [\(wake.wakeClass.rawValue)] → "
                    + "\(CompanionWakeTime.format(wake.due)): \(wake.content)")
        }
    )
}

// MARK: - cancel_wake

/// The deliberate exit: a cancelled wake is a recorded decision with a why —
/// never `dropped`, the silent-loss defect state the weekly review counts.
nonisolated func createCancelWakeTool(
    store: MemoryStore,
    recorder: CompanionFlightRecorder,
    context: CompanionTurnContext
) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "cancel_wake",
        label: "cancel wake",
        description: """
            Cancel a booked wake you no longer need — the moment passed, the \
            owner answered early, the plan changed. Pass its id and one line of \
            why: cancellation is a decision on your record, never a silent \
            drop. A promise you cancel frees that day's promise budget.
            """,
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "id": PropertySchema(
                    type: "string", description: "The wake's id, from the briefing's list."),
                "why": PropertySchema(
                    type: "string", description: "One line: why this wake is no longer needed."),
            ],
            required: ["id", "why"]
        ),
        execute: { _, argsJSON, _, _ in
            guard
                let id = ToolArgExtractor.string(argsJSON, key: "id")
                    .flatMap(UUID.init(uuidString:))
            else {
                throw CompanionToolError(message: "cancel_wake requires the wake's 'id'.")
            }
            guard let why = ToolArgExtractor.string(argsJSON, key: "why"), !why.isEmpty else {
                throw CompanionToolError(
                    message: "cancel_wake requires 'why' — the record must say.")
            }
            guard var wake = try await store.wake(id: id), wake.state == .booked else {
                throw CompanionToolError(
                    message: "No booked wake with that id — only a booked wake can be "
                        + "cancelled; check the briefing's list.")
            }
            wake.state = .cancelled
            try await store.upsertWake(wake)
            await recorder.record(
                "wake.cancelled",
                wakeID: wake.id,
                turnID: context.turnID,
                conversationID: context.conversationID,
                snapshot: ["class": wake.wakeClass.rawValue],
                note: why
            )
            return .text("Cancelled [\(wake.wakeClass.rawValue)]: \(wake.content)")
        }
    )
}

// MARK: - revise_instructions

/// The entity's pen on its own standing document (ADR-0040 §12, sectioned by
/// #370). Section-scoped replacement — a conversation that carries only the
/// IDENTITY section can revise it without fabricating the LOOP POLICY it
/// never saw — appended as a new full-document version, never an edit in
/// place, so the owner's history view always shows what conduct was in
/// force when.
nonisolated func createReviseInstructionsTool(
    store: MemoryStore,
    recorder: CompanionFlightRecorder,
    context: CompanionTurnContext
) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "revise_instructions",
        label: "revise instructions",
        description: """
            Rewrite one section of your standing instructions. IDENTITY is who \
            you are — it rides every conversation you have; LOOP POLICY is your \
            loop conduct — it rides only your Mission Control turns. Pass the \
            section, its COMPLETE new text (that section is replaced wholesale; \
            the other is untouched), and a one-line why. Use it when you learn \
            something durable: a rhythm that fits him, a register correction he \
            gave, a rule he set. Versioned and owner-visible; he can read and \
            edit every revision. Keep it short enough to live by — it rides in \
            your prompts.
            """,
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "section": PropertySchema(
                    type: "string",
                    description: "Which section the text replaces.",
                    enumValues: ["identity", "loop_policy"]
                ),
                "text": PropertySchema(
                    type: "string",
                    description: "The complete new text of that section."
                ),
                "why": PropertySchema(
                    type: "string",
                    description: "One line: what changed and what prompted it."
                ),
            ],
            required: ["section", "text", "why"]
        ),
        execute: { _, argsJSON, _, _ in
            guard let section = ToolArgExtractor.string(argsJSON, key: "section"),
                section == "identity" || section == "loop_policy"
            else {
                throw CompanionToolError(
                    message: "revise_instructions requires section: identity|loop_policy")
            }
            guard let text = ToolArgExtractor.string(argsJSON, key: "text"),
                !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                throw CompanionToolError(message: "revise_instructions requires 'text'")
            }
            guard let why = ToolArgExtractor.string(argsJSON, key: "why"), !why.isEmpty else {
                throw CompanionToolError(
                    message: "revise_instructions requires 'why' — the history must say.")
            }

            let sections = CompanionInstructions.split(
                (try await store.currentInstructions())?.text ?? "")
            let composed =
                section == "identity"
                ? CompanionInstructions.compose(
                    identity: text, loopPolicy: sections.loopPolicy ?? "")
                : CompanionInstructions.compose(
                    identity: sections.identity, loopPolicy: text)
            guard composed.count <= CompanionInstructions.maxLength else {
                return .text(
                    "Too long (\(composed.count) chars, cap \(CompanionInstructions.maxLength)). "
                        + "These ride in every prompt — cut before you grow.")
            }
            let version = try await store.appendInstructions(
                text: composed, author: "entity", note: why)
            await recorder.record(
                "instructions.revised",
                turnID: context.turnID,
                conversationID: context.conversationID,
                snapshot: [
                    "version": String(version), "section": section,
                    "chars": String(composed.count),
                ],
                note: why
            )
            return .text(
                "Instructions revised — now v\(version) (\(section) replaced). "
                    + "In force from your next turn.")
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
        audience: .companionOnly,
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
        audience: .companionOnly,
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
        audience: .companionOnly,
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
        audience: .companionOnly,
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
