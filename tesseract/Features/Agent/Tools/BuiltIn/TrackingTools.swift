//
//  TrackingTools.swift
//  tesseract
//
//  The five typed talk-time tools of the tracking model (#308): the day
//  contract, the step log, the samples, the backlog, and the close-out.
//  Registered in every conversation (the `remember`/`recall`/`contest`
//  precedent) — the check-in IS the measuring instrument, whichever
//  conversation it happens in.
//
//  One door per fact (#333's rule): mood is `log_sample`'s, a step event is
//  `log_step`'s — never `remember`'s. Memory may still *conclude* from what
//  was said; the structured fact has a single origin. Args are schema-checked
//  here so a garbled call can't write a garbage row.
//

import Foundation

nonisolated struct TrackingToolError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

// MARK: - plan_day

nonisolated func createPlanDayTool(store: MemoryStore) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "plan_day",
        label: "plan day",
        description: """
            Record today's day contract after he has confirmed it: one keystone step \
            (the day's win condition, ~20 minutes of hard focus), optionally a short \
            chain of follow-on steps ("and if you finish, what's next?"), and at most \
            two support items. One step is active at a time — finishing a step arms \
            the next the same day. Steps past the keystone are ambition, never \
            obligation: only the keystone decides whether the day is kept.

            Call this once the contract is agreed in conversation, not to propose it. \
            If a contract already exists for today, pass replace=true to overwrite it \
            (say so to him first).
            """,
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "keystone": PropertySchema(
                    type: "string",
                    description: "The ONE hard step that makes today a win."
                ),
                "then": PropertySchema(
                    type: "array",
                    description:
                        "Follow-on hard steps, in order, armed one at a time after the "
                        + "keystone. Usually 0-2.",
                    items: PropertySchema(type: "string", description: "A follow-on step.")
                ),
                "support": PropertySchema(
                    type: "array",
                    description: "Light support items (max 2). Never pushed.",
                    items: PropertySchema(type: "string", description: "A support item.")
                ),
                "replace": PropertySchema(
                    type: "boolean",
                    description: "Overwrite an existing contract for today."
                ),
            ],
            required: ["keystone"]
        ),
        execute: { _, argsJSON, _, _ in
            guard let keystone = ToolArgExtractor.string(argsJSON, key: "keystone"),
                !keystone.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                throw TrackingToolError(message: "plan_day requires a non-empty 'keystone'")
            }
            let then = ToolArgExtractor.stringArray(argsJSON, key: "then") ?? []
            let support = Array(
                (ToolArgExtractor.stringArray(argsJSON, key: "support") ?? []).prefix(2))
            let replace = ToolArgExtractor.bool(argsJSON, key: "replace") ?? false

            let today = TrackingDay.key()
            let existing = try await store.day(today)
            if let existing, !existing.chain.isEmpty, !replace {
                return .text(
                    "A contract for today already exists:\n\(existing.chainSummary)\n"
                        + "Pass replace=true to overwrite it.")
            }

            var chain = [ContractStep(title: keystone, status: .active)]
            chain.append(contentsOf: then.map { ContractStep(title: $0) })
            var day = existing ?? DayRecord(date: today)
            day.chain = chain
            day.support = support
            try await store.upsertDay(day)
            return .text(day.chainSummary)
        }
    )
}

// MARK: - log_step

nonisolated func createLogStepTool(store: MemoryStore) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "log_step",
        label: "log step",
        description: """
            Record what happened to the active step of today's contract chain. \
            Actions: 'started' (he began the step; also re-arms a blocked step), \
            'done' (step finished — the next step in the chain arms immediately), \
            'blocked' (stuck; say why in note), 'switched' (he consciously moved to \
            something else — the drift is named once, recorded, and momentum wins; \
            optionally pass reseed to push the step to tonight/tomorrow's seed), \
            'dropped' (abandoned; the chain moves on). Always tell him what you \
            recorded.
            """,
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "action": PropertySchema(
                    type: "string",
                    description: "What happened.",
                    enumValues: ["started", "done", "blocked", "switched", "dropped"]
                ),
                "note": PropertySchema(
                    type: "string",
                    description: "Context worth keeping — why blocked, what he switched to."
                ),
                "reseed": PropertySchema(
                    type: "string",
                    description:
                        "For 'switched': where the step goes — appended to the day's seed."
                ),
            ],
            required: ["action"]
        ),
        execute: { _, argsJSON, _, _ in
            guard let actionRaw = ToolArgExtractor.string(argsJSON, key: "action") else {
                throw TrackingToolError(message: "log_step requires an 'action'")
            }
            let note = ToolArgExtractor.string(argsJSON, key: "note")
            let today = TrackingDay.key()
            guard var day = try await store.day(today), !day.chain.isEmpty else {
                throw TrackingToolError(
                    message: "No contract exists for today — plan_day first.")
            }

            let now = Date()
            var observationKind: String?

            switch actionRaw {
            case "started":
                guard
                    let index = day.chain.firstIndex(where: {
                        $0.status == .active || $0.status == .blocked
                    }) ?? day.chain.firstIndex(where: { $0.status == .pending })
                else {
                    throw TrackingToolError(message: "No step left to start.")
                }
                day.chain[index].status = .active
                day.chain[index].startedAt = day.chain[index].startedAt ?? now
                if let note { day.chain[index].note = note }
                observationKind = "step-started"
            case "done":
                guard let index = day.chain.firstIndex(where: { $0.status == .active }) else {
                    throw TrackingToolError(message: "No active step to finish.")
                }
                day.chain[index].status = .done
                day.chain[index].closedAt = now
                if let next = day.chain.firstIndex(where: { $0.status == .pending }) {
                    day.chain[next].status = .active
                }
                observationKind = "step-done"
            case "blocked":
                guard let index = day.chain.firstIndex(where: { $0.status == .active }) else {
                    throw TrackingToolError(message: "No active step to mark blocked.")
                }
                day.chain[index].status = .blocked
                if let note { day.chain[index].note = note }
                observationKind = "step-blocked"
            case "switched":
                guard
                    let index = day.chain.firstIndex(where: {
                        $0.status == .active || $0.status == .blocked
                    })
                else {
                    throw TrackingToolError(message: "No active step to switch away from.")
                }
                day.chain[index].status = .switched
                if let note { day.chain[index].note = note }
                if let reseed = ToolArgExtractor.string(argsJSON, key: "reseed") {
                    day.seed = [day.seed, reseed].compactMap { $0 }.joined(separator: " · ")
                }
                observationKind = "conscious-switch"
            case "dropped":
                guard
                    let index = day.chain.firstIndex(where: {
                        $0.status == .active || $0.status == .blocked
                    })
                else {
                    throw TrackingToolError(message: "No active step to drop.")
                }
                day.chain[index].status = .dropped
                if let note { day.chain[index].note = note }
                if let next = day.chain.firstIndex(where: { $0.status == .pending }) {
                    day.chain[next].status = .active
                }
                observationKind = "step-dropped"
            default:
                throw TrackingToolError(
                    message:
                        "Unknown action '\(actionRaw)' — one of started/done/blocked/switched/dropped."
                )
            }

            try await store.upsertDay(day)
            if let observationKind {
                try await store.appendObservation(
                    TrackingObservation(
                        domain: .work, kind: observationKind,
                        value: note ?? day.activeStep?.title ?? day.chain[0].title,
                        source: .elicited))
            }
            return .text(day.chainSummary)
        }
    )
}

// MARK: - log_sample

nonisolated func createLogSampleTool(store: MemoryStore) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "log_sample",
        label: "log sample",
        description: """
            Record one elicited body/mind sample from the conversation: 'sleep' (the \
            morning question), 'mood' or 'energy' (bookends), 'movement' (evening). \
            Value is a short plain answer in his terms — "poor, ~5h", "good", "walk, \
            30 min". The conversation is the measuring instrument: log what he said, \
            in essence, not a number he didn't give.
            """,
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "kind": PropertySchema(
                    type: "string",
                    description: "Which sample.",
                    enumValues: ["sleep", "mood", "energy", "movement"]
                ),
                "value": PropertySchema(
                    type: "string",
                    description: "The sample, short, in his terms."
                ),
            ],
            required: ["kind", "value"]
        ),
        execute: { _, argsJSON, _, _ in
            guard let kind = ToolArgExtractor.string(argsJSON, key: "kind"),
                ["sleep", "mood", "energy", "movement"].contains(kind)
            else {
                throw TrackingToolError(
                    message: "log_sample requires kind: sleep|mood|energy|movement")
            }
            guard let value = ToolArgExtractor.string(argsJSON, key: "value"),
                !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                throw TrackingToolError(message: "log_sample requires a non-empty 'value'")
            }
            let domain: TrackingDomain = (kind == "sleep" || kind == "movement") ? .body : .mind
            try await store.appendObservation(
                TrackingObservation(domain: domain, kind: kind, value: value, source: .elicited))
            return .text("Logged \(kind): \(value)")
        }
    )
}

// MARK: - log_task

nonisolated func createLogTaskTool(store: MemoryStore) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "log_task",
        label: "log task",
        description: """
            The backlog — items a day contract can draw steps from. Actions: 'add' \
            (title required; stream = the life area it belongs to, e.g. "tesseract", \
            "health"; cadence 'daily' makes it a recurring habit that re-arms each \
            day; domain work|body|mind, default work), 'done' (a one-shot closes; a \
            daily habit records today's check-off and stays), 'drop', 'list' (the \
            open backlog). Refer to items by title fragment or id.
            """,
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "action": PropertySchema(
                    type: "string",
                    description: "What to do.",
                    enumValues: ["add", "done", "drop", "list"]
                ),
                "title": PropertySchema(
                    type: "string",
                    description: "For 'add': the item. For 'done'/'drop': title fragment or id."
                ),
                "stream": PropertySchema(
                    type: "string",
                    description: "Life area the item flows in (conversational name)."
                ),
                "cadence": PropertySchema(
                    type: "string",
                    description: "once (default) or daily (a recurring habit).",
                    enumValues: ["once", "daily"]
                ),
                "domain": PropertySchema(
                    type: "string",
                    description: "work (default), body, or mind — habit check-offs land here.",
                    enumValues: ["work", "body", "mind"]
                ),
            ],
            required: ["action"]
        ),
        execute: { _, argsJSON, _, _ in
            guard let action = ToolArgExtractor.string(argsJSON, key: "action") else {
                throw TrackingToolError(message: "log_task requires an 'action'")
            }
            switch action {
            case "add":
                guard let title = ToolArgExtractor.string(argsJSON, key: "title"),
                    !title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                else {
                    throw TrackingToolError(message: "log_task add requires a 'title'")
                }
                let item = WorkItemRecord(
                    title: title,
                    stream: ToolArgExtractor.string(argsJSON, key: "stream"),
                    domain: TrackingDomain(
                        rawValue: ToolArgExtractor.string(argsJSON, key: "domain") ?? "work")
                        ?? .work,
                    cadence: WorkItemCadence(
                        rawValue: ToolArgExtractor.string(argsJSON, key: "cadence") ?? "once")
                        ?? .once
                )
                try await store.upsertWorkItem(item)
                let habit = item.cadence == .daily ? " (daily habit)" : ""
                return .text("Added: \(item.title)\(habit)")
            case "done", "drop":
                guard let ref = ToolArgExtractor.string(argsJSON, key: "title") else {
                    throw TrackingToolError(
                        message: "log_task \(action) requires 'title' (fragment or id)")
                }
                guard var item = try await store.findWorkItem(idOrTitle: ref) else {
                    throw TrackingToolError(
                        message:
                            "No single open item matches '\(ref)' — use log_task list and be more specific."
                    )
                }
                if action == "drop" {
                    item.status = .dropped
                    try await store.upsertWorkItem(item)
                    return .text("Dropped: \(item.title)")
                }
                if item.cadence == .daily {
                    try await store.appendObservation(
                        TrackingObservation(
                            domain: item.domain, kind: "habit-checkoff", value: item.title,
                            source: .elicited, stream: item.stream))
                    return .text("Checked off today: \(item.title) (habit stays on the list)")
                }
                item.status = .done
                try await store.upsertWorkItem(item)
                return .text("Done: \(item.title)")
            case "list":
                let items = try await store.workItems(status: .open)
                guard !items.isEmpty else { return .text("The backlog is empty.") }
                let lines = items.map { item in
                    var tags: [String] = []
                    if let stream = item.stream { tags.append(stream) }
                    if item.cadence == .daily { tags.append("daily") }
                    let suffix = tags.isEmpty ? "" : " [\(tags.joined(separator: ", "))]"
                    return "- \(item.title)\(suffix)"
                }
                return .text(lines.joined(separator: "\n"))
            default:
                throw TrackingToolError(
                    message: "Unknown action '\(action)' — one of add/done/drop/list.")
            }
        }
    )
}

// MARK: - close_day

nonisolated func createCloseDayTool(store: MemoryStore) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "close_day",
        label: "close day",
        description: """
            Close today at the end of the evening journal: stamps the close-out (an \
            unclosed day is what tomorrow's morning opens with a catch-up about) and \
            records tomorrow's seed — what tomorrow's planning should open with. \
            Call it after the journal talk, once the contract is accounted for.
            """,
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "seed": PropertySchema(
                    type: "string",
                    description: "Tomorrow's seed — the thread to pick up at morning planning."
                )
            ],
            required: []
        ),
        execute: { _, argsJSON, _, _ in
            let today = TrackingDay.key()
            var day = try await store.day(today) ?? DayRecord(date: today)
            day.closedAt = Date()
            if let seed = ToolArgExtractor.string(argsJSON, key: "seed"), !seed.isEmpty {
                day.seed = seed
            }
            try await store.upsertDay(day)
            let depth = day.chainDepth
            let kept = (day.keystone?.status == .done)
            let verdict =
                day.chain.isEmpty
                ? "No contract today."
                : (kept ? "Keystone kept, depth \(depth)/\(day.chain.count)." : "Keystone open.")
            return .text("Day closed. \(verdict)\(day.seed.map { " Seed: \($0)" } ?? "")")
        }
    )
}
