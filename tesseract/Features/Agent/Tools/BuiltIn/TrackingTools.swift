//
//  TrackingTools.swift
//  tesseract
//
//  `track(kind, payload)` — the one generic tracking door (ADR-0046, #369).
//  Replaces the five ceremony tools of #308 (plan_day, log_step, log_sample,
//  log_task, close_day): the Observation / Contract Chain / Work Item schema
//  survives as data behind one typed door, and the workflow — when to plan,
//  what to log, how a day closes — is the entity's practice, never the
//  tool's. Registered in every conversation (the `remember`/`recall` \
//  precedent): the check-in IS the measuring instrument, whichever
//  conversation it happens in.
//
//  One door per fact (#333's rule) still holds: a structured tracking fact is
//  written here, never through `remember`. Args are checked loudly — a
//  non-conforming payload fails with the expected shape named, never a
//  silently tolerated coercion (#354's summons-as-string class).
//

import Foundation
import MLXLMCommon

nonisolated struct TrackingToolError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

/// The four sample kinds whose domain is fixed by the tracking model (#308);
/// any other observation kind must name its domain explicitly.
private nonisolated let sampleDomains: [String: TrackingDomain] = [
    "sleep": .body, "movement": .body, "mood": .mind, "energy": .mind,
]

// MARK: - track

nonisolated func createTrackTool(store: MemoryStore) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "track",
        label: "track",
        description: """
            Your one door to the tracking record: dated observations, the day's \
            contract chain, and the backlog. How you track — when to plan, what \
            to elicit, when a day closes — is your practice; this tool only \
            writes the data shapes.

            kind 'observation' — one dated, typed fact from the conversation. \
            payload: {kind, value, domain?, stream?}. The four samples map their \
            own domain (sleep/movement → body, mood/energy → mind); any other \
            kind (step events, habit check-offs, whatever your practice names) \
            needs an explicit domain: work|body|mind. Log what he said, in his \
            terms — never a number he didn't give.

            kind 'day' — the day contract as data. payload: {date?, chain?, \
            support?, seed?, closed?}; only the fields you pass change. 'chain' \
            replaces the whole step list: [{title, status?, note?}], statuses \
            pending|active|done|blocked|switched|dropped, at most ONE active. \
            Steps keep their timestamps across rewrites (matched by title). \
            'support' is at most two light items. 'seed' is the thread the next \
            morning picks up (empty string clears it). 'closed' true/false \
            stamps or reopens the day. 'date' (yyyy-MM-dd) defaults to today — \
            pass yesterday's to settle a day after midnight.

            kind 'item' — the backlog. payload: {action: add|done|drop|list, \
            title?, stream?, cadence?, domain?}. 'add' takes the title with an \
            optional stream (the life area), cadence once|daily (daily = a \
            recurring habit), and domain. 'done' on a daily habit records \
            today's check-off and keeps the item; on a one-shot it closes it. \
            Refer to items by title fragment or id.
            """,
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "kind": PropertySchema(
                    type: "string",
                    description: "Which record family the payload writes.",
                    enumValues: ["observation", "day", "item"]
                ),
                "payload": PropertySchema(
                    type: "object",
                    description: "The kind-shaped data — see the tool description."
                ),
            ],
            required: ["kind", "payload"]
        ),
        execute: { _, argsJSON, _, _ in
            guard let kind = ToolArgExtractor.string(argsJSON, key: "kind") else {
                throw TrackingToolError(
                    message: "track requires 'kind': observation|day|item")
            }
            let payload = try ToolArgExtractor.object(argsJSON, key: "payload")
            switch kind {
            case "observation":
                return .text(try await trackObservation(payload, store: store))
            case "day":
                return .text(try await trackDay(payload, store: store))
            case "item":
                return .text(try await trackItem(payload, store: store))
            default:
                throw TrackingToolError(
                    message: "Unknown kind '\(kind)' — one of observation|day|item.")
            }
        }
    )
}

// MARK: - observation

private nonisolated func trackObservation(
    _ payload: [String: JSONValue], store: MemoryStore
) async throws -> String {
    guard let kind = ToolArgExtractor.string(payload, key: "kind"),
        !kind.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    else {
        throw TrackingToolError(message: "observation payload requires a 'kind'")
    }
    guard let value = ToolArgExtractor.string(payload, key: "value"),
        !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    else {
        throw TrackingToolError(message: "observation payload requires a non-empty 'value'")
    }
    let domain: TrackingDomain
    if let explicit = ToolArgExtractor.string(payload, key: "domain") {
        guard let parsed = TrackingDomain(rawValue: explicit) else {
            throw TrackingToolError(
                message: "Unknown domain '\(explicit)' — one of work|body|mind.")
        }
        domain = parsed
    } else if let mapped = sampleDomains[kind] {
        domain = mapped
    } else {
        throw TrackingToolError(
            message: "Observation kind '\(kind)' needs an explicit 'domain' (work|body|mind).")
    }
    try await store.appendObservation(
        TrackingObservation(
            domain: domain, kind: kind, value: value, source: .elicited,
            stream: ToolArgExtractor.string(payload, key: "stream")))
    return "Recorded \(kind): \(value)"
}

// MARK: - day

private nonisolated func trackDay(
    _ payload: [String: JSONValue], store: MemoryStore
) async throws -> String {
    let dateKey = ToolArgExtractor.string(payload, key: "date") ?? TrackingDay.key()
    guard TrackingDay.startOfDay(forKey: dateKey) != nil else {
        throw TrackingToolError(message: "Bad 'date' — use yyyy-MM-dd.")
    }

    var day = try await store.day(dateKey) ?? DayRecord(date: dateKey)
    var touched = false
    let now = Date()
    var stepObservations: [TrackingObservation] = []

    if payload["chain"] != nil {
        let previous = day.chain
        day.chain = try parseChain(payload, previous: previous, now: now)
        stepObservations = stepTransitionObservations(from: previous, to: day.chain)
        touched = true
    }
    if payload["support"] != nil {
        guard let support = ToolArgExtractor.stringArray(payload, key: "support") else {
            throw TrackingToolError(message: "'support' must be an array of strings.")
        }
        guard support.count <= 2 else {
            throw TrackingToolError(
                message: "'support' carries at most two items — the contract stays light.")
        }
        day.support = support
        touched = true
    }
    if payload["seed"] != nil {
        guard let seed = ToolArgExtractor.string(payload, key: "seed") else {
            throw TrackingToolError(message: "'seed' must be a string.")
        }
        day.seed = seed.isEmpty ? nil : seed
        touched = true
    }
    if let closed = try ToolArgExtractor.strictBool(payload, key: "closed") {
        day.closedAt = closed ? (day.closedAt ?? now) : nil
        touched = true
    }

    guard touched else {
        throw TrackingToolError(
            message: "day payload needs at least one of chain/support/seed/closed.")
    }
    try await store.upsertDay(day)
    for observation in stepObservations {
        try await store.appendObservation(observation)
    }
    return renderDay(day)
}

/// The work-domain Observation stream survives the ceremony's death (#369's
/// data-shapes rule): every step status transition still lands as the same
/// typed row `log_step` used to emit — mechanically, from the chain diff, so
/// the weekly walk keeps its rows without the entity spending a second call.
private nonisolated func stepTransitionObservations(
    from previous: [ContractStep], to chain: [ContractStep]
) -> [TrackingObservation] {
    let previousByTitle = Dictionary(
        previous.map { ($0.title.lowercased(), $0.status) },
        uniquingKeysWith: { first, _ in first })
    return chain.compactMap { step in
        let before = previousByTitle[step.title.lowercased()] ?? .pending
        guard step.status != before else { return nil }
        let kind: String
        switch step.status {
        case .pending: return nil
        case .active: kind = "step-started"
        case .done: kind = "step-done"
        case .blocked: kind = "step-blocked"
        case .switched: kind = "conscious-switch"
        case .dropped: kind = "step-dropped"
        }
        return TrackingObservation(
            domain: .work, kind: kind, value: step.note ?? step.title, source: .elicited)
    }
}

/// The chain arrives as data and leaves as data — the one shape rule enforced
/// here is the Contract Chain's own: at most one active step. Timestamps are
/// measurement, so a rewrite carries them forward (matched by title) and a
/// step newly active/done gains its stamp now.
private nonisolated func parseChain(
    _ payload: [String: JSONValue], previous: [ContractStep], now: Date
) throws -> [ContractStep] {
    let items = try ToolArgExtractor.objectArray(payload, key: "chain")
    let previousByTitle = Dictionary(
        previous.map { ($0.title.lowercased(), $0) },
        uniquingKeysWith: { first, _ in first })

    var chain: [ContractStep] = []
    for item in items {
        guard let title = ToolArgExtractor.string(item, key: "title"),
            !title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw TrackingToolError(message: "Every chain step requires a 'title'.")
        }
        let status: ContractStepStatus
        if let raw = ToolArgExtractor.string(item, key: "status") {
            guard let parsed = ContractStepStatus(rawValue: raw) else {
                throw TrackingToolError(
                    message: "Unknown step status '\(raw)' — one of "
                        + "pending|active|done|blocked|switched|dropped.")
            }
            status = parsed
        } else {
            status = .pending
        }
        let old = previousByTitle[title.lowercased()]
        var step = ContractStep(
            title: title,
            status: status,
            startedAt: old?.startedAt,
            closedAt: old?.closedAt,
            note: ToolArgExtractor.string(item, key: "note") ?? old?.note)
        if step.status == .active, step.startedAt == nil { step.startedAt = now }
        if step.status == .done, step.closedAt == nil { step.closedAt = now }
        chain.append(step)
    }

    guard chain.filter({ $0.status == .active }).count <= 1 else {
        throw TrackingToolError(
            message: "The chain carries at most ONE active step at a time.")
    }
    return chain
}

/// The day as data — no verdicts, no ceremony (#354's "Day closed" class):
/// the close-out stamp renders only when it exists, as a fact with a time.
private nonisolated func renderDay(_ day: DayRecord) -> String {
    var lines = ["Contract for \(day.date):"]
    if day.chain.isEmpty {
        lines.append("(no steps)")
    } else {
        for (index, step) in day.chain.enumerated() {
            let marker = index == 0 ? "keystone" : "step \(index + 1)"
            lines.append("\(index + 1). [\(step.status.rawValue)] \(step.title) (\(marker))")
        }
    }
    if !day.support.isEmpty {
        lines.append("Support: \(day.support.joined(separator: "; "))")
    }
    if let seed = day.seed { lines.append("Seed: \(seed)") }
    if let closedAt = day.closedAt {
        lines.append(
            "Close-out stamped \(closedAt.formatted(date: .omitted, time: .shortened)).")
    }
    return lines.joined(separator: "\n")
}

// MARK: - item

private nonisolated func trackItem(
    _ payload: [String: JSONValue], store: MemoryStore
) async throws -> String {
    guard let action = ToolArgExtractor.string(payload, key: "action") else {
        throw TrackingToolError(message: "item payload requires 'action': add|done|drop|list")
    }
    switch action {
    case "add":
        guard let title = ToolArgExtractor.string(payload, key: "title"),
            !title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw TrackingToolError(message: "item add requires a 'title'")
        }
        let domain: TrackingDomain
        if let raw = ToolArgExtractor.string(payload, key: "domain") {
            guard let parsed = TrackingDomain(rawValue: raw) else {
                throw TrackingToolError(
                    message: "Unknown domain '\(raw)' — one of work|body|mind.")
            }
            domain = parsed
        } else {
            domain = .work
        }
        let cadence: WorkItemCadence
        if let raw = ToolArgExtractor.string(payload, key: "cadence") {
            guard let parsed = WorkItemCadence(rawValue: raw) else {
                throw TrackingToolError(
                    message: "Unknown cadence '\(raw)' — one of once|daily.")
            }
            cadence = parsed
        } else {
            cadence = .once
        }
        let item = WorkItemRecord(
            title: title,
            stream: ToolArgExtractor.string(payload, key: "stream"),
            domain: domain,
            cadence: cadence)
        try await store.upsertWorkItem(item)
        return "Added: \(item.title)\(item.cadence == .daily ? " (daily habit)" : "")"

    case "done", "drop":
        guard let ref = ToolArgExtractor.string(payload, key: "title") else {
            throw TrackingToolError(
                message: "item \(action) requires 'title' (fragment or id)")
        }
        guard var item = try await store.findWorkItem(idOrTitle: ref) else {
            throw TrackingToolError(
                message: "No single open item matches '\(ref)' — list the backlog and be "
                    + "more specific.")
        }
        if action == "drop" {
            item.status = .dropped
            try await store.upsertWorkItem(item)
            return "Dropped: \(item.title)"
        }
        if item.cadence == .daily {
            try await store.appendObservation(
                TrackingObservation(
                    domain: item.domain, kind: "habit-checkoff", value: item.title,
                    source: .elicited, stream: item.stream))
            return "Checked off today: \(item.title) (the habit stays on the list)"
        }
        item.status = .done
        try await store.upsertWorkItem(item)
        return "Done: \(item.title)"

    case "list":
        let items = try await store.workItems(status: .open)
        guard !items.isEmpty else { return "The backlog is empty." }
        return items.map { item in
            var tags: [String] = []
            if let stream = item.stream { tags.append(stream) }
            if item.cadence == .daily { tags.append("daily") }
            let suffix = tags.isEmpty ? "" : " [\(tags.joined(separator: ", "))]"
            return "- \(item.title)\(suffix)"
        }.joined(separator: "\n")

    default:
        throw TrackingToolError(
            message: "Unknown action '\(action)' — one of add/done/drop/list.")
    }
}
