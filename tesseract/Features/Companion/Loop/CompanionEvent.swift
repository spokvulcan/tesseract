//
//  CompanionEvent.swift
//  tesseract
//
//  One perception queued for the entity (ADR-0046, #368): every digital input
//  becomes exactly one Event, events queue in total order, and a granted turn
//  drains everything pending. This file is the vocabulary; the queue's math
//  lives in `MemoryStore+Events.swift`, the producers in
//  `CompanionPerception`.
//

import CryptoKit
import Foundation

/// The v1 Event kinds. `wakeDue` and `reportBack` are defined here but
/// produced by their own tickets (#371's clock, #372's deposit door); the
/// rest are the perception substrate's. Raw values are the persisted tag.
nonisolated enum CompanionEventKind: String, Codable, Sendable {
    /// A booked wake came due (#371 — the clock admits these).
    case wakeDue = "wake-due"
    /// A summoned dialogue's deposit landed (#372).
    case reportBack = "report-back"
    /// First presence of the calendar day.
    case dayStart = "day-start"
    /// The calendar day rolled over.
    case dayEnd = "day-end"
    /// The Mac woke from sleep.
    case macWake = "mac-wake"
    /// The app launched — the gap behind is unwatched time.
    case launchCatchUp = "launch-catch-up"
    /// External power appeared or vanished.
    case powerChange = "power-change"
    /// A sustained app switch (brief flips never become Events).
    case appSwitch = "app-switch"
    /// Another app banner-notified the owner (the Notification Hub, #378).
    case notificationArrived = "notification-arrived"
}

/// The queue's state machine — the wake table's proven shape (fired-but-
/// unconsumed re-presents): `pending` → `presented` (handed to a turn) →
/// `consumed` (that turn completed). A crash between the last two leaves the
/// recovery set.
nonisolated enum CompanionEventState: String, Codable, Sendable {
    case pending
    case presented
    case consumed
}

nonisolated struct CompanionEvent: Identifiable, Equatable, Sendable {
    let id: UUID
    let kind: CompanionEventKind
    /// One announceable line — what the turn's opening renders (#371).
    let content: String
    /// Kind-shaped JSON detail (an app-session span, a power verdict); nil
    /// when the content line is the whole fact.
    let payload: String?
    let occurredAt: Date
    var state: CompanionEventState
    /// Total order, assigned by the store at admission; nil before it.
    var seq: Int64?
    var admittedAt: Date?
    var presentedAt: Date?
    var consumedAt: Date?
    /// The turn that consumed it (#371 wires this).
    var turnID: UUID?

    /// A producer's perception: the five facts a producer owns. Everything
    /// else — state, seq, the admission stamp — is the store's to assign, so
    /// this init doesn't offer them.
    init(
        id: UUID = UUID(),
        kind: CompanionEventKind,
        content: String,
        payload: String? = nil,
        occurredAt: Date = Date()
    ) {
        self.id = id
        self.kind = kind
        self.content = content
        self.payload = payload
        self.occurredAt = occurredAt
        self.state = .pending
        self.seq = nil
        self.admittedAt = nil
        self.presentedAt = nil
        self.consumedAt = nil
        self.turnID = nil
    }

    /// The full row — only the store's decode constructs this shape.
    init(
        id: UUID, kind: CompanionEventKind, content: String, payload: String?,
        occurredAt: Date, state: CompanionEventState, seq: Int64?, admittedAt: Date?,
        presentedAt: Date?, consumedAt: Date?, turnID: UUID?
    ) {
        self.id = id
        self.kind = kind
        self.content = content
        self.payload = payload
        self.occurredAt = occurredAt
        self.state = state
        self.seq = seq
        self.admittedAt = admittedAt
        self.presentedAt = presentedAt
        self.consumedAt = consumedAt
        self.turnID = turnID
    }

    /// Exactly-once for once-per-occasion perceptions: the same occasion
    /// (e.g. `day-end:2026-07-18`) always mints the same id, so a producer
    /// firing twice — a repeated notification, a re-arm — collapses to one
    /// Event at admission instead of needing its own dedupe state.
    static func deterministicID(_ occasion: String) -> UUID {
        SHA256.hash(data: Data(occasion.utf8))
            .withUnsafeBytes { UUID(uuid: $0.loadUnaligned(as: uuid_t.self)) }
    }

    /// The id law for a wake's due-ness Event (ADR-0046): one wake, one
    /// `.wakeDue` Event, ever — and the one place a reader can recognize
    /// "this Event is that wake" (the render de-dup and the deferral
    /// telemetry, #404) without a payload parse.
    static func wakeDueID(_ wakeID: UUID) -> UUID {
        deterministicID("wake-due:\(wakeID.uuidString)")
    }

    /// Kind-shaped payload JSON — the one encoder every producer shares, so
    /// no door hand-rolls (and mis-escapes) its own literal.
    static func payloadJSON(_ value: some Encodable) -> String? {
        (try? JSONEncoder().encode(value)).flatMap { String(data: $0, encoding: .utf8) }
    }

    /// The correlation hints projected off the payload for the flight-recorder
    /// record — both decoded in one pass, because admission surfaces both and
    /// decoding each on its own would parse the same payload string twice:
    /// - `app`: the source app notification and app-switch events name.
    /// - `at`: for a span-shaped payload, the session's *start* — when the owner
    ///   switched *to* the app, not when he later left it. The app-switch admits
    ///   at the session's close, so surfacing `start` lets the inferred-miss
    ///   tally (#380) correlate against a real switch-to rather than the app he
    ///   was already in. Nil for a notification payload (its epoch field is
    ///   `occurredAt`, not `start`).
    /// Either is nil when the payload lacks that field (or has no payload).
    var recordHints: (app: String?, at: Int?) {
        guard let payload, let data = payload.data(using: .utf8),
            let hints = try? JSONDecoder().decode(PayloadHints.self, from: data)
        else { return (nil, nil) }
        let trimmed = hints.app?.trimmingCharacters(in: .whitespacesAndNewlines)
        return ((trimmed?.isEmpty ?? true) ? nil : trimmed, hints.start)
    }

    /// The minimal projection `recordHints` decodes — only the two correlation
    /// keys, never the full body the payload also carries.
    private struct PayloadHints: Decodable {
        let app: String?
        let start: Int?
    }
}

// MARK: - Notification Hub (#378)

/// One banner the Notification Hub watched, as the AX layer read it — the
/// producer's raw perception before it becomes an Event. Fields are best-
/// effort: the AX tree exposes a display name (never a bundle ID) and,
/// defensively, may hand back only the flattened `AXDescription`, so any
/// field may be empty.
nonisolated struct CapturedNotification: Sendable, Equatable {
    /// Source-app display name — the only identity the tree exposes.
    let app: String
    let title: String
    let subtitle: String
    let body: String
    /// The banner's `AXIdentifier` — a stable per-notification UUID when the
    /// tree carried one; nil falls back to a content-derived id.
    let uuid: String?
    let occurredAt: Date

    init(
        app: String, title: String, subtitle: String = "", body: String = "",
        uuid: String? = nil, occurredAt: Date = Date()
    ) {
        self.app = app
        self.title = title
        self.subtitle = subtitle
        self.body = body
        self.uuid = uuid
        self.occurredAt = occurredAt
    }
}

extension CompanionEvent {

    /// The announceable content line caps the body at a few hundred chars; the
    /// full body rides uncapped in the payload (#378).
    nonisolated static let notificationBodyCap = 500

    /// The payload shape a notification Event persists — the full fields the
    /// content line trims; `recordHints` decodes `app` back off it at admission.
    nonisolated struct NotificationPayload: Codable {
        let app: String
        let title: String
        let subtitle: String?
        let body: String
        let uuid: String?
        /// Unix seconds — the same epoch convention the span payload keeps.
        let occurredAt: Int
    }

    /// Turn a watched banner into exactly one Event, or nil to drop it. The
    /// self-exclusion invariant lives here (#378): Tesseract's own banners
    /// never become Events, matched on display name because the tree carries no
    /// bundle IDs. The id is deterministic from the banner UUID, so the same
    /// banner re-observed across a watcher re-attach collapses at admission;
    /// without a UUID it derives from the content, still collapsing exact
    /// repeats.
    ///
    /// `nonisolated`: extension members default to `@MainActor` under the
    /// project's default isolation, but the producer path reads banners off the
    /// main actor (`NotificationCenterWatcher.handleCreated`), so this must run
    /// anywhere — the base type is already a `nonisolated struct`.
    nonisolated static func notification(
        from captured: CapturedNotification, selfDisplayNames: Set<String>
    ) -> CompanionEvent? {
        let app = captured.app.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !app.isEmpty else { return nil }
        let excluded = selfDisplayNames.map { $0.lowercased() }
        guard !excluded.contains(app.lowercased()) else { return nil }

        let title = captured.title.trimmingCharacters(in: .whitespacesAndNewlines)

        // The invariant's stacked form (#378, field-found): a collapsed
        // notification stack reads as app "Stacked summary" with the real
        // app leading the title ("Tesseract Agent: Jarvis — 32m ago"), so
        // the display-name match above never fires. Match the title's
        // leading segment against the self set too.
        if app.lowercased() == "stacked summary",
            let leading = title.split(separator: ":").first.map({
                $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            }),
            excluded.contains(leading)
        {
            return nil
        }
        let subtitle = captured.subtitle.trimmingCharacters(in: .whitespacesAndNewlines)
        let body = captured.body.trimmingCharacters(in: .whitespacesAndNewlines)

        let occasion =
            captured.uuid.map { "notification:\($0)" }
            ?? "notification:\(app)|\(title)|\(subtitle)|\(body)"

        let cappedBody =
            body.count > notificationBodyCap
            ? String(body.prefix(notificationBodyCap)) + "…" : body
        let tail = [title, cappedBody].filter { !$0.isEmpty }.joined(separator: " — ")
        let content = tail.isEmpty ? app : "\(app): \(tail)"

        let payload = NotificationPayload(
            app: app, title: title, subtitle: subtitle.isEmpty ? nil : subtitle,
            body: body, uuid: captured.uuid,
            occurredAt: Int(captured.occurredAt.timeIntervalSince1970))

        return CompanionEvent(
            id: deterministicID(occasion),
            kind: .notificationArrived,
            content: content,
            payload: payloadJSON(payload),
            occurredAt: captured.occurredAt)
    }
}

/// The drained batch as the turn's opening sees it (#371): everything that
/// reached the entity since its last turn, in total order — the fold's
/// `events` argument, rendered.
nonisolated enum CompanionEventBatch {

    /// `dueWakes` are the wakes the situation block already renders under
    /// DUE NOW: their `.wakeDue` Events are suppressed here (#404), so one
    /// due wake reaches the entity once, not in two formats. Presentation
    /// only — the fold still consumes the suppressed Event exactly as
    /// before. A `.wakeDue` Event whose wake is *not* in the list (the wake
    /// was cancelled or completed between admission and this turn) still
    /// renders: nothing else tells the entity it happened.
    static func render(
        _ events: [CompanionEvent], dueWakes: [CompanionWake] = [], now: Date = Date()
    ) -> String {
        let dueEventIDs = Set(dueWakes.map { CompanionEvent.wakeDueID($0.id) })
        let rendered = events.filter { $0.kind != .wakeDue || !dueEventIDs.contains($0.id) }
        guard !rendered.isEmpty else { return "" }
        let lines = rendered.enumerated().map { index, event -> String in
            let when = event.occurredAt.formatted(date: .omitted, time: .shortened)
            return "\(index + 1). [\(event.kind.rawValue)] \(when) — \(event.content)"
        }
        return """
            <events>
            Everything that reached you since your last turn, in order:
            \(lines.joined(separator: "\n"))
            </events>
            """
    }
}
