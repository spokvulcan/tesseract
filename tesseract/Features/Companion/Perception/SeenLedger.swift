//
//  SeenLedger.swift
//  tesseract
//
//  Which of other apps' notifications the owner has already seen. A banner
//  counts as seen when its app comes to the front after it arrived (within
//  15 minutes while the owner was present, or on their return if it arrived
//  while they were away). Only unseen, unresolved banners ever reach Jarvis,
//  in batches, at Breakpoints and during Triage — never one by one, and
//  never twice. The owner's rules apply first. Unresolved items expire after
//  24 hours. Pure: a value the Day Engine keeps in its state.
//

import Foundation

nonisolated struct SeenLedger: Sendable, Equatable, Codable {

    struct Entry: Sendable, Equatable, Codable, Identifiable {
        var notification: ObservedNotification
        /// The owner was at the Mac when it arrived.
        var arrivedPresent: Bool
        /// The owner's rule for it, if one matched.
        var rule: TriageRule.Action?
        var seenAt: Date?
        /// Judged by a Triage and left to wait.
        var triagedAt: Date?
        /// Put in front of the owner (raised, or on a Breakpoint card).
        var presentedAt: Date?
        var id: String { notification.id }
    }

    /// A present owner who opens the app within this window has seen it.
    static let seenWindow: TimeInterval = 15 * 60
    /// Unresolved items stop mattering after a day.
    static let expiry: TimeInterval = 24 * 3600
    /// Keep the ledger bounded.
    static let capacity = 300

    private(set) var entries: [Entry] = []

    /// Admit a banner. Returns false for a banner already in the ledger. The
    /// owner's rules come first; without one, its source decides (noise is
    /// ignored, an app's news is held for the next Breakpoint).
    @discardableResult
    mutating func arrived(_ notification: ObservedNotification, present: Bool, rules: [TriageRule])
        -> Bool
    {
        guard !entries.contains(where: { $0.id == notification.id }) else { return false }
        let rule =
            TriageRules.verdict(for: notification, rules: rules)
            ?? NotificationSources.defaultAction(for: notification.source)
        entries.append(Entry(notification: notification, arrivedPresent: present, rule: rule))
        if entries.count > Self.capacity { entries.removeFirst(entries.count - Self.capacity) }
        return true
    }

    /// An app came to the front: what it notified about is seen. Returns the
    /// ids newly marked.
    @discardableResult
    mutating func appActivated(_ appName: String, at time: Date) -> [String] {
        let app = appName.lowercased()
        var marked: [String] = []
        for index in entries.indices {
            let entry = entries[index]
            guard entry.seenAt == nil, entry.notification.app.lowercased() == app,
                entry.notification.arrivedAt <= time
            else { continue }
            let recent = time.timeIntervalSince(entry.notification.arrivedAt) <= Self.seenWindow
            if recent || !entry.arrivedPresent {
                entries[index].seenAt = time
                marked.append(entry.id)
            }
        }
        return marked
    }

    /// Unseen, not ignored by a rule, not yet in front of the owner, and not
    /// expired.
    func unresolved(now: Date) -> [Entry] {
        entries.filter {
            $0.seenAt == nil && $0.rule != .ignore && $0.presentedAt == nil
                && now.timeIntervalSince($0.notification.arrivedAt) < Self.expiry
        }
    }

    /// What a Triage should judge: unresolved, not held by a rule, not judged.
    func untriaged(now: Date) -> [Entry] {
        unresolved(now: now).filter { $0.triagedAt == nil && $0.rule != .hold }
    }

    mutating func markSeen(_ ids: [String], at time: Date) {
        for index in entries.indices
        where ids.contains(entries[index].id) && entries[index].seenAt == nil {
            entries[index].seenAt = time
        }
    }

    func entry(_ id: String) -> Entry? { entries.first { $0.id == id } }

    mutating func markTriaged(_ ids: [String], at time: Date) {
        for index in entries.indices where ids.contains(entries[index].id) {
            entries[index].triagedAt = time
        }
    }

    mutating func markPresented(_ ids: [String], at time: Date) {
        for index in entries.indices where ids.contains(entries[index].id) {
            entries[index].presentedAt = time
        }
    }

    /// Drop what expired a while ago, so the ledger stays small.
    mutating func prune(now: Date) {
        entries.removeAll { now.timeIntervalSince($0.notification.arrivedAt) > 2 * Self.expiry }
    }
}

// MARK: - Rules

/// An owner rule, set in plain words ("never tell me about CI passing") and
/// stored as a simple matcher. Rules apply before the model sees anything.
nonisolated struct TriageRule: Sendable, Equatable, Hashable, Codable, Identifiable {
    enum Action: String, Sendable, Equatable, Codable {
        /// Never show it.
        case ignore
        /// Let it wait for the next Breakpoint; Triage never raises it.
        case hold
        /// Raise it at once, without the model.
        case raise
    }

    var id: String
    /// The app's display name; nil matches any app.
    var app: String?
    /// Matches the banner's title (usually the sender); nil matches any.
    var sender: String?
    /// Every keyword must appear in the banner's text.
    var keywords: [String]
    var action: Action
    /// The owner's own words for it.
    var phrase: String

    init(
        id: String = UUID().uuidString, app: String? = nil, sender: String? = nil,
        keywords: [String] = [], action: Action, phrase: String
    ) {
        self.id = id
        self.app = app
        self.sender = sender
        self.keywords = keywords
        self.action = action
        self.phrase = phrase
    }

    /// A rule must narrow something, or it would swallow every banner.
    var isSpecific: Bool { app != nil || sender != nil || !keywords.isEmpty }

    func matches(_ notification: ObservedNotification) -> Bool {
        guard isSpecific else { return false }
        if let app, notification.app.lowercased() != app.lowercased() { return false }
        if let sender, !notification.title.lowercased().contains(sender.lowercased()) {
            return false
        }
        let text = "\(notification.title) \(notification.subtitle) \(notification.body)"
            .lowercased()
        return keywords.allSatisfy { text.contains($0.lowercased()) }
    }
}

nonisolated enum TriageRules {
    /// The first matching rule wins; rules are kept newest first.
    static func verdict(for notification: ObservedNotification, rules: [TriageRule])
        -> TriageRule.Action?
    {
        rules.first { $0.matches(notification) }?.action
    }

    static func decode(_ json: String) -> [TriageRule] {
        (try? JSONDecoder().decode([TriageRule].self, from: Data(json.utf8))) ?? []
    }

    static func encode(_ rules: [TriageRule]) -> String {
        (try? JSONEncoder().encode(rules)).flatMap { String(data: $0, encoding: .utf8) } ?? "[]"
    }
}
