//
//  CapturedNotification.swift
//  tesseract
//
//  Another app's banner, as the Notification Center watcher read it, and the
//  one door that turns it into an observed notification: Tesseract's own
//  banners never count, and repeats of the same banner collapse on one id.
//

import Foundation

/// One banner as the AX layer read it. Fields are best-effort: the tree
/// exposes a display name (never a bundle id) and may only carry the
/// flattened description, so any field may be empty.
nonisolated struct CapturedNotification: Sendable, Equatable {
    /// Source-app display name, the only identity the tree exposes.
    let app: String
    let title: String
    let subtitle: String
    let body: String
    /// The banner's `AXIdentifier`, a stable per-notification UUID when the
    /// tree carried one.
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

    /// Bodies longer than this are cut for the model; triage needs the gist.
    static let bodyCap = 500

    /// The admission door, or nil to drop the banner. Tesseract's own banners
    /// are excluded by display name, including the collapsed-stack form
    /// ("Stacked summary" with "Tesseract Agent: …" leading the title). The id
    /// comes from the banner's UUID, or from its content when there is none,
    /// so the same banner seen twice collapses.
    func admitted(selfDisplayNames: Set<String>) -> ObservedNotification? {
        let app = app.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !app.isEmpty else { return nil }
        let excluded = Set(selfDisplayNames.map { $0.lowercased() })
        guard !excluded.contains(app.lowercased()) else { return nil }

        let title = title.trimmingCharacters(in: .whitespacesAndNewlines)
        if app.lowercased() == "stacked summary",
            let leading = title.split(separator: ":").first.map({
                $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            }),
            excluded.contains(leading)
        {
            return nil
        }
        let subtitle = subtitle.trimmingCharacters(in: .whitespacesAndNewlines)
        let body = body.trimmingCharacters(in: .whitespacesAndNewlines)
        let id =
            uuid.map { "notification:\($0)" } ?? "notification:\(app)|\(title)|\(subtitle)|\(body)"
        let capped = body.count > Self.bodyCap ? String(body.prefix(Self.bodyCap)) + "…" : body
        return ObservedNotification(
            id: id, app: app, title: title, subtitle: subtitle, body: capped,
            arrivedAt: occurredAt)
    }
}

/// A banner admitted into the seen ledger.
nonisolated struct ObservedNotification: Sendable, Equatable, Hashable, Codable, Identifiable {
    let id: String
    let app: String
    let title: String
    let subtitle: String
    let body: String
    let arrivedAt: Date
    /// Who it is from, as code classified it on arrival; nil for a banner
    /// admitted before sources existed (it is treated as a person's).
    let source: NotificationSource?

    init(
        id: String, app: String, title: String, subtitle: String, body: String,
        arrivedAt: Date, source: NotificationSource? = nil
    ) {
        self.id = id
        self.app = app
        self.title = title
        self.subtitle = subtitle
        self.body = body
        self.arrivedAt = arrivedAt
        self.source = source
    }

    /// The same banner with its source classified.
    func classified(_ source: NotificationSource) -> ObservedNotification {
        ObservedNotification(
            id: id, app: app, title: title, subtitle: subtitle, body: body, arrivedAt: arrivedAt,
            source: source)
    }

    /// "App: title — body", the line the model and the card read.
    var line: String {
        let tail = [title, subtitle, body].filter { !$0.isEmpty }.joined(separator: " — ")
        return tail.isEmpty ? app : "\(app): \(tail)"
    }
}
