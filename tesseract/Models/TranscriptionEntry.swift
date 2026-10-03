//
//  TranscriptionEntry.swift
//  tesseract
//

import Foundation

nonisolated struct TranscriptionEntry: Identifiable, Codable, Sendable {
    /// The app a take went to, as the history keeps it: enough for the
    /// **Lens** (PRD #612) to name the app and leave a Learned Word alone
    /// there when a take is fixed from the Dictation page.
    struct App: Codable, Equatable, Sendable {
        let bundleID: String?
        let name: String
    }

    let id: UUID
    /// What the take wrote; a fix in the **Lens** rewrites it.
    var text: String
    let timestamp: Date
    let duration: TimeInterval
    let model: String
    /// The **Correction Pair** this entry's take was recorded as; `nil` for
    /// entries predating the flywheel (ticket #289).
    let pairID: UUID?
    /// What the **Learned Words** caught in `text`, positioned in it; empty
    /// for entries predating them. A fix in the Lens moves them with it.
    var catches: [LearnedWordCatch]
    /// The app in front when the take started; nil when unknown.
    let app: App?

    init(
        id: UUID = UUID(),
        text: String,
        timestamp: Date = Date(),
        duration: TimeInterval,
        model: String,
        pairID: UUID? = nil,
        catches: [LearnedWordCatch] = [],
        app: App? = nil
    ) {
        self.id = id
        self.text = text
        self.timestamp = timestamp
        self.duration = duration
        self.model = model
        self.pairID = pairID
        self.catches = catches
        self.app = app
    }

    private enum CodingKeys: String, CodingKey {
        case id, text, timestamp, duration, model, pairID, catches, app
    }

    /// Entries written before PRD #612 have no `catches` or `app`.
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        text = try c.decode(String.self, forKey: .text)
        timestamp = try c.decode(Date.self, forKey: .timestamp)
        duration = try c.decode(TimeInterval.self, forKey: .duration)
        model = try c.decode(String.self, forKey: .model)
        pairID = try c.decodeIfPresent(UUID.self, forKey: .pairID)
        catches = try c.decodeIfPresent([LearnedWordCatch].self, forKey: .catches) ?? []
        app = try c.decodeIfPresent(App.self, forKey: .app)
    }
}
