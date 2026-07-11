//
//  TranscriptionEntry.swift
//  tesseract
//

import Foundation

struct TranscriptionEntry: Identifiable, Codable, Sendable {
    let id: UUID
    let text: String
    let timestamp: Date
    let duration: TimeInterval
    let model: String
    /// The **Correction Pair** this entry's take was recorded as; `nil` for
    /// entries predating the flywheel (ticket #289).
    let pairID: UUID?

    init(
        id: UUID = UUID(),
        text: String,
        timestamp: Date = Date(),
        duration: TimeInterval,
        model: String,
        pairID: UUID? = nil
    ) {
        self.id = id
        self.text = text
        self.timestamp = timestamp
        self.duration = duration
        self.model = model
        self.pairID = pairID
    }
}
