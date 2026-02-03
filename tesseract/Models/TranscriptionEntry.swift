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

    init(
        id: UUID = UUID(),
        text: String,
        timestamp: Date = Date(),
        duration: TimeInterval,
        model: String
    ) {
        self.id = id
        self.text = text
        self.timestamp = timestamp
        self.duration = duration
        self.model = model
    }
}
