//
//  TranscriptionResult.swift
//  whisper-on-device
//

import Foundation

struct TranscriptionSegment: Sendable {
    let text: String
    let startTime: TimeInterval
    let endTime: TimeInterval
}

struct TranscriptionResult: Sendable {
    let text: String
    let segments: [TranscriptionSegment]
    let language: String
    let processingTime: TimeInterval

    static let empty = TranscriptionResult(
        text: "",
        segments: [],
        language: "en",
        processingTime: 0
    )
}
