//
//  TickerGapCredit.swift
//  tesseract
//
//  The session ticker's time credit, shared by the endpointer's
//  voiced-seconds accumulator and the Echo Floor's attack/decay integration:
//  each ingest credits the time since the previous one, clamped so a stalled
//  ticker banks no phantom time.
//

import Foundation

nonisolated struct TickerGapCredit {

    /// The widest ingest-to-ingest gap credited — anything longer means the
    /// ticker stalled, not that time passed at the mic.
    static let maxCreditedGap: TimeInterval = 0.25

    private var lastIngestAt: TimeInterval?

    /// The clamped time since the previous credit; 0 on the first.
    mutating func credit(at time: TimeInterval) -> TimeInterval {
        defer { lastIngestAt = time }
        guard let last = lastIngestAt else { return 0 }
        return max(0, min(time - last, Self.maxCreditedGap))
    }

    mutating func reset() {
        lastIngestAt = nil
    }
}
