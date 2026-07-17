//
//  CompanionTraceLine.swift
//  tesseract
//
//  The flight recorder's line schema (#326): flat discriminated JSONL,
//  schema-versioned per file (the `CompletionTraceLine` / ADR-0031
//  precedent), `jq`/DuckDB-minable. Replay-grade means verbatim for
//  everything variable (the snapshot) and by-reference for everything
//  policy (instructions version, model id).
//
//  Only the app writes these. The model reads via a typed tool and adds
//  testimony through `log_feedback` — stamped `model-reported`, never
//  edited history.
//

import Foundation

// MARK: - Header

nonisolated struct CompanionTraceHeader: Codable, Sendable {
    let schemaVersion: Int
    let createdAt: TimeInterval
}

// MARK: - Record

nonisolated struct CompanionTraceRecord: Codable, Sendable {
    static let currentSchemaVersion = 1

    /// Unix seconds — the store-wide convention; `jq` and SQLite both read it.
    let ts: TimeInterval
    /// Dot-namespaced event, e.g. `wake.booked`, `turn.failed`,
    /// `delivery.spoken`, `reaction.dismissed`, `feedback.solicited`.
    let event: String
    /// `app-observed` (code saw it happen) vs `model-reported` (testimony).
    let source: String

    // Correlation — whichever apply.
    let wakeID: String?
    let turnID: String?
    let conversationID: String?

    // Policy by version reference (#326): the standing-instructions version
    // and the model that acted under it.
    let policyVersion: String?
    let modelID: String?

    /// Verbatim variable inputs at the decision seam — presence state,
    /// schedule slot, contract state, the composed line, the spoken words.
    let snapshot: [String: String]?
    let note: String?

    init(
        ts: TimeInterval,
        event: String,
        source: CompanionTraceSource,
        wakeID: String? = nil,
        turnID: String? = nil,
        conversationID: String? = nil,
        policyVersion: String? = nil,
        modelID: String? = nil,
        snapshot: [String: String]? = nil,
        note: String? = nil
    ) {
        self.ts = ts
        self.event = event
        self.source = source.rawValue
        self.wakeID = wakeID
        self.turnID = turnID
        self.conversationID = conversationID
        self.policyVersion = policyVersion
        self.modelID = modelID
        self.snapshot = snapshot
        self.note = note
    }
}

nonisolated enum CompanionTraceSource: String, Codable, Sendable {
    case appObserved = "app-observed"
    case modelReported = "model-reported"
}

// MARK: - Line (header | record)

nonisolated enum CompanionTraceLine: Sendable {
    case header(CompanionTraceHeader)
    case record(CompanionTraceRecord)

    static func decode(_ data: Data, decoder: JSONDecoder = JSONDecoder()) -> CompanionTraceLine? {
        if let record = try? decoder.decode(CompanionTraceRecord.self, from: data) {
            return .record(record)
        }
        if let header = try? decoder.decode(CompanionTraceHeader.self, from: data) {
            return .header(header)
        }
        return nil
    }
}
