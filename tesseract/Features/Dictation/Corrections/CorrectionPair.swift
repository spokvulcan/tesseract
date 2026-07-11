//
//  CorrectionPair.swift
//  tesseract
//
//  The **Correction Pair** (map #283, ticket #289): one dictation take's
//  full text lineage — raw ASR, regex-cleaned, proofread output + verdict,
//  what was committed, and the owner's correction — plus the capture
//  conditions and a reference to its Capture Dump audio. Every take is a
//  training-pair candidate; an owner edit or a wrong-flag makes it gold.
//

import Foundation

nonisolated struct CorrectionPair: Codable, Equatable, Identifiable, Sendable {

    /// What the **Proofread Pass** did with this take.
    enum Verdict: String, Codable, Sendable {
        /// The pass didn't run (disabled, model missing, GPU busy, error).
        case skipped
        /// The pass ran and found nothing to fix.
        case unchanged
        /// The pass corrected the text; `proofread` holds its output.
        case corrected
        /// The pass judged the take unintelligible; nothing was committed.
        case rejected
    }

    /// The capture conditions a training consumer needs to weigh the pair.
    struct Conditions: Codable, Equatable, Sendable {
        var duration: TimeInterval
        var language: String
        var asrModel: String
    }

    let id: UUID
    let timestamp: Date
    /// The recognizer's text before any cleanup.
    let rawASR: String
    /// The regex post-processor's output — what the Proofread Pass saw.
    let cleaned: String
    /// The pass's corrected text; `nil` unless `verdict == .corrected`.
    let proofread: String?
    let verdict: Verdict
    /// The pass's rejection reason; `nil` unless `verdict == .rejected`.
    let rejectReason: String?
    /// What was actually injected; `nil` for a rejected take.
    let committed: String?
    /// The owner's hand-corrected text (full editing lives in the history
    /// window) — the gold half of a training pair.
    var correction: String?
    /// One-click "that was wrong" from the overlay affordance.
    var flaggedWrong: Bool
    let conditions: Conditions
    /// The Capture Dump WAV holding this take's audio, when the dump saved
    /// one. A reference, not ownership: the dump remains bounded; gold
    /// pairs' files are exempted from its ring eviction.
    let audioFileName: String?

    /// Gold pairs carry an owner signal (an edit or a wrong-flag) — they are
    /// evicted last and their audio is protected.
    var isGold: Bool { correction != nil || flaggedWrong }

    init(
        id: UUID = UUID(),
        timestamp: Date = Date(),
        rawASR: String,
        cleaned: String,
        proofread: String? = nil,
        verdict: Verdict,
        rejectReason: String? = nil,
        committed: String?,
        correction: String? = nil,
        flaggedWrong: Bool = false,
        conditions: Conditions,
        audioFileName: String? = nil
    ) {
        self.id = id
        self.timestamp = timestamp
        self.rawASR = rawASR
        self.cleaned = cleaned
        self.proofread = proofread
        self.verdict = verdict
        self.rejectReason = rejectReason
        self.committed = committed
        self.correction = correction
        self.flaggedWrong = flaggedWrong
        self.conditions = conditions
        self.audioFileName = audioFileName
    }
}
