//
//  CorrectionPair.swift
//  tesseract
//
//  The **Correction Pair** (map #283, ticket #289): one dictation take's
//  full text lineage — raw ASR, regex-cleaned, after the **Learned Words**,
//  proofread output + verdict, what was committed, and the owner's
//  correction — plus the capture conditions and a reference to its Capture
//  Dump audio. Every take is a training-pair candidate; an owner fix in the
//  **Lens** (PRD #612), an edit or a wrong-flag makes it gold.
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

    /// One word the owner fixed in the **Lens** (PRD #612): what stood there,
    /// what they meant, how the take was reached, and the app it went to.
    struct Fix: Codable, Equatable, Sendable {
        /// Where the fix was made.
        enum How: String, Codable, Sendable {
            /// The take waited in the Lens before pasting.
            case heldTake
            /// ⌃⌥Space reopened the take after it pasted.
            case afterPaste
            /// A take opened from the Dictation page.
            case page
        }

        let heard: String
        let meant: String
        let how: How
        /// The bundle id of the app the take went to, when known.
        let app: String?
        let at: Date
    }

    let id: UUID
    let timestamp: Date
    /// The recognizer's text before any cleanup.
    let rawASR: String
    /// The regex post-processor's output.
    let cleaned: String
    /// `cleaned` after the **Learned Words**; `nil` when they caught
    /// nothing (then the Proofread Pass saw `cleaned`).
    let learned: String?
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
    /// The words the owner fixed in the Lens, oldest first.
    var fixes: [Fix]
    let conditions: Conditions
    /// The Capture Dump WAV holding this take's audio, when the dump saved
    /// one. A reference, not ownership: the dump remains bounded; gold
    /// pairs' files are exempted from its ring eviction.
    let audioFileName: String?

    /// Gold pairs carry an owner signal (a fix, an edit or a wrong-flag) —
    /// they are evicted last and their audio is protected.
    var isGold: Bool { correction != nil || flaggedWrong || !fixes.isEmpty }

    init(
        id: UUID = UUID(),
        timestamp: Date = Date(),
        rawASR: String,
        cleaned: String,
        learned: String? = nil,
        proofread: String? = nil,
        verdict: Verdict,
        rejectReason: String? = nil,
        committed: String?,
        correction: String? = nil,
        flaggedWrong: Bool = false,
        fixes: [Fix] = [],
        conditions: Conditions,
        audioFileName: String? = nil
    ) {
        self.id = id
        self.timestamp = timestamp
        self.rawASR = rawASR
        self.cleaned = cleaned
        self.learned = learned
        self.proofread = proofread
        self.verdict = verdict
        self.rejectReason = rejectReason
        self.committed = committed
        self.correction = correction
        self.flaggedWrong = flaggedWrong
        self.fixes = fixes
        self.conditions = conditions
        self.audioFileName = audioFileName
    }

    private enum CodingKeys: String, CodingKey {
        case id, timestamp, rawASR, cleaned, learned, proofread, verdict, rejectReason
        case committed, correction, flaggedWrong, fixes, conditions, audioFileName
    }

    /// Pairs written before PRD #612 have no `learned` or `fixes`.
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        timestamp = try c.decode(Date.self, forKey: .timestamp)
        rawASR = try c.decode(String.self, forKey: .rawASR)
        cleaned = try c.decode(String.self, forKey: .cleaned)
        learned = try c.decodeIfPresent(String.self, forKey: .learned)
        proofread = try c.decodeIfPresent(String.self, forKey: .proofread)
        verdict = try c.decode(Verdict.self, forKey: .verdict)
        rejectReason = try c.decodeIfPresent(String.self, forKey: .rejectReason)
        committed = try c.decodeIfPresent(String.self, forKey: .committed)
        correction = try c.decodeIfPresent(String.self, forKey: .correction)
        flaggedWrong = try c.decodeIfPresent(Bool.self, forKey: .flaggedWrong) ?? false
        fixes = try c.decodeIfPresent([Fix].self, forKey: .fixes) ?? []
        conditions = try c.decode(Conditions.self, forKey: .conditions)
        audioFileName = try c.decodeIfPresent(String.self, forKey: .audioFileName)
    }
}
