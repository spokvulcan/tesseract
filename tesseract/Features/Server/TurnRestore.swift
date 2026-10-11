//
//  TurnRestore.swift
//  tesseract
//
//  The **Turn Restore** (ADR-0069's 2026-10-11 amendment): how one keyed
//  turn's cache opened, as plan application carried it out — the Cache
//  Claim's typed check-out outcome, plus the restore that yielded no cache
//  and ran cold. The one value every report of the turn reads: the `lookup`
//  line, the `restored` memory phase, the `leafStore` report, the leaf's
//  source for the Active-Inference Reserve and the trace record. It replaces
//  the wire string and the loose values the outcome used to be flattened
//  into.
//

import Foundation

nonisolated enum TurnRestore: Sendable, Equatable {
    /// No restore was planned: the turn prefilled from a fresh cache.
    case cold
    /// The claim took the leaf by **Leaf Handoff**; the turn generates into
    /// its objects. `waitSeconds` is the Pending-Payload Wait it ended.
    case handoff(Handoff)
    /// Restored by copy, with why the claim did not hand the leaf over (`nil`
    /// when it had nothing to check out).
    case copy(CacheClaim.Copy?)
    /// A planned restore that yielded no cache, so the turn ran cold
    /// (ADR-0069's 2026-10-03 amendment). The copy is the claim's answer
    /// before the restore failed. What the restore threw and what the
    /// failed-restore policy dropped are the policy's own line
    /// (`restore/unrestorable-snapshot`); no report reads them from here.
    case failedThenCold(CacheClaim.Copy?)

    struct Handoff: Sendable, Equatable {
        let leaseID: UUID
        let rewindStateBytes: Int
        let waitSeconds: TimeInterval
    }

    /// The `restoreMode` wire string.
    var mode: String {
        switch self {
        case .cold: "cold"
        case .handoff: "handoff"
        case .copy: "copy"
        case .failedThenCold: "failedCopy"
        }
    }

    /// The claim's copy answer, when the turn restored (or tried to) by copy.
    var copy: CacheClaim.Copy? {
        switch self {
        case .cold, .handoff: nil
        case .copy(let copy), .failedThenCold(let copy): copy
        }
    }

    /// Seconds a Pending-Payload Wait cost, whether it ended in a copy or a
    /// handoff (#523).
    var waitSeconds: TimeInterval {
        switch self {
        case .handoff(let handoff): handoff.waitSeconds
        default: copy?.waitSeconds ?? 0
        }
    }

    /// Whether a planned restore fell back to a cold prefill.
    var fellBack: Bool {
        if case .failedThenCold = self { true } else { false }
    }

    /// Whether the turn began from a restored cache, by handoff or by copy.
    var restored: Bool {
        switch self {
        case .handoff, .copy: true
        case .cold, .failedThenCold: false
        }
    }

    /// The `restored` memory phase's restore facts (beside the cache's own).
    var memoryFacts: [String: String] {
        var facts = [
            "restoreMode": mode,
            "restoreCopyReason": copy?.reason.rawValue ?? "none",
            "restoreCopyRefusal": copy?.refusal.rawValue ?? "none",
            "restoreCopyWaitMs": PrefixCacheDiagnostics.milliseconds(waitSeconds),
            "recurrentRewindStateBytes": "0",
            "leafLeaseActive": "false",
            "leafLeaseID": "none",
        ]
        if case .handoff(let handoff) = self {
            facts["recurrentRewindStateBytes"] = "\(handoff.rewindStateBytes)"
            facts["leafLeaseActive"] = "true"
            facts["leafLeaseID"] = handoff.leaseID.uuidString
        }
        if fellBack { facts["restoreFallback"] = "cold" }
        return facts
    }
}
