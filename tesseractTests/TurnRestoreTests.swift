import Foundation
import Testing

@testable import Tesseract_Agent

/// The **Turn Restore** (ADR-0069's 2026-10-11 amendment): one keyed turn's
/// restore outcome, and the wire projections every report of the turn reads
/// from it — the `restoreMode`, the copy reason, refusal and wait, the
/// fallback flag, the `restored` memory phase's facts, and the stored leaf's
/// source. Pure rows, no completion: the whole-completion suites check that
/// plan application produces the right outcome.
struct TurnRestoreTests {

    private static let lease = UUID()
    private static let pending = CacheClaim.Copy(refusal: .pendingFullPayload, waitSeconds: 0.25)
    private static let disabled = CacheClaim.Copy(refusal: .checkoutDisabled, waitSeconds: 0)

    struct Row: Sendable, CustomTestStringConvertible {
        let restore: TurnRestore
        let mode: String
        let copyReason: String
        let copyRefusal: String
        let waitMs: String
        let leaseID: String
        let rewindBytes: String
        let fellBack: Bool
        var testDescription: String { "\(restore)" }
    }

    static let rows: [Row] = [
        Row(
            restore: .cold, mode: "cold", copyReason: "none", copyRefusal: "none",
            waitMs: "0.000", leaseID: "none", rewindBytes: "0", fellBack: false),
        Row(
            restore: .handoff(.init(leaseID: lease, rewindStateBytes: 64, waitSeconds: 0.5)),
            mode: "handoff", copyReason: "none", copyRefusal: "none", waitMs: "500.000",
            leaseID: lease.uuidString, rewindBytes: "64", fellBack: false),
        Row(
            restore: .copy(nil), mode: "copy", copyReason: "none", copyRefusal: "none",
            waitMs: "0.000", leaseID: "none", rewindBytes: "0", fellBack: false),
        Row(
            restore: .copy(pending), mode: "copy", copyReason: "pendingFullPayload",
            copyRefusal: "pendingFullPayload", waitMs: "250.000", leaseID: "none",
            rewindBytes: "0", fellBack: false),
        Row(
            restore: .failedThenCold(nil), mode: "failedCopy",
            copyReason: "none", copyRefusal: "none", waitMs: "0.000", leaseID: "none",
            rewindBytes: "0", fellBack: true),
        Row(
            restore: .failedThenCold(disabled), mode: "failedCopy",
            copyReason: "checkoutDisabled", copyRefusal: "checkoutDisabled", waitMs: "0.000",
            leaseID: "none", rewindBytes: "0", fellBack: true),
    ]

    @Test(arguments: rows)
    func everyReportReadsTheSameOutcome(_ row: Row) {
        #expect(row.restore.mode == row.mode)
        #expect(row.restore.fellBack == row.fellBack)
        let facts = row.restore.memoryFacts
        #expect(facts["restoreMode"] == row.mode)
        #expect(facts["restoreCopyReason"] == row.copyReason)
        #expect(facts["restoreCopyRefusal"] == row.copyRefusal)
        #expect(facts["restoreCopyWaitMs"] == row.waitMs)
        #expect(facts["leafLeaseID"] == row.leaseID)
        #expect(facts["leafLeaseActive"] == (row.leaseID == "none" ? "false" : "true"))
        #expect(facts["recurrentRewindStateBytes"] == row.rewindBytes)
        #expect(facts["restoreFallback"] == (row.fellBack ? "cold" : nil))

        // The `leafStore` report renders the same restore.
        var report = LeafStorePhase.Report()
        report.restore = row.restore
        let fields = Dictionary(uniqueKeysWithValues: report.fields)
        #expect(fields["restoreMode"] == row.mode)
        #expect(fields["copyReason"] == (row.copyReason == "none" ? nil : row.copyReason))
        #expect(fields["copyRefusal"] == (row.copyRefusal == "none" ? nil : row.copyRefusal))
        #expect(fields["copyWaitMs"] == (row.waitMs == "0.000" ? nil : row.waitMs))
    }

    /// The stored leaf's source, which the reserve prices a lane by: a
    /// live or direct leaf after a copy restore reads `copy` (its source body
    /// is already counted); otherwise it reads what the capture did.
    @Test func theStoredLeafsSourceFollowsThePathAndTheRestore() {
        typealias Source = LeafStorePhase.Report.Source
        let failed = TurnRestore.failedThenCold(nil)
        for path in [LeafStorePhase.Report.Path.live, .direct] {
            #expect(Source.stored(path: path, restore: .copy(nil), handedOff: false) == .copy)
            #expect(Source.stored(path: path, restore: failed, handedOff: true) == .copy)
            #expect(Source.stored(path: path, restore: .cold, handedOff: true) == .handoff)
            #expect(Source.stored(path: path, restore: .cold, handedOff: false) == .live)
            #expect(Source.stored(path: path, restore: nil, handedOff: false) == .live)
        }
        #expect(Source.stored(path: .boundary, restore: .copy(nil), handedOff: true) == .handoff)
        #expect(Source.stored(path: .boundary, restore: .cold, handedOff: false) == .boundary)
        #expect(Source.stored(path: .rewind, restore: .cold, handedOff: false) == .rewind)
        #expect(Source.stored(path: .skipped, restore: .cold, handedOff: false) == nil)
    }
}
