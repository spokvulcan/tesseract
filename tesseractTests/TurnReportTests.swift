import Foundation
import Testing

@testable import Tesseract_Agent

/// The **Turn Report** (ADR-0069's 2026-10-11 amendment) through its
/// interface: the turn's one `leafStore` event — the claim's Leaf Rewind
/// before the Leaf Store phase, the phase's report from it on — and the filed
/// Turn Restore every output reads. The whole-completion suites
/// (`EmittedPathSynthesizedReplayTests`' cancelled turns, the keyed
/// sequencing suite) check the report through a real turn.
struct TurnReportTests {

    private static func report(_ modelID: String) -> TurnReport {
        TurnReport(
            diagnostics: .init(requestID: UUID(), modelID: modelID, kvBits: nil, kvGroupSize: 64))
    }

    private static func leafStores(_ events: [PromptCacheTelemetryEvent]) -> [[String: String]] {
        events.filter { $0.eventName == "leafStore" }.map {
            Dictionary(uniqueKeysWithValues: $0.fields.map { ($0.key, $0.value) })
        }
    }

    @Test func aRewindBeforeThePhaseIsTheTurnsOneLeafStore() {
        let modelID = "turn-report-rewind-\(UUID().uuidString)"
        let capture = TelemetryCapture(modelID: modelID)
        defer { capture.stop() }
        let report = Self.report(modelID)
        let lease = UUID()

        report.leafRewound(leaseID: lease, rewindStateBytes: 64, offset: 12)

        let stores = Self.leafStores(capture.drain())
        #expect(stores.count == 1)
        #expect(stores.first?["path"] == "rewind")
        #expect(stores.first?["source"] == "rewind")
        #expect(stores.first?["restoreMode"] == "handoff")
        #expect(stores.first?["leafOffset"] == "12")
        #expect(stores.first?["copyWaitMs"] == nil)
    }

    /// Once the phase has begun its report says what became of the leaf, so
    /// a rewind in the phase or at the claim's conclusion logs no second
    /// `leafStore`.
    @Test func fromThePhaseOnTheRewindLogsNoSecondLeafStore() {
        let modelID = "turn-report-phase-\(UUID().uuidString)"
        let capture = TelemetryCapture(modelID: modelID)
        defer { capture.stop() }
        let report = Self.report(modelID)
        report.file(.handoff(.init(leaseID: UUID(), rewindStateBytes: 64, waitSeconds: 0.5)))

        report.beginLeafStorePhase()
        report.leafRewound(leaseID: UUID(), rewindStateBytes: 64, offset: 12)
        var phase = LeafStorePhase.Report()
        phase.mode = "keyed"
        phase.path = .rewind
        phase.leafOffset = 12
        report.logLeafStore(report.filed(phase))

        let stores = Self.leafStores(capture.drain())
        #expect(stores.count == 1)
        #expect(stores.first?["restoreMode"] == "handoff")
        #expect(stores.first?["copyWaitMs"] == "500.000")
    }

    /// The phase's report reads the filed restore, and so does the leaf's
    /// source the reserve prices a lane by.
    @Test func thePhasesReportReadsTheFiledRestore() {
        let report = Self.report("turn-report-restore")
        #expect(report.restore == .cold)
        let copy = CacheClaim.Copy(refusal: .imageKeySpace, waitSeconds: 0)
        report.file(.copy(copy))

        var phase = LeafStorePhase.Report()
        phase.path = .live
        let filed = report.filed(phase)

        #expect(filed.restore == .copy(copy))
        #expect(filed.source == .copy)
        let fields = Dictionary(uniqueKeysWithValues: filed.fields)
        #expect(fields["restoreMode"] == "copy")
        #expect(fields["copyReason"] == "imageKeySpace")
    }
}
