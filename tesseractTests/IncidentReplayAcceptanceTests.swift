import Foundation
import Testing

@testable import Tesseract_Agent

/// Interrupt-readiness acceptance over the archived 2026-06-12 incident
/// corpus (PRD #94, issue #101). Opt-in: the trace lives outside the repo
/// (`~/projects/tesseract-traces/2026-06-12-interrupt-rewind/trace-2026-06-12.jsonl`,
/// user project content) and is supplied through `TESSERACT_FIDELITY_CORPUS`,
/// the same env var the canonical-echo corpus gate reads. See
/// `docs/testing.md` and `scripts/interrupt-drill.sh`.
///
/// The corpus assertions are the regression net the dashboard's rewind
/// telemetry backs: the restore floor never overshoots the divergence, and
/// the replay is deterministic so steady-state hit rate / token reuse move
/// only when behaviour does. The live TTFT proof is the drill script — it
/// needs the model and the incident hardware, not a unit host.
@MainActor
struct IncidentReplayAcceptanceTests {

    private nonisolated static var traceURL: URL? {
        guard let root = ProcessInfo.processInfo.environment["TESSERACT_FIDELITY_CORPUS"]
        else { return nil }
        return URL(fileURLWithPath: NSString(string: root).expandingTildeInPath)
            .appendingPathComponent("trace-2026-06-12.jsonl")
    }

    private nonisolated static var traceAvailable: Bool {
        guard let traceURL else { return false }
        return FileManager.default.fileExists(atPath: traceURL.path)
    }

    @Test(.enabled(if: traceAvailable))
    func restoreFloorNeverOvershootsTheDivergence() throws {
        let records = CompletionTraceLog.readRecords(at: [Self.traceURL!])
        try #require(!records.isEmpty, "incident trace decoded zero records")

        // The restore floor (`restoredOffset`) is the deepest boundary
        // at-or-below where the request diverged from the cached path
        // (`sharedPrefixLength`). A floor past the divergence would mean
        // restoring tokens the request does not share — the bug the gate
        // forbids.
        let overshoots = records.filter { $0.restoredOffset > $0.sharedPrefixLength }
        let sample = overshoots.prefix(3)
            .map { "\($0.restoredOffset)>\($0.sharedPrefixLength)" }
        #expect(
            overshoots.isEmpty,
            "restore floor overshot the divergence on \(overshoots.count) record(s): \(sample)"
        )
    }

    @Test(.enabled(if: traceAvailable))
    func incidentCorpusReplayIsDeterministicAndSurfacesRewindMetrics() throws {
        let records = CompletionTraceLog.readRecords(at: [Self.traceURL!])
        try #require(!records.isEmpty, "incident trace decoded zero records")

        let first = TraceReplayHarness.replay(records: records)
        let second = TraceReplayHarness.replay(records: records)
        #expect(first == second, "replay over the incident corpus must be deterministic")

        // Steady-state reuse signals are computed (the no-regression
        // yardstick a future change is measured against), and the rewind
        // roll-up the dashboard reads is present and self-consistent.
        #expect(first.observed.totalHitTokens >= 0)
        #expect(first.recordCount == records.count)
        #expect(first.observed.rewindEvents >= 0)
        #expect(first.observed.rewindSizeP95Tokens >= first.observed.rewindSizeP50Tokens)
        #expect(first.observed.maxRewindSizeTokens >= first.observed.rewindSizeP95Tokens)
    }
}
