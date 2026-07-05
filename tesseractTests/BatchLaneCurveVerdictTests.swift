import Foundation
import Testing

@testable import Tesseract_Agent

/// The decode-shape verdict (spike phase 1 of PRD #173): batched matmul
/// ships as a follow-up step function iff, at the pre-registered N=4 on the
/// bench model, batched decode delivers ≥1.8× the single-lane aggregate
/// tok/s AND per-lane latency stays ≤1.5× a single lane's — at the WORST
/// measured context point, all outputs finite. Otherwise the shipped
/// interleaved round-robin stands and batched decode waits.
struct BatchLaneCurveVerdictTests {

    private func measurement(
        shape: BatchLaneCurveSupport.Shape,
        lanes: Int,
        context: Int,
        roundMs: Double,
        finite: Bool = true
    ) -> BatchLaneCurveSupport.Measurement {
        BatchLaneCurveSupport.Measurement(
            model: "small",
            shape: shape.rawValue,
            lanes: lanes,
            contexts: Array(repeating: context, count: lanes),
            roundMs: roundMs,
            aggregateTokPerS: Double(lanes) / (roundMs / 1000),
            outputsFinite: finite
        )
    }

    @Test func batchedWinPassesBothPreRegisteredBars() throws {
        // Single lane: 20 ms/step → 50 tok/s. Batched N=4: 25 ms/round →
        // 160 tok/s aggregate (3.2×), per-lane latency 1.25×.
        let verdict = BatchLaneCurveSupport.verdict(measurements: [
            measurement(shape: .interleaved, lanes: 1, context: 2048, roundMs: 20),
            measurement(shape: .batched, lanes: 4, context: 2048, roundMs: 25),
        ])
        #expect(verdict.batchedDecodeJustified)
        let cell = try #require(verdict.cells.first)
        #expect(abs(cell.aggregateRatio - 3.2) < 0.001)
        #expect(abs(cell.perLaneLatencyRatio - 1.25) < 0.001)
        #expect(cell.passed)
    }

    @Test func insufficientAggregateGainFailsTheVerdict() {
        // Batched N=4 at 48 ms/round → 83.3 tok/s vs 50 single-lane: only
        // 1.67× — below the 1.8× bar even though latency (2.4×)… both fail.
        let verdict = BatchLaneCurveSupport.verdict(measurements: [
            measurement(shape: .interleaved, lanes: 1, context: 2048, roundMs: 20),
            measurement(shape: .batched, lanes: 4, context: 2048, roundMs: 48),
        ])
        #expect(!verdict.batchedDecodeJustified)
    }

    @Test func latencyBlowupFailsEvenWithAggregateWin() {
        // 4 lanes at 31 ms/round = 129 tok/s (2.58× aggregate ✓) but
        // per-lane latency 1.55× — over the 1.5× bar.
        let verdict = BatchLaneCurveSupport.verdict(measurements: [
            measurement(shape: .interleaved, lanes: 1, context: 2048, roundMs: 20),
            measurement(shape: .batched, lanes: 4, context: 2048, roundMs: 31),
        ])
        #expect(!verdict.batchedDecodeJustified)
    }

    @Test func worstContextPointDecides() {
        // Short context flies (4×), long context crawls (1.6×) — the gate
        // takes the worst cell, so the verdict fails.
        let verdict = BatchLaneCurveSupport.verdict(measurements: [
            measurement(shape: .interleaved, lanes: 1, context: 2048, roundMs: 20),
            measurement(shape: .batched, lanes: 4, context: 2048, roundMs: 20),
            measurement(shape: .interleaved, lanes: 1, context: 8192, roundMs: 25),
            measurement(shape: .batched, lanes: 4, context: 8192, roundMs: 62.5),
        ])
        #expect(verdict.cells.count == 2)
        #expect(!verdict.batchedDecodeJustified)
    }

    @Test func nonFiniteOutputsFailTheStabilityBar() {
        let verdict = BatchLaneCurveSupport.verdict(measurements: [
            measurement(shape: .interleaved, lanes: 1, context: 2048, roundMs: 20),
            measurement(
                shape: .batched, lanes: 4, context: 2048, roundMs: 25, finite: false),
        ])
        #expect(!verdict.batchedDecodeJustified)
    }

    @Test func missingCounterpartsProduceNoVerdictCells() {
        // Only interleaved data (no batched N=4 run): nothing to judge —
        // explicitly not justified rather than vacuously passed.
        let verdict = BatchLaneCurveSupport.verdict(measurements: [
            measurement(shape: .interleaved, lanes: 1, context: 2048, roundMs: 20),
            measurement(shape: .interleaved, lanes: 4, context: 2048, roundMs: 80),
        ])
        #expect(verdict.cells.isEmpty)
        #expect(!verdict.batchedDecodeJustified)
    }
}
