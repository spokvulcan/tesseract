import Foundation
import Testing

@testable import Tesseract_Agent

/// The replay gate's rules (ticket #477), one failure each, on hand-built
/// turn accounts: every failure names its turn and carries the whole
/// account — leaf source and reason, registration, path length, prefilled
/// count and tail — so a red corpus run reads without re-running the walk.
struct EmittedPathReplayGateTests {

    private typealias Verdict = CanonicalEchoFidelity.EmittedPathVerdict
    private typealias Turn = EmittedPathReplayGate.TurnAccount

    /// A turn the gate passes: live, registered, the whole path served,
    /// six glue tokens, a fast tail.
    private static func verdict(
        registration: String = "registered",
        pathLength: Int? = 1_000,
        mode: HTTPLeafStoreMode = .directToolLeaf,
        source: LeafStorePhase.Report.Source = .live,
        boundaryReason: String? = nil,
        overwrote: Bool = false,
        nextIndexedPrefix: Int? = 1_000,
        nextPrefilled: Int? = 46,
        nextNewMessageTokens: Int? = 40,
        tailSeconds: Double? = 0.010
    ) -> Verdict {
        Verdict(
            registration: registration, pathLength: pathLength, mode: mode.rawValue,
            source: source.rawValue, boundaryReason: boundaryReason, overwrote: overwrote,
            nextIndexedPrefix: nextIndexedPrefix, nextSuffixTokens: nextPrefilled,
            nextMissReason: nextIndexedPrefix == nil ? "noEntry" : nil,
            nextPrefilled: nextPrefilled, nextNewMessageTokens: nextNewMessageTokens,
            tailSeconds: tailSeconds, renderSeconds: tailSeconds, registerSeconds: 0)
    }

    private static func turn(
        _ verdict: Verdict, index: Int = 7,
        kind: CanonicalEchoFidelity.Boundary.Kind = .toolContinuation
    ) -> Turn {
        Turn(requestIndex: index, kind: kind, verdict: verdict)
    }

    private static func rules(_ turn: Turn) -> [EmittedPathReplayGate.Failure.Rule] {
        EmittedPathReplayGate.check(turn).map(\.rule)
    }

    @Test func aFaithfulLiveTurnPasses() {
        #expect(EmittedPathReplayGate.check([Self.turn(Self.verdict())]).isEmpty)
    }

    @Test func aHandedOffTurnPassesAndStillRequiresRegistration() {
        #expect(Self.rules(Self.turn(Self.verdict(source: .handoff))).isEmpty)
        #expect(
            Self.rules(
                Self.turn(
                    Self.verdict(
                        registration: "ineligibleRender", source: .handoff))) == [.registration])
    }

    @Test func aToolStretchTurnFromTheBoundaryFailsTheLeafSource() throws {
        let turn = Self.turn(
            Self.verdict(
                registration: "intervened", pathLength: nil, source: .boundary,
                boundaryReason: "intervened", nextIndexedPrefix: nil))
        let failures = EmittedPathReplayGate.check(turn)
        #expect(failures.map(\.rule) == [.leafSource])
        let description = try #require(failures.first).description
        #expect(description.contains("request#7"))
        #expect(description.contains("source=boundary"))
        #expect(description.contains("boundary=intervened"))
        #expect(description.contains("registration=intervened"))
        #expect(description.contains("prefilled="))
        #expect(description.contains("tailMs="))
    }

    @Test func aStopTurnAtTheThinkStrippingBoundaryIsExplained() {
        let explained = Self.turn(
            Self.verdict(
                registration: "thinkStrippingUserBoundary", pathLength: nil,
                mode: .canonicalUserLeaf, source: .boundary,
                boundaryReason: "thinkStrippingUserBoundary", nextIndexedPrefix: nil),
            kind: .canonicalUser)
        #expect(Self.rules(explained).isEmpty)
        let unexplained = Self.turn(
            Self.verdict(
                registration: "noGeneratedTokens", pathLength: nil,
                mode: .canonicalUserLeaf, source: .boundary,
                boundaryReason: "noGeneratedTokens", nextIndexedPrefix: nil),
            kind: .canonicalUser)
        #expect(Self.rules(unexplained) == [.leafSource])
    }

    @Test func aLiveTurnThatRegisteredNothingFails() {
        let skipped = Self.turn(
            Self.verdict(registration: "renderUnavailable", pathLength: nil, nextIndexedPrefix: nil)
        )
        #expect(Self.rules(skipped) == [.registration])
        // The harness's own limitation is tolerated and reported, not failed.
        let unsimulated = Self.turn(
            Self.verdict(
                registration: EmittedPathReplayGate.promptNotTokenPrefix, pathLength: nil,
                nextIndexedPrefix: nil))
        #expect(Self.rules(unsimulated).isEmpty)
    }

    @Test func aFidelityRejectionFails() {
        let rejected = Self.turn(
            Self.verdict(registration: "fidelityRejected", pathLength: nil, nextIndexedPrefix: nil))
        #expect(Self.rules(rejected) == [.fidelity])
    }

    @Test func aSameKeyOverwriteFails() {
        #expect(Self.rules(Self.turn(Self.verdict(overwrote: true))) == [.overwrite])
    }

    @Test func aShallowHitFailsThePrefillAndTheGlue() {
        // 100 of the stored turn re-prefilled: the prefix is short and the
        // glue is over the allowance by the same 100.
        let shallow = Self.turn(
            Self.verdict(nextIndexedPrefix: 900, nextPrefilled: 146, nextNewMessageTokens: 40))
        #expect(Self.rules(shallow) == [.prefill, .glue])
        let miss = Self.turn(
            Self.verdict(nextIndexedPrefix: nil, nextPrefilled: 1_046, nextNewMessageTokens: 40))
        let failures = EmittedPathReplayGate.check(miss)
        #expect(failures.map(\.rule) == [.prefill, .glue])
        #expect(failures.first?.detail == "next request served 0 of 1000 path tokens")
        #expect(failures.last?.description.contains("glue=1006") == true)
    }

    @Test func theGlueAllowanceIsSixTokens() {
        #expect(EmittedPathReplayGate.glueTokenAllowance == 6)
        let atAllowance = Self.turn(Self.verdict(nextPrefilled: 46, nextNewMessageTokens: 40))
        #expect(Self.rules(atAllowance).isEmpty)
        let overAllowance = Self.turn(Self.verdict(nextPrefilled: 47, nextNewMessageTokens: 40))
        #expect(Self.rules(overAllowance) == [.glue])
        let unmeasured = Self.turn(Self.verdict(nextPrefilled: nil))
        #expect(Self.rules(unmeasured) == [.glue])
    }

    @Test func aSlowTailFailsBelowTwentyThousandTokensOnly() {
        let slow = Self.turn(Self.verdict(tailSeconds: 0.150))
        let failures = EmittedPathReplayGate.check(slow)
        #expect(failures.map(\.rule) == [.tail])
        #expect(failures.first?.detail == "150.0 ms, budget 150.0 ms")
        let fast = Self.turn(Self.verdict(tailSeconds: 0.149))
        #expect(Self.rules(fast).isEmpty)
        let long = Self.turn(
            Self.verdict(pathLength: 20_000, nextIndexedPrefix: 20_000, tailSeconds: 0.400))
        #expect(Self.rules(long).isEmpty)
        let unmeasured = Self.turn(Self.verdict(tailSeconds: nil))
        #expect(Self.rules(unmeasured) == [.tail])
    }

    @Test func aSessionReportWithoutAnIndexHasNoTurns() {
        let report = CanonicalEchoFidelity.SessionReport(
            sessionAffinity: nil, boundaries: [], skipped: [])
        #expect(EmittedPathReplayGate.check(report).isEmpty)
    }
}
