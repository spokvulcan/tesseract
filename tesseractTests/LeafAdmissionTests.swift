//
//  LeafAdmissionTests.swift
//  tesseractTests
//
//  The **Leaf Admission** (ADR-0078) through its interface: prepared, then
//  admitted inside a toy Model Session over a real manager. Covers the three
//  ownership cases and the move-or-copy rule, a cache the capture cannot
//  take, and one classification per admission, whose tally a request merges
//  without logging again. Every producer's stage labels are pinned byte for
//  byte, since the diagnostics lines must read as they did before.
//

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

@MainActor
struct LeafAdmissionTests {
    private let sessions = ToyModelSessionProvider(model: ToyLanguageModel(script: [1, 2]))

    nonisolated private static let labels = LeafAdmission.Labels(
        capture: "testCapture", admission: "testAdmission", source: "testLeaf")

    /// One attention layer of `rows` float32 rows, 64 wide: `rows * 512` bytes.
    nonisolated private static func cache(rows: Int = 10) -> [any KVCache] {
        let attention = KVCacheSimple()
        attention.state = [MLXArray.ones([1, 1, rows, 64]), MLXArray.ones([1, 1, rows, 64])]
        eval(attention.state)
        return [attention]
    }

    private struct Fixture {
        let key: CachePartitionKey
        let diagnostics: PrefixCacheDiagnostics.Context
        let manager: PrefixCacheManager
        let telemetry: TelemetryCapture
    }

    /// A model id of its own per test keeps each test's lines apart.
    private func makeFixture(_ name: String, kvBits: Int? = nil, budget: Int = 1 << 30) -> Fixture {
        let key = CachePartitionKey(
            modelID: "leaf-admission-\(name)", kvBits: kvBits, kvGroupSize: 64)
        return Fixture(
            key: key,
            diagnostics: .init(
                requestID: UUID(), modelID: key.modelID, kvBits: kvBits, kvGroupSize: 64),
            manager: PrefixCacheManager(memoryBudgetBytes: budget),
            telemetry: TelemetryCapture(modelID: key.modelID))
    }

    private func prepare(_ fixture: Fixture, tokens: [Int]) async -> LeafAdmission {
        await LeafAdmission.prepare(
            storedTokens: tokens, partitionKey: fixture.key, reachesSSD: false,
            requestID: fixture.diagnostics.requestID, prefixCache: fixture.manager,
            diagnostics: fixture.diagnostics)
    }

    // MARK: - Labels

    @Test func everyProducerKeepsItsStageNames() {
        #expect(
            LeafStorePhase.LeafStages.direct.admissionLabels
                == .init(capture: "leafCapture", admission: "leafAdmission", source: "leaf"))
        #expect(
            LeafAdmission.Labels.speculativeCanonicalPrefill(preempted: false)
                == .init(
                    capture: "speculativePrefill", admission: "speculativePrefill",
                    source: "speculativeLeaf"))
        #expect(
            LeafAdmission.Labels.speculativeCanonicalPrefill(preempted: true)
                == .init(
                    capture: "speculativePrefill", admission: "speculativePrefill",
                    source: "speculativePartialLeaf"))
        #expect(
            LeafAdmission.Labels.salvageOnCancel
                == .init(
                    capture: "salvageOnCancel", admission: "salvageOnCancel",
                    source: "cancelledPrefillSalvage"))
    }

    // MARK: - Whose cache

    @Test func anOwnedCacheMovesItsObjectsIntoTheTree() async throws {
        let fixture = makeFixture("owned")
        defer { fixture.telemetry.stop() }
        let tokens = Array(1...10)
        let admission = await prepare(fixture, tokens: tokens)
        let labels = Self.labels
        let (outcome, arrays) = await sessions.withSession { session in
            let owned = Self.cache()
            let arrays = owned.flatMap(\.state).map(ObjectIdentifier.init)
            return (await admission.admit(.owned(owned), in: session, labels: labels), arrays)
        }

        guard case .admitted(let admitted) = outcome else {
            Issue.record("expected an admitted leaf, got \(outcome)")
            return
        }
        #expect(admitted.capture.handedOff)
        #expect(admitted.capture.offset == tokens.count)
        #expect(admitted.survived)
        let leaf = try #require(
            fixture.manager.lookup(tokens: tokens, partitionKey: fixture.key).snapshot)
        #expect(leaf.layers.flatMap(\.state).map(ObjectIdentifier.init) == arrays)
        let capture = fixture.telemetry.drain().first { $0.eventName == "capture" }
        #expect(capture?.field("source") == "testLeaf")
    }

    @Test func aLentCacheIsCopiedAndStaysWithItsOwner() async throws {
        let fixture = makeFixture("lent")
        defer { fixture.telemetry.stop() }
        let tokens = Array(1...10)
        let admission = await prepare(fixture, tokens: tokens)
        let labels = Self.labels
        let (outcome, lentArrays, keptArrays) = await sessions.withSession { session in
            let lent = Self.cache()
            let before = lent.flatMap(\.state).map(ObjectIdentifier.init)
            let outcome = await admission.admit(.lent(lent), in: session, labels: labels)
            return (outcome, before, lent.flatMap(\.state).map(ObjectIdentifier.init))
        }

        guard case .admitted(let admitted) = outcome else {
            Issue.record("expected an admitted leaf, got \(outcome)")
            return
        }
        #expect(!admitted.capture.handedOff)
        #expect(keptArrays == lentArrays, "the lender keeps its arrays")
        let leaf = try #require(
            fixture.manager.lookup(tokens: tokens, partitionKey: fixture.key).snapshot)
        let stored = Set(leaf.layers.flatMap(\.state).map(ObjectIdentifier.init))
        #expect(stored.isDisjoint(with: lentArrays), "the tree holds a copy")
    }

    struct MoveCase: Sendable, CustomTestStringConvertible {
        let textOnly: Bool
        let kvBits: Int?
        let arm: SpeculativeArm?
        let moves: Bool
        var testDescription: String {
            "textOnly=\(textOnly) kvBits=\(kvBits.map(String.init) ?? "nil") "
                + "arm=\(arm?.rawValue ?? "nil") moves=\(moves)"
        }
    }

    nonisolated static let moveCases: [MoveCase] = [
        MoveCase(textOnly: true, kvBits: nil, arm: nil, moves: true),
        MoveCase(textOnly: true, kvBits: nil, arm: .dflash2, moves: true),
        MoveCase(textOnly: false, kvBits: nil, arm: nil, moves: false),
        MoveCase(textOnly: true, kvBits: 8, arm: nil, moves: false),
        MoveCase(textOnly: true, kvBits: nil, arm: .mtp, moves: false),
    ]

    @Test(arguments: moveCases)
    func aFinishedTurnMovesOnlyATextOnlyUnquantizedTurnWithoutMTP(_ move: MoveCase) async throws {
        let fixture = makeFixture("finished-\(move.moves)-\(move.textOnly)", kvBits: move.kvBits)
        defer { fixture.telemetry.stop() }
        let tokens = Array(1...10)
        let admission = await prepare(fixture, tokens: tokens)
        let (_, handOver) = await fixture.manager.resolveHoldingClaim(
            tokens: tokens + [99], partitionKey: fixture.key, diagnostics: fixture.diagnostics,
            sessions: sessions)
        let live = FinalGenerationCache(Self.cache())
        let sessions = sessions
        let labels = Self.labels
        let outcome = await handOver.withClaim { claim in
            await sessions.withSession { session in
                await admission.admit(
                    .finishedTurn(live, claim: claim, textOnly: move.textOnly, arm: move.arm),
                    in: session, labels: labels)
            }
        }

        guard case .admitted(let admitted) = outcome else {
            Issue.record("expected an admitted leaf, got \(outcome)")
            return
        }
        #expect(admitted.capture.handedOff == move.moves)
        #expect(live.cache.isEmpty == move.moves, "a move takes the objects from the turn")
        #expect(admitted.survived)
        #expect(fixture.manager.lookup(tokens: tokens, partitionKey: fixture.key).snapshot != nil)
    }

    @Test func anEmptyCacheIsNotCapturedAndNothingIsAdmitted() async {
        let fixture = makeFixture("empty")
        defer { fixture.telemetry.stop() }
        let admission = await prepare(fixture, tokens: Array(1...8))
        let labels = Self.labels
        let outcomes = await sessions.withSession { session in
            let lent = await admission.admit(.lent([]), in: session, labels: labels)
            let owned = await admission.admit(.owned([]), in: session, labels: labels)
            return [lent, owned]
        }

        for outcome in outcomes {
            guard case .notCaptured(let reason) = outcome else {
                Issue.record("expected no capture, got \(outcome)")
                continue
            }
            #expect(reason == "unsupported-cache-type")
        }
        let skips = fixture.telemetry.drain().filter { $0.eventName == "skip" }
        #expect(skips.count == 2)
        #expect(
            skips.allSatisfy {
                $0.field("stage") == "testCapture" && $0.field("reason") == "unsupported-cache-type"
            })
        #expect(fixture.manager.stats.snapshotCount == 0)
    }

    // MARK: - One classification per admission

    @Test func anAdmissionLogsWhatItSupersededOnceAndItsTallyMergesSilently() async throws {
        let fixture = makeFixture("supersede")
        defer { fixture.telemetry.stop() }
        let ancestorTokens = Array(1...8)
        let ancestor = await prepare(fixture, tokens: ancestorTokens)
        let extending = await prepare(fixture, tokens: ancestorTokens + [9, 10])
        let labels = Self.labels
        let outcome = await sessions.withSession { session in
            _ = await ancestor.admit(.lent(Self.cache(rows: 8)), in: session, labels: labels)
            return await extending.admit(.lent(Self.cache(rows: 10)), in: session, labels: labels)
        }

        guard case .admitted(let admitted) = outcome else {
            Issue.record("expected an admitted leaf, got \(outcome)")
            return
        }
        let tally = try #require(admitted.tally)
        #expect(tally.terminalEvictionCount == 0 && tally.recoveredEvictionCount == 0)
        let supersessions = fixture.telemetry.drain().filter {
            $0.eventName == "leafSupersession"
        }
        #expect(supersessions.count == 1)
        #expect(supersessions.first?.intField("offset") == ancestorTokens.count)

        var request = CompletionTraceAccumulator()
        request.merge(tally)
        #expect(fixture.telemetry.drain().isEmpty, "merging a tally logs nothing")
    }

    @Test func anAdmissionLogsItsEvictionsOnceAndHandsBackTheirTally() async throws {
        // Room for the newer twelve-row leaf alone: its admission evicts the
        // older ten-row leaf.
        let fixture = makeFixture("evict", budget: 12 * 512)
        defer { fixture.telemetry.stop() }
        let older = await prepare(fixture, tokens: Array(1...10))
        let newer = await prepare(fixture, tokens: Array(20...31))
        let labels = Self.labels
        let outcome = await sessions.withSession { session in
            _ = await older.admit(.lent(Self.cache(rows: 10)), in: session, labels: labels)
            return await newer.admit(.lent(Self.cache(rows: 12)), in: session, labels: labels)
        }

        guard case .admitted(let admitted) = outcome else {
            Issue.record("expected an admitted leaf, got \(outcome)")
            return
        }
        #expect(admitted.survived)
        let tally = try #require(admitted.tally)
        #expect(tally.terminalEvictionCount + tally.recoveredEvictionCount == 1)
        let evictions = fixture.telemetry.drain().filter { $0.eventName == "eviction" }
        #expect(evictions.count == 1, "one line per eviction, logged once")
        #expect(evictions.first?.intField("offset") == 10)

        var request = CompletionTraceAccumulator()
        request.merge(tally)
        #expect(
            request.terminalEvictionCount + request.recoveredEvictionCount == 1,
            "the request's record counts what the admission classified")
    }

    // MARK: - Whether the leaf survived (#578)

    /// The admission evicts another conversation's leaf of the same length.
    /// The new leaf is still in the tree, so it survived: no
    /// `capturedThenEvicted` warning, and the turn keeps its tuner record
    /// and speculative seed.
    @Test func evictingAnotherLeafOfTheSameLengthLeavesTheNewLeafStored() async throws {
        let fixture = makeFixture("same-length", budget: 10 * 512)
        defer { fixture.telemetry.stop() }
        let olderTokens = Array(1...10)
        let newerTokens = Array(20...29)
        let older = await prepare(fixture, tokens: olderTokens)
        let newer = await prepare(fixture, tokens: newerTokens)
        let labels = Self.labels
        let outcome = await sessions.withSession { session in
            _ = await older.admit(.lent(Self.cache(rows: 10)), in: session, labels: labels)
            return await newer.admit(.lent(Self.cache(rows: 10)), in: session, labels: labels)
        }

        guard case .admitted(let admitted) = outcome else {
            Issue.record("expected an admitted leaf, got \(outcome)")
            return
        }
        #expect(admitted.survived)
        #expect(
            fixture.manager.lookup(tokens: newerTokens, partitionKey: fixture.key).snapshot != nil)
        #expect(
            fixture.manager.lookup(tokens: olderTokens, partitionKey: fixture.key).snapshot == nil)
        let events = fixture.telemetry.drain()
        #expect(events.filter { $0.eventName == "eviction" }.map { $0.intField("offset") } == [10])
        #expect(!events.contains { $0.field("reason") == "capturedThenEvicted" })
    }

    /// A Cache Claim holds the leaf at this path by Leaf Handoff, so the tree
    /// refuses the new body (ADR-0064). Nothing was stored: the admission
    /// says so, and the claim's rewind puts the original leaf back.
    @Test func aLeafTheTreeRefusesIsNotReportedStored() async throws {
        let fixture = makeFixture("refused")
        defer { fixture.telemetry.stop() }
        let tokens = Array(1...16)
        let kv = KVCacheSimple()
        kv.state = [MLXArray.ones([1, 1, 16, 64]), MLXArray.ones([1, 1, 16, 64])]
        let recurrent = MambaCache()
        recurrent.state = [MLXArray.ones([4]), MLXArray.ones([4])]
        recurrent.offset = 16
        let owner = FinalGenerationCache([kv, recurrent])
        eval(owner.cache)
        let original = try #require(owner.moveSnapshot(offset: 16))
        let originalArrays = original.layers.flatMap(\.state).map(ObjectIdentifier.init)
        fixture.manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: tokens, snapshot: original, storage: .ramOnly,
                    partitionKey: fixture.key)))
        let requested = tokens + [7]
        let (checkout, handOver) = await fixture.manager.checkOutHoldingClaim(
            .init(
                lookup: fixture.manager.lookup(tokens: requested, partitionKey: fixture.key),
                hydratedFromSSD: false, hydrationSeconds: 0),
            tokens: requested, maximumAdvance: 2, diagnostics: fixture.diagnostics,
            sessions: sessions)
        guard case .handoff = checkout else {
            Issue.record("expected the claim to hold the leaf by handoff, got \(checkout)")
            return
        }

        let admission = await prepare(fixture, tokens: tokens)
        let labels = Self.labels
        let outcome = await sessions.withSession { session in
            await admission.admit(.lent(Self.cache(rows: 16)), in: session, labels: labels)
        }

        guard case .admitted(let admitted) = outcome else {
            Issue.record("expected an admission attempt, got \(outcome)")
            return
        }
        #expect(!admitted.survived)
        let events = fixture.telemetry.drain()
        #expect(
            events.contains {
                $0.eventName == "leafLeaseRefused" && $0.field("reason") == "bodyReplacement"
            })
        #expect(!events.contains { $0.field("reason") == "capturedThenEvicted" })

        // The rewind rebuilds the recurrent layer's state, so the original
        // shows in its attention arrays and its two layers (the refused body
        // had one).
        await handOver.rewindAndConclude(sessions: sessions)
        let stored = try #require(
            fixture.manager.lookup(tokens: tokens, partitionKey: fixture.key).snapshot)
        #expect(stored.layers.count == 2)
        #expect(
            stored.layers.first?.state.map(ObjectIdentifier.init)
                == Array(originalArrays.prefix(2)))
    }
}
