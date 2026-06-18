//
//  SnapshotStateTests.swift
//  tesseractTests
//
//  The primary payoff of issue #10: the snapshot lifecycle transition
//  table and queries, exercised purely through the public `SnapshotState`
//  / `StateEffect` surface — no SSD writer, no MainActor mutation, no
//  reaching into node fields.
//
//  These tests target the **total transition core** (which never traps),
//  not the strict `TokenRadixTree` wrappers: the inapplicable
//  combinations that return `.ignored(reason)` are asserted as normal
//  outcomes precisely because the core is total. Strict-wrapper
//  `precondition` failures (which abort the process) are intentionally
//  not asserted here.
//

import Foundation
import Testing
import MLXLMCommon

@testable import Tesseract_Agent

@MainActor
struct SnapshotStateTests {

    // MARK: - Fixtures

    private func body(
        _ type: HybridCacheSnapshot.CheckpointType = .leaf,
        offset: Int = 4
    ) -> HybridCacheSnapshot {
        PrefixCacheTestFixtures.makeUniformSnapshot(offset: offset, type: type)
    }

    private func ref(
        id: String = UUID().uuidString,
        type: HybridCacheSnapshot.CheckpointType = .leaf
    ) -> SnapshotRef {
        SnapshotRef(
            snapshotID: id,
            partitionDigest: "deadbeef",
            tokenOffset: 4,
            checkpointType: type,
            bytesOnDisk: 1024
        )
    }

    /// One instance of every state, labelled, for matrix sweeps.
    private func allStates() -> [(label: String, state: SnapshotState)] {
        [
            ("empty", .empty),
            ("ramOnly", .ramOnly(body())),
            ("pendingWrite", .pendingWrite(body(), ref())),
            ("pendingDropped", .pendingDropped(ref())),
            ("committed", .committed(body(), ref())),
            ("ssdOnly", .ssdOnly(ref())),
        ]
    }

    // MARK: - Query truth-table

    @Test func queryTruthTable() {
        // (label, canEvictNode, hasResidentBody, isHittable, committed, hasBody, hasRef)
        // Test-local tuple; named struct not worth it (evolving MVP, see CLAUDE.md).
        // swiftlint:disable:next large_tuple
        let expected: [String: (Bool, Bool, Bool, Bool, Bool, Bool)] = [
            "empty": (true, false, false, false, false, false),
            "ramOnly": (true, true, true, false, true, false),
            "pendingWrite": (false, true, true, false, true, true),
            "pendingDropped": (false, false, false, false, false, true),
            "committed": (false, true, true, true, true, true),
            "ssdOnly": (false, false, true, true, false, true),
        ]
        for (label, state) in allStates() {
            let e = expected[label]!
            #expect(state.canEvictNode == e.0, "canEvictNode for \(label)")
            #expect(state.hasResidentBody == e.1, "hasResidentBody for \(label)")
            #expect(state.isHittable == e.2, "isHittable for \(label)")
            #expect(state.committed == e.3, "committed for \(label)")
            #expect((state.body != nil) == e.4, "body for \(label)")
            #expect((state.ref != nil) == e.5, "ref for \(label)")
            #expect(state.label == label)
        }
    }

    @Test func residentBodyBytesIgnoresSSDRef() {
        // ssdOnly must never count against the RAM budget.
        #expect(SnapshotState.ssdOnly(ref()).residentBodyBytes == 0)
        // storageBytes is the SSD-side count, present whenever a ref is.
        #expect(SnapshotState.ssdOnly(ref()).storageBytes == 1024)
        // ramOnly: body bytes count, no storage bytes.
        let ram = SnapshotState.ramOnly(body())
        #expect(ram.residentBodyBytes > 0)
        #expect(ram.storageBytes == 0)
        // committed: both present and distinct concepts.
        let committed = SnapshotState.committed(body(), ref())
        #expect(committed.residentBodyBytes > 0)
        #expect(committed.storageBytes == 1024)
    }

    // MARK: - storingBody (always settled)

    @Test func storingBodyMatrix() {
        let expected: [String: String] = [
            "empty": "ramOnly",
            "ramOnly": "ramOnly",
            "pendingWrite": "pendingWrite",
            "pendingDropped": "pendingWrite",
            "committed": "committed",
            "ssdOnly": "committed",
        ]
        for (label, state) in allStates() {
            let (next, effect) = state.storingBody(body(.system))
            #expect(next.label == expected[label], "storingBody from \(label)")
            #expect(effect == .settled, "storingBody effect from \(label)")
            #expect(next.body != nil)
            // Ref identity is preserved where one existed.
            #expect(next.ref?.snapshotID == state.ref?.snapshotID, "ref preserved from \(label)")
        }
    }

    // MARK: - admitting (precondition hasResidentBody; surfaces supersededID)

    @Test func admittingMatrix() {
        let newRef = ref(id: "NEW")
        // applicable iff hasResidentBody → always lands in pendingWrite
        let applicable: Set<String> = ["ramOnly", "pendingWrite", "committed"]
        for (label, state) in allStates() {
            let (next, effect, superseded) = state.admitting(newRef)
            if applicable.contains(label) {
                #expect(next.label == "pendingWrite", "admit from \(label)")
                #expect(effect == .settled, "admit effect from \(label)")
                #expect(next.refID == "NEW", "new ref installed from \(label)")
                // Superseded ID present iff a ref was replaced.
                let hadRef = state.ref != nil
                #expect((superseded != nil) == hadRef, "supersededID from \(label)")
                #expect(superseded == state.refID, "supersededID value from \(label)")
            } else {
                #expect(next.label == label, "admit ignored leaves \(label) unchanged")
                #expect(effect == .ignored(.notResident), "admit ignored reason from \(label)")
                #expect(superseded == nil)
            }
        }
    }

    // MARK: - committing (forgiving)

    @Test func committingMatrix() {
        let id = "ID-1"
        // Test-local tuple; named struct not worth it (evolving MVP, see CLAUDE.md).
        // swiftlint:disable:next large_tuple
        let cases: [(label: String, state: SnapshotState, result: String, effect: StateEffect)] = [
            ("empty", .empty, "empty", .ignored(.notPending)),
            ("ramOnly", .ramOnly(body()), "ramOnly", .ignored(.notPending)),
            ("pendingWrite", .pendingWrite(body(), ref(id: id)), "committed", .settled),
            ("pendingDropped", .pendingDropped(ref(id: id)), "ssdOnly", .settled),
            (
                "committed", .committed(body(), ref(id: id)), "committed",
                .ignored(.alreadyCommitted)
            ),
            ("ssdOnly", .ssdOnly(ref(id: id)), "ssdOnly", .ignored(.alreadyCommitted)),
        ]
        for c in cases {
            let (next, effect) = c.state.committing(expectedID: id)
            #expect(next.label == c.result, "commit from \(c.label)")
            #expect(effect == c.effect, "commit effect from \(c.label)")
        }
    }

    @Test func committingIDMismatchIsIgnored() {
        let (next, effect) =
            SnapshotState
            .pendingWrite(body(), ref(id: "REAL"))
            .committing(expectedID: "STALE")
        #expect(next.label == "pendingWrite")
        #expect(effect == .ignored(.idMismatch))
    }

    // MARK: - droppingRef (forgiving, id-gated)

    @Test func droppingRefMatrix() {
        let id = "ID-2"
        // Test-local tuple; named struct not worth it (evolving MVP, see CLAUDE.md).
        // swiftlint:disable:next large_tuple
        let cases: [(label: String, state: SnapshotState, result: String, effect: StateEffect)] = [
            ("empty", .empty, "empty", .ignored(.notPending)),
            ("ramOnly", .ramOnly(body()), "ramOnly", .ignored(.notPending)),
            ("pendingWrite", .pendingWrite(body(), ref(id: id)), "ramOnly", .settled),
            ("pendingDropped", .pendingDropped(ref(id: id)), "empty", .becameEmpty),
            ("committed", .committed(body(), ref(id: id)), "committed", .ignored(.notPending)),
            ("ssdOnly", .ssdOnly(ref(id: id)), "ssdOnly", .ignored(.notPending)),
        ]
        for c in cases {
            let (next, effect) = c.state.droppingRef(expectedID: id)
            #expect(next.label == c.result, "dropRef from \(c.label)")
            #expect(effect == c.effect, "dropRef effect from \(c.label)")
        }
    }

    @Test func droppingRefIDMismatchIsIgnored() {
        let (next, effect) =
            SnapshotState
            .pendingDropped(ref(id: "REAL"))
            .droppingRef(expectedID: "STALE")
        #expect(next.label == "pendingDropped")
        #expect(effect == .ignored(.idMismatch))
    }

    // MARK: - droppingBody (DropBodyResult)

    @Test func droppingBodyMatrix() {
        let refID = "ID-3"
        // ramOnly → empty, refID nil, becameEmpty
        do {
            let (next, result) = SnapshotState.ramOnly(body(.leaf)).droppingBody()
            #expect(next.label == "empty")
            #expect(result.effect == .becameEmpty)
            #expect(result.refID == nil)
            #expect(result.droppedCheckpointType == .leaf)
            #expect(result.droppedBodyBytes > 0)
        }
        // pendingWrite → pendingDropped, refID kept, settled
        do {
            let (next, result) =
                SnapshotState
                .pendingWrite(body(.branchPoint), ref(id: refID)).droppingBody()
            #expect(next.label == "pendingDropped")
            #expect(result.effect == .settled)
            #expect(result.refID == refID)
            #expect(result.droppedCheckpointType == .branchPoint)
        }
        // committed → ssdOnly, refID kept, settled
        do {
            let (next, result) =
                SnapshotState
                .committed(body(.system), ref(id: refID)).droppingBody()
            #expect(next.label == "ssdOnly")
            #expect(result.effect == .settled)
            #expect(result.refID == refID)
            #expect(result.droppedCheckpointType == .system)
        }
        // body-less states → ignored
        for state: SnapshotState in [.empty, .pendingDropped(ref()), .ssdOnly(ref())] {
            let (next, result) = state.droppingBody()
            #expect(next.label == state.label, "dropBody ignored leaves \(state.label) unchanged")
            #expect(result.effect == .ignored(.notResident))
            #expect(result.droppedCheckpointType == nil)
            #expect(result.droppedBodyBytes == 0)
            #expect(result.refID == nil)
        }
    }

    // MARK: - hydrating (strict edge)

    @Test func hydratingMatrix() {
        for (label, state) in allStates() {
            let (next, effect) = state.hydrating(body(.leaf))
            if label == "ssdOnly" {
                #expect(next.label == "committed", "hydrate from ssdOnly")
                #expect(effect == .settled)
                #expect(next.body != nil)
            } else {
                #expect(next.label == label, "hydrate ignored leaves \(label) unchanged")
                #expect(effect == .ignored(.notResident), "hydrate from \(label)")
            }
        }
    }

    // MARK: - clearingCommittedRefAfterBackingLoss

    @Test func clearingCommittedRefAfterBackingLossMatrix() {
        // Test-local tuple; named struct not worth it (evolving MVP, see CLAUDE.md).
        // swiftlint:disable:next large_tuple
        let cases: [(label: String, state: SnapshotState, result: String, effect: StateEffect)] = [
            ("empty", .empty, "empty", .ignored(.notResident)),
            ("ramOnly", .ramOnly(body()), "ramOnly", .ignored(.notResident)),
            ("pendingWrite", .pendingWrite(body(), ref()), "pendingWrite", .ignored(.notResident)),
            ("pendingDropped", .pendingDropped(ref()), "pendingDropped", .ignored(.notResident)),
            ("committed", .committed(body(), ref()), "ramOnly", .settled),
            ("ssdOnly", .ssdOnly(ref()), "empty", .becameEmpty),
        ]
        for c in cases {
            let (next, effect) = c.state.clearingCommittedRefAfterBackingLoss()
            #expect(next.label == c.result, "clear committed ref from \(c.label)")
            #expect(effect == c.effect, "clear committed ref effect from \(c.label)")
        }
    }

    // MARK: - discardingRefAfterExplicitDelete

    @Test func discardingRefAfterExplicitDeleteMatrix() {
        // Test-local tuple; named struct not worth it (evolving MVP, see CLAUDE.md).
        // swiftlint:disable:next large_tuple
        let cases: [(label: String, state: SnapshotState, result: String, effect: StateEffect)] = [
            ("empty", .empty, "empty", .ignored(.notResident)),
            ("ramOnly", .ramOnly(body()), "ramOnly", .ignored(.notResident)),
            ("pendingWrite", .pendingWrite(body(), ref()), "ramOnly", .settled),
            ("pendingDropped", .pendingDropped(ref()), "empty", .becameEmpty),
            ("committed", .committed(body(), ref()), "ramOnly", .settled),
            ("ssdOnly", .ssdOnly(ref()), "empty", .becameEmpty),
        ]
        for c in cases {
            let (next, effect) = c.state.discardingRefAfterExplicitDelete()
            #expect(next.label == c.result, "discard ref from \(c.label)")
            #expect(effect == c.effect, "discard ref effect from \(c.label)")
        }
    }

    // MARK: - restoringCommittedRef (warm start, strict edge)

    @Test func restoringCommittedRefMatrix() {
        for (label, state) in allStates() {
            let (next, effect) = state.restoringCommittedRef(ref(id: "RESTORED"))
            if label == "empty" {
                #expect(next.label == "ssdOnly", "restore from empty")
                #expect(effect == .settled)
                #expect(next.committed)
                #expect(next.refID == "RESTORED")
            } else {
                #expect(next.label == label, "restore ignored leaves \(label) unchanged")
                #expect(effect == .ignored(.notResident), "restore from \(label)")
            }
        }
    }

    // MARK: - State-layer deletion test: becameEmpty IFF resulting state is empty

    @Test func becameEmptyOccursIffResultingStateIsEmpty() {
        // The state-layer deletion test, in two directions:
        //  (1) `.becameEmpty` ⟹ the resulting state is `empty`.
        //  (2) a transition from a NON-empty state into `empty` ⟹ the
        //      effect was `.becameEmpty` (an already-empty input that
        //      stays empty via an `.ignored` no-op is excluded).
        let id = "ID-4"
        var observations: [(start: String, effect: StateEffect, end: String)] = []

        for (startLabel, state) in allStates() {
            func record(_ effect: StateEffect, _ next: SnapshotState) {
                observations.append((startLabel, effect, next.label))
            }
            let storing = state.storingBody(body()); record(storing.1, storing.0)
            let admit = state.admitting(ref(id: "X")); record(admit.effect, admit.state)
            let commit = state.committing(expectedID: id); record(commit.1, commit.0)
            let drop = state.droppingRef(expectedID: state.refID ?? id); record(drop.1, drop.0)
            let dropBody = state.droppingBody(); record(dropBody.1.effect, dropBody.0)
            let hydrate = state.hydrating(body()); record(hydrate.1, hydrate.0)
            let clear = state.clearingCommittedRefAfterBackingLoss(); record(clear.1, clear.0)
            let discard = state.discardingRefAfterExplicitDelete(); record(discard.1, discard.0)
            let restore = state.restoringCommittedRef(ref()); record(restore.1, restore.0)
        }

        for o in observations {
            if o.effect == .becameEmpty {
                #expect(o.end == "empty", "becameEmpty must resolve to empty, got \(o.end)")
            }
            if o.start != "empty" && o.end == "empty" {
                #expect(
                    o.effect == .becameEmpty,
                    "\(o.start) → empty must report becameEmpty, got \(o.effect)")
            }
        }
    }
}
