//
//  SnapshotState.swift
//  tesseract
//
//  The prefix-cache snapshot lifecycle as a single value-type deep
//  module. Each `RadixTreeNode` owns one `SnapshotState`, replacing the
//  old pair of raw fields (`snapshot` RAM body + `storageRef` SSD ref +
//  `committed` bool) that spread a five-state condition across the tree,
//  the tiered store, and the cache manager.
//
//  The load-bearing invariant — *never remove a node that still owns a
//  live SSD ref, or the persisted file is orphaned* — is answered here
//  by the `canEvictNode` query rather than re-derived at every call
//  site, so it becomes unrepresentable-when-violated instead of
//  guarded-by-comment.
//
//  **Posture: total core, strictness in the wrappers.** Every transition
//  below is total for every (state, transition) pair and never traps: an
//  inapplicable transition leaves the state unchanged and returns
//  `.ignored(reason)`. Strictness is *not* a partial function — it is a
//  `precondition` applied by the `TokenRadixTree` wrapper, layered over
//  this total core. The split is deliberate: it keeps the whole matrix
//  unit-testable with normal assertions (a `precondition` failure aborts
//  the process and cannot be a test assertion), while the two forgiving
//  SSD-writer callback edges (`commit`, `dropRef`) propagate `.ignored`
//  to preserve today's "log at debug, newer ref wins" semantics.
//
//  Naming note: `SnapshotState` is the prefix-cache lifecycle enum. It is
//  unrelated to `HybridCacheSnapshot.LayerState` (MLX layer tensors) and
//  to `@Observable` view/app state. See CONTEXT.md.
//

import Foundation
import MLXLMCommon

// MARK: - Transition outcomes

/// The topology-only outcome a `SnapshotState` transition reports to its
/// caller. Carries no telemetry payload — eviction telemetry rides on
/// `DropBodyResult`, because the common body drops (states 2/4) are
/// `.settled`, not `.becameEmpty`.
nonisolated enum StateEffect: Equatable {
    /// The state changed (or was re-set) and the node stays in the tree.
    case settled

    /// The resulting state is `empty`. This *triggers* the tree's
    /// self-heal, which then detaches the node only if topology allows
    /// (leaf → evict, single-child → collapse, multi-child empty node →
    /// retained as a structural junction). It is **not** itself "removed".
    case becameEmpty

    /// The transition was inapplicable to the current state. Returned by
    /// the total core for any such combination; the wrapper decides
    /// whether to propagate (forgiving edges) or `precondition` (strict
    /// edges).
    case ignored(IgnoreReason)
}

/// Why a transition was ignored. Preserved so an SSD-callback router can
/// keep today's diagnostic fidelity (`notPending` / `idMismatch` /
/// `alreadyCommitted`).
nonisolated enum IgnoreReason: Equatable {
    /// No pending write to act on (the state holds no pending ref).
    case notPending
    /// The current ref's ID does not match the expected ID — a later
    /// admission superseded it and the newer ref wins.
    case idMismatch
    /// The ref is already committed; a duplicate/late commit is a no-op.
    case alreadyCommitted
    /// The state has no resident body / ref the transition requires.
    case notResident
}

/// The richer result of `droppingBody`. Eviction telemetry
/// (`PrefixCacheManager.EvictionEvent`) is built on *every* body drop
/// from the dropped body's `checkpointType` + `memoryBytes` + the
/// surviving ref's ID, and the common state-2/4 drops are `.settled`, so
/// the ref ID cannot ride on `.becameEmpty`.
nonisolated struct DropBodyResult: Equatable {
    /// The dropped body's checkpoint type, or `nil` when no body was
    /// resident (the drop was `.ignored`).
    let droppedCheckpointType: HybridCacheSnapshot.CheckpointType?
    /// `memoryBytes` of the dropped body; `0` when no body was resident.
    let droppedBodyBytes: Int
    /// Present for state 2/4 (the ref survives the drop), `nil` for
    /// state 1 (`ramOnly` → `empty`) and for ignored drops.
    let refID: String?
    /// `.settled` (2→3, 4→5), `.becameEmpty` (1→empty), or
    /// `.ignored(.notResident)` (no body present).
    let effect: StateEffect
}

// MARK: - SnapshotState

/// The six-case lifecycle of a radix node's KV-cache snapshot. Owns both
/// the RAM body (`HybridCacheSnapshot`) and the on-disk identity
/// (`SnapshotRef`); transitions are pure functions returning the new
/// state plus a `StateEffect`.
nonisolated enum SnapshotState {
    /// State 0: no body, no ref. The only removable-and-collateral state.
    case empty
    /// State 1: RAM body, no ref.
    case ramOnly(HybridCacheSnapshot)
    /// State 2: RAM body + pending (uncommitted) ref.
    case pendingWrite(HybridCacheSnapshot, SnapshotRef)
    /// State 3: body dropped, pending ref still in flight.
    case pendingDropped(SnapshotRef)
    /// State 4: RAM body + committed ref.
    case committed(HybridCacheSnapshot, SnapshotRef)
    /// State 5: committed ref, no body — hydratable from SSD.
    case ssdOnly(SnapshotRef)

    // MARK: Reads

    /// The RAM-resident body, if any (states 1/2/4).
    var body: HybridCacheSnapshot? {
        switch self {
        case .ramOnly(let b), .pendingWrite(let b, _), .committed(let b, _): b
        case .empty, .pendingDropped, .ssdOnly: nil
        }
    }

    /// The on-disk ref, if any (states 2/3/4/5).
    var ref: SnapshotRef? {
        switch self {
        case .pendingWrite(_, let r), .pendingDropped(let r),
             .committed(_, let r), .ssdOnly(let r): r
        case .empty, .ramOnly: nil
        }
    }

    // MARK: Queries

    /// The orphan invariant: true iff removing the node structure cannot
    /// orphan an SSD-resident snapshot — i.e. the node holds no live ref.
    /// True for `empty`/`ramOnly` only.
    ///
    /// Necessary but *not* sufficient for node removal: actual structural
    /// removal additionally requires topology (the tree's self-heal).
    /// Distinct from `hasResidentBody`.
    var canEvictNode: Bool { ref == nil }

    /// RAM-budget concept: true iff a RAM body is resident (states
    /// 1/2/4). Explicitly distinct from `canEvictNode`.
    var hasResidentBody: Bool { body != nil }

    /// True iff a lookup can land here: a RAM body is present, or the ref
    /// is committed (hydratable). A `pendingDropped` (state 3) is never
    /// hittable — returning it would race the writer.
    var isHittable: Bool { body != nil || committed }

    /// True iff the ref has committed to disk (states 4/5).
    var committed: Bool {
        switch self {
        case .committed, .ssdOnly: true
        case .empty, .ramOnly, .pendingWrite, .pendingDropped: false
        }
    }

    /// The ref's snapshot ID, if any.
    var refID: String? { ref?.snapshotID }

    /// RAM-resident body bytes (feeds the **RAM** budget). `0` when no
    /// body — an `ssdOnly` node must never count against the RAM budget.
    var residentBodyBytes: Int { body?.memoryBytes ?? 0 }

    /// On-disk bytes of the ref (SSD-tier accounting only). `0` when no
    /// ref. Never conflated with `residentBodyBytes`.
    var storageBytes: Int { ref?.bytesOnDisk ?? 0 }

    /// Case name for logging / telemetry / test assertions. Does not
    /// expose payloads.
    var label: String {
        switch self {
        case .empty: "empty"
        case .ramOnly: "ramOnly"
        case .pendingWrite: "pendingWrite"
        case .pendingDropped: "pendingDropped"
        case .committed: "committed"
        case .ssdOnly: "ssdOnly"
        }
    }

    // MARK: - Transitions (total core)

    /// Install (or replace) the RAM body. Always applicable; always
    /// `.settled`. `empty`→`ramOnly`, `pendingDropped`→`pendingWrite`,
    /// `ssdOnly`→`committed`; the body-bearing states swap their body.
    func storingBody(_ newBody: HybridCacheSnapshot) -> (SnapshotState, StateEffect) {
        switch self {
        case .empty, .ramOnly:
            (.ramOnly(newBody), .settled)
        case .pendingWrite(_, let r), .pendingDropped(let r):
            (.pendingWrite(newBody, r), .settled)
        case .committed(_, let r), .ssdOnly(let r):
            (.committed(newBody, r), .settled)
        }
    }

    /// Attach a freshly enqueued pending ref. Applicable iff
    /// `hasResidentBody` (states 1/2/4) — re-admission over a still-pending
    /// or committed ref is legal and supersedes the old ref. The
    /// superseded ref's ID is surfaced so the router can delete its SSD
    /// backing (otherwise the old write orphans a file + manifest entry).
    func admitting(_ newRef: SnapshotRef)
        -> (state: SnapshotState, effect: StateEffect, supersededID: String?)
    {
        switch self {
        case .ramOnly(let b):
            (.pendingWrite(b, newRef), .settled, nil)
        case .pendingWrite(let b, let old):
            (.pendingWrite(b, newRef), .settled, old.snapshotID)
        case .committed(let b, let old):
            (.pendingWrite(b, newRef), .settled, old.snapshotID)
        case .empty, .pendingDropped, .ssdOnly:
            (self, .ignored(.notResident), nil)
        }
    }

    /// Commit a pending ref (SSD-writer callback, forgiving). `pendingWrite`
    /// → `committed`, `pendingDropped` → `ssdOnly`, gated on `expectedID`.
    func committing(expectedID: String) -> (SnapshotState, StateEffect) {
        switch self {
        case .pendingWrite(let b, let r):
            r.snapshotID == expectedID
                ? (.committed(b, r), .settled)
                : (self, .ignored(.idMismatch))
        case .pendingDropped(let r):
            r.snapshotID == expectedID
                ? (.ssdOnly(r), .settled)
                : (self, .ignored(.idMismatch))
        case .committed, .ssdOnly:
            (self, .ignored(.alreadyCommitted))
        case .empty, .ramOnly:
            (self, .ignored(.notPending))
        }
    }

    /// Drop a pending ref (SSD-writer callback, forgiving). `pendingWrite`
    /// → `ramOnly` (`.settled`, RAM body survives), `pendingDropped` →
    /// `empty` (`.becameEmpty`). Gated on `expectedID` so a superseded
    /// ref's late drop is ignored (newer ref wins).
    func droppingRef(expectedID: String) -> (SnapshotState, StateEffect) {
        switch self {
        case .pendingWrite(let b, let r):
            r.snapshotID == expectedID
                ? (.ramOnly(b), .settled)
                : (self, .ignored(.idMismatch))
        case .pendingDropped(let r):
            r.snapshotID == expectedID
                ? (.empty, .becameEmpty)
                : (self, .ignored(.idMismatch))
        case .empty, .ramOnly, .committed, .ssdOnly:
            (self, .ignored(.notPending))
        }
    }

    /// Drop the RAM body (RAM-budget eviction). `ramOnly` → `empty`
    /// (`.becameEmpty`), `pendingWrite` → `pendingDropped` (`.settled`,
    /// ref survives), `committed` → `ssdOnly` (`.settled`, ref survives).
    /// Body-less states return an `.ignored(.notResident)` result.
    func droppingBody() -> (SnapshotState, DropBodyResult) {
        switch self {
        case .ramOnly(let b):
            (.empty, DropBodyResult(
                droppedCheckpointType: b.checkpointType,
                droppedBodyBytes: b.memoryBytes,
                refID: nil,
                effect: .becameEmpty
            ))
        case .pendingWrite(let b, let r):
            (.pendingDropped(r), DropBodyResult(
                droppedCheckpointType: b.checkpointType,
                droppedBodyBytes: b.memoryBytes,
                refID: r.snapshotID,
                effect: .settled
            ))
        case .committed(let b, let r):
            (.ssdOnly(r), DropBodyResult(
                droppedCheckpointType: b.checkpointType,
                droppedBodyBytes: b.memoryBytes,
                refID: r.snapshotID,
                effect: .settled
            ))
        case .empty, .pendingDropped, .ssdOnly:
            (self, DropBodyResult(
                droppedCheckpointType: nil,
                droppedBodyBytes: 0,
                refID: nil,
                effect: .ignored(.notResident)
            ))
        }
    }

    /// Hydrate a committed ref with a freshly loaded body (`ssdOnly` →
    /// `committed`). Strict edge — only state 5 is hydratable.
    func hydrating(_ loadedBody: HybridCacheSnapshot) -> (SnapshotState, StateEffect) {
        switch self {
        case .ssdOnly(let r):
            (.committed(loadedBody, r), .settled)
        case .empty, .ramOnly, .pendingWrite, .pendingDropped, .committed:
            (self, .ignored(.notResident))
        }
    }

    /// Clear a committed ref after a hydration failure (file missing /
    /// fingerprint mismatch / decode error). `ssdOnly` → `empty`
    /// (`.becameEmpty`); a still-bodied committed/pending node keeps its
    /// body (→ `ramOnly`). No-ref states return `.ignored(.notResident)`.
    func clearingRef() -> (SnapshotState, StateEffect) {
        switch self {
        case .ssdOnly, .pendingDropped:
            (.empty, .becameEmpty)
        case .committed(let b, _), .pendingWrite(let b, _):
            (.ramOnly(b), .settled)
        case .empty, .ramOnly:
            (self, .ignored(.notResident))
        }
    }

    /// Warm-start restore: construct a committed-ref, body-less node
    /// directly (`empty` → `ssdOnly`). Strict edge — only an `empty`
    /// (freshly inserted) node is restorable.
    func restoringCommittedRef(_ restoredRef: SnapshotRef) -> (SnapshotState, StateEffect) {
        switch self {
        case .empty:
            (.ssdOnly(restoredRef), .settled)
        case .ramOnly, .pendingWrite, .pendingDropped, .committed, .ssdOnly:
            (self, .ignored(.notResident))
        }
    }
}
