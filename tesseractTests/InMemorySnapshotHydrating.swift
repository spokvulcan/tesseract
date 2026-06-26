//
//  InMemorySnapshotHydrating.swift
//  tesseractTests
//
//  The in-memory **Snapshot Hydrating** peer — the second adapter that makes
//  the consumer seam real (ADR-0001's "two adapters" rule). It runs the
//  off-main `loadSync` contract against programmed outcomes, so **Snapshot
//  Resolution**'s lookup-then-hydrate composition is assertable through
//  `PrefixCacheManager.resolve` with no loaded model, no temp directory, and
//  no concrete `SSDSnapshotStore`: no engines, no disk, no GPU.
//
//  It records the `loadSync` / `loadSyncPrefix` / `recordHit` call sequence so
//  the load-bearing ordering (recordHit + promote on success; neither on
//  failure) is pinned by the peer's observed trace. It deliberately does NOT
//  model `loadSync`'s on-disk file-deletion side effect — that role belongs to
//  the manager's real clear, exercised against the real tree.
//

import Foundation
import MLX
import MLXLMCommon
@testable import Tesseract_Agent

/// `@unchecked Sendable`: resolution drives `loadSync` off the MainActor then
/// hops to the MainActor for `recordHit`, and the test reads the trace only
/// after `await resolve` returns — a happens-before chain (the `MainActor.run`
/// barrier between them), so the recorded state is never accessed concurrently.
/// Plain `final class` (nonisolated by default) so its members satisfy the
/// `nonisolated` protocol contract and run on the caller's Metal-affine scope.
final class InMemorySnapshotHydrating: @unchecked Sendable, SnapshotHydrating {

    /// Programmed outcome map: a snapshot/owner id → the canned body a
    /// successful hydration returns. An id absent from the map fails (returns
    /// `nil`) — the explicit default, so a test opts each hittable id in.
    private var outcomes: [String: HybridCacheSnapshot] = [:]

    /// Every `loadSync` snapshot id, in call order.
    private(set) var loadSyncCalls: [String] = []
    /// Every `loadSyncPrefix` owner id, in call order.
    private(set) var loadSyncPrefixCalls: [String] = []
    /// Every `recordHit` id, in call order — the recency-bump trace.
    private(set) var recordHitCalls: [String] = []

    /// Program a successful hydration for `id` returning `body`. The body's
    /// `tokenOffset` should match the ref/point it satisfies.
    func programSuccess(id: String, body: HybridCacheSnapshot) {
        outcomes[id] = body
    }

    /// Program a failed hydration for `id` (returns `nil`, the analogue of a
    /// missing file / fingerprint mismatch / decode error). Idempotent: a
    /// success is overwritten so a test can flip an id mid-scenario.
    func programFailure(id: String) {
        outcomes.removeValue(forKey: id)
    }

    nonisolated func loadSync(
        snapshotRef: SnapshotRef,
        expectedFingerprint: String
    ) -> HybridCacheSnapshot? {
        let id = snapshotRef.snapshotID
        loadSyncCalls.append(id)
        return outcomes[id]
    }

    nonisolated func loadSyncPrefix(
        point: ChainPrefixRestorePoint,
        expectedFingerprint: String
    ) -> HybridCacheSnapshot? {
        let id = point.ownerSnapshotID
        loadSyncPrefixCalls.append(id)
        return outcomes[id]
    }

    nonisolated func recordHit(id: String) {
        recordHitCalls.append(id)
    }
}
