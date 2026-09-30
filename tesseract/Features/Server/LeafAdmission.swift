//
//  LeafAdmission.swift
//  tesseract
//
//  The **Leaf Admission** (ADR-0078): the one way a leaf enters the prefix
//  cache, whichever producer made it (a finished turn, a boundary re-prefill,
//  a Speculative Canonical Prefill, a salvaged cancelled prefill). The
//  producer brings its cache to the final state and says whose it is. The
//  admission does the rest, in the order ADR-0069 fixes:
//
//  1. `prepare`, before the producer enters its Model Session: an admission
//     that may reach SSD resolves its Leaf Extension Admission base on the
//     MainActor, so no capture waits on the MainActor for it.
//  2. `admit`, inside the producer's session: capture by move or by copy,
//     check a leased leaf in through its Cache Claim before anything is
//     extracted, build the Snapshot Payload and the Snapshot Admission, admit
//     it in one MainActor hop, report a leaf its own admission evicted, and
//     classify what the admission evicted and superseded, once.
//
//  Each producer's diagnostics lines keep the stage labels they carried
//  before the admission existed (`Labels`).
//

import Foundation
import MLX
import MLXLMCommon

nonisolated struct LeafAdmission: Sendable {

    /// Whose cache the producer hands over, which decides how it is captured.
    enum Cache {
        /// The finished turn's live cache. Its objects may be the ones the
        /// request's Cache Claim leased by Leaf Handoff. They move only for a
        /// text-only turn with unquantized KV whose MTP drafter did not run,
        /// and a moved leaf is checked in through `claim` before anything is
        /// extracted; otherwise the cache is copied.
        case finishedTurn(
            FinalGenerationCache, claim: CacheClaim, textOnly: Bool, arm: SpeculativeArm?)
        /// A cache the producer owns outright, such as a boundary re-prefill's
        /// restored cache: moved when every layer class can move, else copied.
        case owned([any KVCache])
        /// A cache the producer only lends: always copied.
        case lent([any KVCache])
    }

    /// The stage labels one producer's admission lines carry.
    struct Labels: Sendable, Equatable {
        /// The skip stage when the cache cannot be captured.
        let capture: String
        /// The skip stage when the admission is refused or evicts its own leaf.
        let admission: String
        /// The capture event's source.
        let source: String
    }

    /// What the **Active-Inference Reserve** observes of a leaf stored inside
    /// a turn (#522): the report path and restore mode its source is made
    /// from, once the capture says whether the leaf moved, and the turn's
    /// maximum advance. A leaf stored outside a turn passes none.
    struct Turn: Sendable {
        let path: LeafStorePhase.Report.Path
        let restoreMode: String
        let maximumAdvance: Int
    }

    /// One capture: the leaf's offset and bytes, whether it moved, and how
    /// long the capture took.
    struct Capture: Sendable {
        let offset: Int
        let bytes: Int
        let handedOff: Bool
        let seconds: TimeInterval
    }

    /// What one admission did.
    enum Outcome: Sendable {
        /// The cache could not be snapshotted (a layer class capture does not
        /// support). Nothing reached the tree.
        case notCaptured(reason: String)
        /// Captured by move, then taken back by the claim at check-in: the
        /// request was cancelled, or the tree refused the leaf. The original
        /// leaf is in the tree again, and nothing was extracted or admitted.
        case returned(CacheClaim.RewindCause, Capture)
        /// Captured and offered to the prefix cache.
        case admitted(Admitted)
    }

    struct Admitted: Sendable {
        let capture: Capture
        /// Whether the leaf is in the tree after its admission's own eviction
        /// pass; also false when the admission was refused before it touched
        /// the cache (an empty body, an invalid path).
        let survived: Bool
        /// The admission's eviction tally. Its lines are already logged, so a
        /// request merges it without logging again; `nil` when no admission
        /// was attempted.
        let tally: CompletionTraceAccumulator?
        let payloadSeconds: TimeInterval
        let admitSeconds: TimeInterval
    }

    let storedTokens: [Int]
    let partitionKey: CachePartitionKey
    let requestID: UUID
    let prefixCache: PrefixCacheManager
    let diagnostics: PrefixCacheDiagnostics.Context
    /// Whether this leaf may be written to SSD. A RAM-only admission resolves
    /// no extension base and builds no payload.
    let reachesSSD: Bool
    /// The **Leaf Extension Admission** base for `storedTokens`: the deepest
    /// SSD-backed ancestor leaf, when one exists.
    let extensionBase: SnapshotExtension?

    /// Prepare a leaf's admission before the producer enters its Model
    /// Session. One that may reach SSD resolves its extension base here, in
    /// one MainActor hop; a RAM-only one makes no hop and may be prepared
    /// anywhere, a session included.
    static func prepare(
        storedTokens: [Int],
        partitionKey: CachePartitionKey,
        reachesSSD: Bool,
        requestID: UUID,
        prefixCache: PrefixCacheManager,
        diagnostics: PrefixCacheDiagnostics.Context
    ) async -> LeafAdmission {
        var extensionBase: SnapshotExtension?
        if reachesSSD {
            assert(
                !ModelSessionScope.isInside,
                "prepare an SSD-bound leaf admission before entering the Model Session")
            extensionBase = await MainActor.run {
                prefixCache.extensionBase(tokens: storedTokens, partitionKey: partitionKey)
            }
        }
        return LeafAdmission(
            storedTokens: storedTokens, partitionKey: partitionKey, requestID: requestID,
            prefixCache: prefixCache, diagnostics: diagnostics, reachesSSD: reachesSSD,
            extensionBase: extensionBase)
    }

    /// Admit the leaf at `storedTokens.count`, inside the producer's Model
    /// Session. `turn` feeds the reserve for a leaf stored inside a turn;
    /// `memory` receives the request-memory phases between capture and
    /// admission.
    func admit(
        _ cache: Cache,
        in session: any ModelSession,
        labels: Labels,
        turn: Turn? = nil,
        memory: RequestMemoryTelemetry? = nil
    ) async -> Outcome {
        let moving: FinalGenerationCache?
        let copied: [any KVCache]
        switch cache {
        case .finishedTurn(let live, _, let textOnly, let arm):
            let moves = textOnly && partitionKey.kvBits == nil && arm != .mtp
            moving = moves ? live : nil
            copied = moves ? [] : live.cache
        case .owned(let owned):
            let moves = HybridCacheSnapshot.canCaptureMoving(cache: owned)
            moving = moves ? FinalGenerationCache(owned) : nil
            copied = moves ? [] : owned
        case .lent(let lent):
            moving = nil
            copied = lent
        }

        let captureStart = Date.timeIntervalSinceReferenceDate
        guard
            let leaf = moving != nil
                ? moving?.moveSnapshot(offset: storedTokens.count)
                : session.captureSnapshot(cache: copied, offset: storedTokens.count, type: .leaf)
        else {
            diagnostics.logSkip(stage: labels.capture, reason: "unsupported-cache-type")
            return .notCaptured(reason: "unsupported-cache-type")
        }
        let capture = Capture(
            offset: leaf.tokenOffset, bytes: leaf.memoryBytes, handedOff: moving != nil,
            seconds: Date.timeIntervalSinceReferenceDate - captureStart)
        memory?.mark(
            .preparingPayload,
            facts: [
                "leafSnapshotArrayBytes": "\(leaf.memoryBytes)",
                "leafCaptureMode": moving == nil ? "copy" : "handoff",
                "requestCacheLayerCountAfterCapture": "\(moving?.cache.count ?? copied.count)",
            ])

        // Check in before the payload is extracted (ADR-0069): the check-in
        // frees the recurrent rewind backup just before an extension payload
        // allocates arrays of the same shapes, so the check-in peak holds two
        // copies of the recurrent state, not three. A refused check-in skips
        // extraction entirely. Nothing can take the leaf in between:
        // check-outs are serialized by this Model Session, and only a live
        // generation writes arrays in place.
        if let moving, case .finishedTurn(_, let claim, _, _) = cache,
            case .rewound(let cause) = await claim.checkIn(
                leaf, from: moving, tokens: storedTokens, in: session)
        {
            // Not committed: the claim took the objects back and returned the
            // original leaf.
            return .returned(cause, capture)
        }

        let payloadStart = Date.timeIntervalSinceReferenceDate
        let storage = SnapshotAdmission.Storage.intent(
            for: leaf, ssdEnabled: reachesSSD, extending: extensionBase)
        let payloadSeconds = Date.timeIntervalSinceReferenceDate - payloadStart
        var payloadFacts = ["ssdPayloadMode": "none", "ssdPayloadArrayBytes": "0"]
        if case .ramAndSSD(let payload) = storage {
            payloadFacts = [
                "ssdPayloadMode": payload.extending == nil ? "full" : "extension",
                "ssdPayloadArrayBytes": "\(payload.totalBytes)",
            ]
        }
        memory?.mark(.admittingLeaf, facts: payloadFacts)
        memory?.mark(
            .admittingLeaf,
            facts: ["leafLeaseActive": "false", "recurrentRewindStateBytes": "0"])

        let admitStart = Date.timeIntervalSinceReferenceDate
        // The reserve observes the source the `leafStore` event will report
        // for this leaf: the same fold, from the same facts.
        let stored = await store(
            leaf,
            storage: storage,
            labels: labels,
            source: turn.flatMap {
                LeafStorePhase.Report.Source.stored(
                    path: $0.path, restoreMode: $0.restoreMode, handedOff: moving != nil)
            },
            maximumAdvance: turn?.maximumAdvance ?? .max)
        let admitSeconds = Date.timeIntervalSinceReferenceDate - admitStart

        var tally: CompletionTraceAccumulator?
        if let store = stored.store {
            // Classification and its correlated lines, once per admission,
            // through the accumulator that pairs the tally with the lines.
            var classified = CompletionTraceAccumulator()
            classified.ingest(evictions: store.evictions, diagnostics: diagnostics)
            classified.logSupersessions(store.supersededLeaves, diagnostics: diagnostics)
            tally = classified
        }
        return .admitted(
            Admitted(
                capture: capture, survived: stored.survived, tally: tally,
                payloadSeconds: payloadSeconds, admitSeconds: admitSeconds))
    }

    /// Build the leaf's Snapshot Admission, log the capture, admit it on the
    /// MainActor, and say whether it survived its own eviction pass. `store`
    /// is `nil` when the admission was refused before any cache mutation.
    private func store(
        _ leaf: HybridCacheSnapshot,
        storage: SnapshotAdmission.Storage,
        labels: Labels,
        source: LeafStorePhase.Report.Source?,
        maximumAdvance: Int
    ) async -> (survived: Bool, store: PrefixCacheManager.StoreDiagnostics?) {
        guard !leaf.layers.isEmpty else {
            diagnostics.logSkip(stage: labels.admission, reason: "empty-cache-body")
            return (false, nil)
        }
        guard
            let admission = SnapshotAdmission.leaf(
                storedTokens: storedTokens,
                snapshot: leaf,
                storage: storage,
                partitionKey: partitionKey,
                requestID: requestID,
                source: source,
                maximumAdvance: maximumAdvance
            )
        else {
            diagnostics.logSkip(
                stage: labels.admission,
                reason: "invalid-path",
                extraFields: [
                    ("offset", "\(leaf.tokenOffset)"),
                    ("storedLen", "\(storedTokens.count)"),
                ]
            )
            return (false, nil)
        }

        diagnostics.log(
            PrefixCacheDiagnostics.CaptureEvent(
                offset: leaf.tokenOffset,
                checkpointType: leaf.checkpointType,
                bytes: leaf.memoryBytes,
                duringPrefill: false,
                source: labels.source
            ))

        // Coalesce admit + stats read in one MainActor hop; the post-store
        // budget/total snapshot feeds the capturedThenEvicted diagnostic
        // without another hop.
        let prefixCache = self.prefixCache
        let (storeDiagnostics, postStoreBudgetBytes, postStoreSnapshotBytes) =
            await MainActor.run { () -> (PrefixCacheManager.StoreDiagnostics, Int, Int) in
                let d = prefixCache.admit(admission)
                return (d, prefixCache.memoryBudgetBytes, prefixCache.totalSnapshotBytes)
            }
        let admissionEvicted = storeDiagnostics.evictions.contains { event in
            event.offset == leaf.tokenOffset && event.checkpointType == .leaf
        }
        if admissionEvicted {
            diagnostics.logSkip(
                stage: labels.admission,
                reason: "capturedThenEvicted",
                level: .warning,
                extraFields: [
                    ("offset", "\(leaf.tokenOffset)"),
                    ("bytes", "\(leaf.memoryBytes)"),
                    ("budgetBytes", "\(postStoreBudgetBytes)"),
                    ("snapshotBytesAfter", "\(postStoreSnapshotBytes)"),
                ]
            )
            return (false, storeDiagnostics)
        }
        return (true, storeDiagnostics)
    }
}

// MARK: - The producers' labels

nonisolated extension LeafAdmission.Labels {
    /// The Speculative Canonical Prefill's leaf: a completed pass, or the
    /// partial leaf a preempted pass keeps.
    static func speculativeCanonicalPrefill(preempted: Bool) -> Self {
        Self(
            capture: "speculativePrefill", admission: "speculativePrefill",
            source: preempted ? "speculativePartialLeaf" : "speculativeLeaf")
    }

    /// Salvage-on-cancel's leaf, from a cancelled foreground prefill.
    static let salvageOnCancel = Self(
        capture: "salvageOnCancel", admission: "salvageOnCancel",
        source: "cancelledPrefillSalvage")
}
