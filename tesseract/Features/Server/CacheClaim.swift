import Foundation
import MLX
import MLXLMCommon
import os

/// **Cache Claim** — one keyed request's whole hold on the prefix cache
/// (ADR-0069): its lane in the **Active-Inference Reserve**, its **Restore
/// Pins** and, after a **Leaf Handoff**, its **Leaf Lease**. Snapshot
/// Resolution adds the lane and the pins; the check-out takes the lease.
/// The claim concludes exactly once, inside the request's GPU lease, and
/// only a claim lets go of what it holds.
///
/// A claim is held only inside an owner scope, one owner at a time. `start`
/// owns a request's claim inside `withRequestClaim` and hands it to the
/// drive, which redeems the hand-over with `HandOver.withClaim`. A
/// **Speculative Canonical Prefill** pass owns a copy-only claim of its own
/// (`withCopyOnlyClaim`). Each scope concludes on exit, whether it returns
/// or throws: a leaf still leased is rewound back into the tree first, then
/// the pins and the lane go in one MainActor hop. The conclusion never
/// reads task cancellation (stream termination cancels the drive task on a
/// normal finish too) and runs in a detached task the scope awaits, so
/// cancellation cannot cut it short.
///
/// The steps — check-out, check-in, rewind — are ownership steps only, taken
/// at quiescent points inside the Model Session (their `session` argument is
/// the witness). Copy restore, suffix prefill and decode stay in plan
/// application and the drive (ADR-0033). Enter an owner scope outside the
/// Model Session: debug builds trap when a scope opens inside one.
///
/// What the types cannot prevent trips the tripwire: a claim dropped
/// without concluding (a hand-over never redeemed), a hand-over redeemed
/// twice, a second check-out, or a step taken while the claim is between
/// owners or after it concluded. Debug builds trap. Release builds log an
/// error and emit a `cacheClaimTripwire` event naming the request and what
/// it held. A claim dropped unconcluded, which nothing else will conclude,
/// also lets go of its pins and lane on the MainActor; any other violation
/// leaves that to the claim's own conclusion, after the leaf is back. A Leaf
/// Lease still held stays held (no exact return is possible without the
/// session), and the event names it.
nonisolated final class CacheClaim: @unchecked Sendable {

    // MARK: - Tokens

    /// The one value `PrefixCacheManager.release(_:)` accepts. Only a claim
    /// mints it, in its conclusion or its tripwire.
    struct Release: Sendable {
        let requestID: UUID
        fileprivate init(requestID: UUID) { self.requestID = requestID }
    }

    /// `start` hands its claim to the drive with this, once. The statement
    /// that receives it also creates the drive task, and the drive redeems
    /// it around its whole body.
    struct HandOver: Sendable {
        fileprivate let claim: CacheClaim

        /// Own the claim for `body`, then conclude it. A second redemption
        /// trips the tripwire and concludes nothing.
        func withClaim<R>(_ body: (CacheClaim) async throws -> R) async rethrows -> R {
            // This conclusion is the one that may enter the session again.
            assert(!ModelSessionScope.isInside, "redeem a Cache Claim outside the Model Session")
            let redeemed = claim.redeem()
            defer { if redeemed { await claim.conclude() } }
            return try await body(claim)
        }
    }

    // MARK: - Outcomes

    /// How the request restores, decided by the check-out.
    enum RestoreOutcome: Sendable {
        /// Nothing to check out: a miss, or a resolved snapshot without a
        /// partition, which plan application restores by copy if it can. A
        /// check-out the tripwire refused answers this too.
        case cold
        /// Restore by copy, with why the leaf was not taken.
        case copy(Copy)
        /// The claim holds the Leaf Lease, and the request generates into
        /// the leaf's own cache objects.
        case handoff(Handoff)
    }

    struct Copy: Sendable, Equatable {
        /// The precise rung that refused the leaf (wire `copyRefusal`).
        let refusal: CopyRefusal
        /// Seconds the Pending-Payload Wait cost before settling (#523);
        /// `0` when it never waited.
        let waitSeconds: TimeInterval

        /// The coarser reason reported as `copyReason`, byte for byte what
        /// it was before `copyRefusal` existed.
        var reason: LeafStorePhase.Report.CopyReason { refusal.copyReason }
    }

    struct Handoff: Sendable {
        /// The leaf's own cache objects, now the request's live cache.
        let cache: FinalGenerationCache
        let leaseID: UUID
        /// The recurrent backup exact rewind keeps: the claim's only array
        /// allocation.
        let rewindStateBytes: Int
        /// Seconds a Pending-Payload Wait cost before it ended in this
        /// handoff; `0` when it never waited.
        let waitSeconds: TimeInterval
    }

    /// Why a copy restore did not take the leaf: the precise rung of the
    /// check-out that refused, reported as `copyRefusal` beside `copyReason`.
    /// Several refusals share one `copyReason` — every lease refusal reads
    /// `pendingFullPayload`, the path, offset and moved-body checks read
    /// `checkpoint` — so this is what tells them apart.
    enum CopyRefusal: String, Sendable, Equatable {
        // The guard ladder, before the tree is asked.
        case prefixView, warmBody, imageKeySpace, quantizedPartition, chainPrefixRestore
        case offsetMismatch, nothingToExtend
        // The manager's checks.
        case checkoutDisabled, notResidentLeaf, compressing, bodyReplaced
        // The body's own answer (`HybridCacheSnapshot.checkoutRefusal`).
        case notLeafCheckpoint, immutableBody, emptyBody, quantizedLayer, untrimmable, rotating
        // The tree's lease refusals.
        case wrongTree, notLeaf, alreadyLeased, pendingFullPayload, writerReading

        /// A lease refusal the tree answered with.
        init(leaseRefusal: LeafLeaseRefusedEvent.Reason) {
            switch leaseRefusal {
            case .wrongTree: self = .wrongTree
            case .notLeaf: self = .notLeaf
            case .alreadyLeased: self = .alreadyLeased
            case .writerReading: self = .writerReading
            default: self = .pendingFullPayload
            }
        }

        var copyReason: LeafStorePhase.Report.CopyReason {
            switch self {
            case .prefixView, .chainPrefixRestore, .offsetMismatch, .nothingToExtend,
                .notResidentLeaf, .bodyReplaced, .notLeafCheckpoint, .emptyBody:
                .checkpoint
            case .warmBody: .warmBody
            case .imageKeySpace: .imageKeySpace
            case .quantizedPartition, .quantizedLayer: .quantized
            case .checkoutDisabled: .checkoutDisabled
            case .compressing, .immutableBody: .immutableBody
            case .untrimmable: .untrimmable
            case .rotating: .rotating
            case .wrongTree, .notLeaf, .alreadyLeased, .pendingFullPayload, .writerReading:
                .pendingFullPayload
            }
        }

        /// A refusal the tree gave for the lease itself — the refusals the
        /// Pending-Payload Wait may outlast.
        fileprivate var isLeaseRefusal: Bool { copyReason == .pendingFullPayload }
    }

    enum CheckIn: Sendable, Equatable {
        /// The claim holds no lease on this cache: nothing to check in.
        case notLeased
        /// The tree committed the leaf and the lease ended.
        case committed
        /// The leaf was not committed. The claim took its objects back and
        /// rewound them, so the original leaf is in the tree again.
        case rewound(RewindCause)
    }

    enum RewindCause: Sendable, Equatable {
        /// The request was cancelled before its leaf was committed: the old
        /// leaf stays for the resend.
        case cancelled
        /// The tree refused the check-in.
        case refused(LeafLeaseRefusedEvent.Reason)
    }

    /// What a debug build traps on and a release build reports.
    enum Violation: String, Sendable {
        case droppedUnconcluded, handOverRedeemedTwice, concludedTwice, secondCheckOut
        case stepBetweenOwners, stepAfterConclusion
    }

    /// Whether a violation traps. `standard` traps in debug builds and
    /// reports in release builds; `reporting` only reports, in every build
    /// — what a release build does, for tests of that behaviour.
    struct Tripwire: Sendable {
        let traps: Bool

        static let standard = Tripwire(traps: true)
        static let reporting = Tripwire(traps: false)
    }

    // MARK: - State

    private enum Owner: String {
        case start, inTransit, drive, copyOnly, concluded
    }

    let requestID: UUID
    private let context: PrefixCacheDiagnostics.Context
    private let prefixCache: PrefixCacheManager
    /// Where the conclusion rewinds a lease still held; `nil` for a
    /// copy-only claim, which never holds one.
    private let sessions: (any ModelSessionProviding)?
    private let memory: RequestMemoryTelemetry?
    private let tripwire: Tripwire
    private let lock = NSLock()
    private var owner: Owner
    private var held: CheckedOutLeaf?
    private var checkedOut = false

    private init(
        requestID: UUID, owner: Owner, context: PrefixCacheDiagnostics.Context,
        prefixCache: PrefixCacheManager, sessions: (any ModelSessionProviding)?,
        memory: RequestMemoryTelemetry?, tripwire: Tripwire
    ) {
        self.requestID = requestID
        self.owner = owner
        self.context = context
        self.prefixCache = prefixCache
        self.sessions = sessions
        self.memory = memory
        self.tripwire = tripwire
    }

    deinit {
        let (owner, lease) = lock.withLock { (self.owner, held?.grant.lease) }
        guard owner != .concluded else { return }
        Self.trip(
            .droppedUnconcluded, owner: owner, lease: lease, releasing: true,
            requestID: requestID, context: context, prefixCache: prefixCache,
            tripwire: tripwire)
    }

    // MARK: - Owner scopes

    /// Hold a request's claim for `body`, the request's `start`. A normal
    /// return hands the claim over; a throw concludes it here. The claim's
    /// request is `context.requestID`; `sessions` is where its conclusion
    /// rewinds a lease still held.
    static func withRequestClaim<R>(
        context: PrefixCacheDiagnostics.Context,
        prefixCache: PrefixCacheManager,
        sessions: any ModelSessionProviding,
        memory: RequestMemoryTelemetry?,
        tripwire: Tripwire = .standard,
        _ body: (CacheClaim) async throws -> R
    ) async rethrows -> (value: R, handOver: HandOver) {
        assert(!ModelSessionScope.isInside, "open a Cache Claim outside the Model Session")
        let claim = CacheClaim(
            requestID: context.requestID, owner: .start, context: context,
            prefixCache: prefixCache, sessions: sessions, memory: memory, tripwire: tripwire)
        var handedOver = false
        defer { if !handedOver { await claim.conclude() } }
        let value = try await body(claim)
        claim.handOver()
        handedOver = true
        return (value, HandOver(claim: claim))
    }

    /// Hold a Speculative Canonical Prefill pass's claim for `body`, then
    /// conclude it. It pins what the pass restores from and holds a lane;
    /// it has no step that could take a leaf.
    static func withCopyOnlyClaim<R>(
        context: PrefixCacheDiagnostics.Context,
        prefixCache: PrefixCacheManager,
        tripwire: Tripwire = .standard,
        _ body: (CopyOnlyClaim) async throws -> R
    ) async rethrows -> R {
        assert(!ModelSessionScope.isInside, "open a Cache Claim outside the Model Session")
        let claim = CacheClaim(
            requestID: UUID(), owner: .copyOnly, context: context, prefixCache: prefixCache,
            sessions: nil, memory: nil, tripwire: tripwire)
        defer { await claim.conclude() }
        return try await body(CopyOnlyClaim(claim: claim))
    }

    // MARK: - Steps

    /// The request Snapshot Resolution adds this claim's lane and pins
    /// under, or `nil` when the claim is between owners or has concluded
    /// (the tripwire fires and resolution pins nothing).
    func resolutionEntry() -> UUID? {
        ownsStep() ? requestID : nil
    }

    /// Decide handoff or copy for the resolved snapshot, in plan
    /// application's restore arm, before any model verb touches the cache.
    /// Never fails: the worst outcome is a copy. On a handoff the claim
    /// holds the lease until check-in or rewind.
    func checkOut(
        _ resolved: PrefixCacheManager.Resolved, tokens: [Int], maximumAdvance: Int,
        identityKeySpace: Bool, in session: any ModelSession
    ) async -> RestoreOutcome {
        guard ownsStep() else { return .cold }
        let second = lock.withLock {
            defer { checkedOut = true }
            return checkedOut
        }
        guard !second else {
            trip(.secondCheckOut)
            return .cold
        }
        guard let snapshot = resolved.lookup.snapshot,
            let key = resolved.lookup.partitionKey
        else { return .cold }
        func copy(_ refusal: CopyRefusal, waited: TimeInterval = 0) -> RestoreOutcome {
            .copy(Copy(refusal: refusal, waitSeconds: waited))
        }
        guard !snapshot.isPrefixView else { return copy(.prefixView) }
        guard !snapshot.isWarm else { return copy(.warmBody) }
        guard identityKeySpace else { return copy(.imageKeySpace) }
        guard key.kvBits == nil else { return copy(.quantizedPartition) }
        guard !resolved.wasChainPrefixRestore else { return copy(.chainPrefixRestore) }
        guard resolved.lookup.snapshotTokenOffset == snapshot.tokenOffset else {
            return copy(.offsetMismatch)
        }
        guard tokens.count > snapshot.tokenOffset else { return copy(.nothingToExtend) }
        let bodyRefusal = snapshot.checkoutRefusal(maximumAdvance: maximumAdvance)
        var waited: TimeInterval = 0
        var leased = await prefixCache.leaseLeaf(
            snapshot: snapshot, tokens: tokens, partitionKey: key, bodyRefusal: bodyRefusal,
            context: context)
        // The one refusal that clears itself: a full payload aliases the
        // body until the SSD writer finishes using its borrowed arrays
        // (ADR-0064 decision 5, ADR-0019's Deferred Payload Extraction
        // amendment). Every lease refusal takes this path, as it did when
        // they all read `pendingFullPayload`.
        if case .refused(let refusal) = leased, refusal.isLeaseRefusal {
            (leased, waited) = await awaitPendingFullPayload(
                refusal, snapshot: snapshot, tokens: tokens, partitionKey: key,
                bodyRefusal: bodyRefusal)
        }
        switch leased {
        case .refused(let refusal):
            return copy(refusal, waited: waited)
        case .leased(let grant):
            let taken = CheckedOutLeaf.take(snapshot, under: grant)
            lock.withLock { held = taken }
            return .handoff(
                Handoff(
                    cache: taken.live, leaseID: grant.lease.id,
                    rewindStateBytes: taken.rewind.stateBytes, waitSeconds: waited))
        }
    }

    /// Check the leaf in, at a quiescent point after capture: the tree
    /// commits `leaf` under `tokens` and the lease ends. `leaf` is `live`'s
    /// objects, captured by move. A request already cancelled, or a leaf the
    /// tree refuses, is not committed: the claim takes the objects back and
    /// rewinds them in this same step, so a cancelled turn keeps the old
    /// leaf for the resend. Either way the lease ends.
    func checkIn(
        _ leaf: HybridCacheSnapshot, from live: FinalGenerationCache, tokens: [Int],
        in session: any ModelSession
    ) async -> CheckIn {
        guard ownsStep(), let held = lock.withLock({ self.held }), held.live === live else {
            return .notLeased
        }
        if Task.isCancelled {
            live.recoverUnadmitted(leaf)
            await rewindHeld()
            return .rewound(.cancelled)
        }
        if let refusal = await MainActor.run(body: { held.grant.checkIn(leaf, tokens: tokens) }) {
            live.recoverUnadmitted(leaf)
            await rewindHeld()
            return .rewound(.refused(refusal))
        }
        // Frees the recurrent backup.
        lock.withLock { self.held = nil }
        return .committed
    }

    /// **Leaf Rewind** as an explicit step: return the leased leaf,
    /// trimmed and rebuilt, to its node. The rewound leaf's offset, or
    /// `nil` when nothing is leased.
    @discardableResult
    func rewind(in session: any ModelSession) async -> Int? {
        guard ownsStep() else { return nil }
        return await rewindHeld()
    }

    /// Whether the claim still holds the lease its check-out took.
    var holdsLease: Bool { lock.withLock { held != nil } }

    /// The turn's maximum advance: the prompt tokens prefilled past the
    /// restore offset plus the output ceiling plus the speculative
    /// allowance, `Int.max` when the output is unbounded. What check-out
    /// eligibility is judged against, and what the **Active-Inference
    /// Reserve** prices the turn's growth at (#522).
    static func maximumAdvance(
        newPromptTokens: Int, outputCeiling: Int?, speculativeAllowance: Int
    ) -> Int {
        guard let outputCeiling else { return Int.max }
        let (withPrompt, overflow) = max(0, outputCeiling).addingReportingOverflow(
            max(0, newPromptTokens))
        let (total, draftOverflow) = withPrompt.addingReportingOverflow(
            max(0, speculativeAllowance))
        return overflow || draftOverflow ? Int.max : total
    }

    // MARK: - The Pending-Payload Wait

    /// How often the wait re-reads the writer's answer. Small enough that
    /// a materialization is turned into a handoff promptly, large enough
    /// that the longest bound costs a bounded number of MainActor hops.
    static let pendingFullPayloadPoll: Duration = .milliseconds(5)

    /// The bounded wait of #523. The lease was refused only because the
    /// leaf's full payload is still pending; while the SSD writer reports
    /// that payload `.inProgress`, the request waits up to the **Eviction
    /// Configuration** bound and then asks for the lease once more. A
    /// payload still `.queued` behind other writes has no bounded
    /// completion time and is not waited for — it copies at once.
    ///
    /// The wait is an `await`, never a blocking sleep: no Metal work runs
    /// and no thread is held, and it happens before any Model Session verb
    /// touches the cache. Cancellation settles it as a copy at once.
    /// `refusal` is the tree's answer to the first attempt; a copy that
    /// settles without asking the tree again keeps it.
    private func awaitPendingFullPayload(
        _ refusal: CopyRefusal, snapshot: HybridCacheSnapshot, tokens: [Int],
        partitionKey: CachePartitionKey, bodyRefusal: CopyRefusal?
    ) async -> (outcome: PrefixCacheManager.LeafLeaseOutcome, waitedSeconds: TimeInterval) {
        let prefixCache = prefixCache
        let context = context
        func progress() async -> PendingPayloadProgress {
            await prefixCache.pendingFullPayloadProgress(
                snapshot: snapshot, tokens: tokens, partitionKey: partitionKey)
        }
        func reattempt() async -> PrefixCacheManager.LeafLeaseOutcome {
            await prefixCache.leaseLeaf(
                snapshot: snapshot, tokens: tokens, partitionKey: partitionKey,
                bodyRefusal: bodyRefusal, context: context)
        }
        let bound = await prefixCache.pendingFullPayloadWait
        guard bound > .zero else { return (.refused(refusal), 0) }
        switch await progress() {
        case .absent:
            // The writer let go between the refusal and this read, so the
            // refusal is already stale. Nothing to wait for — but nothing
            // to copy for either, so ask again and take the leaf.
            return (await reattempt(), 0)
        case .queued:
            // Queued behind other writes, with no bounded completion time
            // of its own. This request copies now, exactly as before #523.
            return (.refused(refusal), 0)
        case .inProgress:
            break
        }
        let started = ContinuousClock.now
        var elapsed = Duration.zero
        while elapsed < bound {
            do {
                try await Task.sleep(for: min(Self.pendingFullPayloadPoll, bound - elapsed))
            } catch {
                break  // cancelled: settle as a copy rather than hold the request
            }
            elapsed = ContinuousClock.now - started
            if await progress() != .inProgress { break }
        }
        let waitedSeconds = elapsed.seconds
        guard !Task.isCancelled else { return (.refused(refusal), waitedSeconds) }
        return (await reattempt(), waitedSeconds)
    }

    // MARK: - Leaf Rewind

    /// Trim, rebuild and compact the live cache, move it back into the tree
    /// under the leased node, and end the lease. `nil` when nothing is
    /// leased. Inside the Model Session, after generation has quiesced.
    @discardableResult
    private func rewindHeld() async -> Int? {
        guard
            let held = lock.withLock({ () -> CheckedOutLeaf? in
                defer { self.held = nil }
                return self.held
            })
        else { return nil }
        let lease = held.grant.lease
        memory?.mark(
            .rewindingLeaf, facts: ["recurrentRewindStateBytes": "\(held.rewind.stateBytes)"])
        let (compaction, rewoundFacts) = await held.returnByRewind()
        lease.context.log(
            LeafRewindEvent(
                lease: lease, recurrentBytes: held.rewind.stateBytes,
                fullAttentionArrayBytes: Int(
                    rewoundFacts["requestFullAttentionArrayBytes"] ?? "") ?? 0,
                fullAttentionLogicalBytes: Int(
                    rewoundFacts["requestFullAttentionLogicalBytes"] ?? "") ?? 0,
                compactedBytes: compaction.freedBytes),
            level: .notice)
        var report = LeafStorePhase.Report()
        report.mode = "keyed"
        report.path = .rewind
        report.restoreMode = "handoff"
        report.leafOffset = lease.offset
        lease.context.log(report, level: .notice)
        memory?.markCacheReleased(
            .rewoundLeaf,
            facts: [
                "leafSource": "rewind", "recurrentRewindStateBytes": "0",
                "leafLeaseActive": "false",
                "rewoundLeafFullAttentionArrayBytes":
                    rewoundFacts["requestFullAttentionArrayBytes"] ?? "0",
                "rewoundLeafFullAttentionLogicalBytes":
                    rewoundFacts["requestFullAttentionLogicalBytes"] ?? "0",
                "rewoundLeafFullAttentionUnusedArrayBytes":
                    rewoundFacts["requestFullAttentionUnusedArrayBytes"] ?? "0",
                "rewoundLeafCompactedBytes": "\(compaction.freedBytes)",
            ])
        return lease.offset
    }

    // MARK: - Ownership

    /// Whether the current owner may take a step now. A claim between
    /// owners or already concluded trips the tripwire instead.
    private func ownsStep() -> Bool {
        let owner = lock.withLock { self.owner }
        switch owner {
        case .start, .drive, .copyOnly:
            return true
        case .inTransit:
            trip(.stepBetweenOwners, owner: owner)
            return false
        case .concluded:
            trip(.stepAfterConclusion, owner: owner)
            return false
        }
    }

    private func handOver() {
        lock.withLock {
            precondition(owner == .start, "only start hands a claim over")
            owner = .inTransit
        }
    }

    private func redeem() -> Bool {
        let previous = lock.withLock { () -> Owner in
            let previous = owner
            if previous == .inTransit { owner = .drive }
            return previous
        }
        guard previous == .inTransit else {
            trip(.handOverRedeemedTwice, owner: previous)
            return false
        }
        return true
    }

    /// Exactly once, by an owner scope: the leaf comes back, then the pins
    /// go, then the lane. A lease still held is rewound in the Model Session
    /// — the one reason the conclusion enters it, and entering cannot fail
    /// (ADR-0016) — and the MLX buffer cache is cleared after it, so the
    /// rewind's compaction does not leave its freed buffers there.
    private func conclude() async {
        let previous = lock.withLock { () -> Owner in
            let previous = owner
            owner = .concluded
            return previous
        }
        guard previous != .concluded else {
            trip(.concludedTwice, owner: previous)
            return
        }
        let token = Release(requestID: requestID)
        await Task.detached { [self] in
            if holdsLease, let sessions {
                await sessions.withSession { _ in
                    await self.rewindHeld()
                    Memory.clearCache()
                }
            }
            memory?.mark(.releasingRequest)
            let facts = await MainActor.run {
                prefixCache.release(token)
                return prefixCache.memoryTelemetryFacts()
            }
            memory?.mark(.releasingRequest, facts: facts)
        }.value
    }

    // MARK: - Tripwire

    /// A violation by a claim that is still referenced: its owner scope
    /// still concludes it, returning the leaf before the pins and the lane,
    /// so the tripwire reports and lets go of nothing.
    private func trip(_ violation: Violation, owner: Owner? = nil) {
        let (current, lease) = lock.withLock { (self.owner, held?.grant.lease) }
        Self.trip(
            violation, owner: owner ?? current, lease: lease, releasing: false,
            requestID: requestID, context: context, prefixCache: prefixCache,
            tripwire: tripwire)
    }

    /// `releasing` only for a claim dropped unconcluded, which nothing else
    /// will ever conclude: its pins and lane go here, and a lease it still
    /// holds stays held, since only the Model Session can return it exactly.
    private static func trip(
        _ violation: Violation, owner: Owner, lease: LeafLease?, releasing: Bool,
        requestID: UUID, context: PrefixCacheDiagnostics.Context,
        prefixCache: PrefixCacheManager, tripwire: Tripwire
    ) {
        let message =
            "Cache Claim tripwire: \(violation.rawValue), request \(requestID.uuidString) "
            + "(\(owner.rawValue))" + (releasing ? "; letting go of its pins and lane" : "")
            + (lease.map { ", keeping its lease \($0.id.uuidString)" } ?? "") + " (ADR-0069)"
        Log.server.error("\(message)")
        let token = Release(requestID: requestID)
        Task { @MainActor in
            let released = releasing ? prefixCache.release(token) : (lane: false, pins: 0)
            context.log(
                CacheClaimTripwireEvent(
                    violation: violation, owner: owner.rawValue, claimRequestID: requestID,
                    releasedLane: released.lane, releasedPins: released.pins, heldLease: lease),
                level: .error)
        }
        if tripwire.traps { assertionFailure(message) }
    }
}

/// A Speculative Canonical Prefill pass's claim: Snapshot Resolution pins
/// and adds a lane under it, and it has no step that could take a leaf.
nonisolated struct CopyOnlyClaim: Sendable {
    fileprivate let claim: CacheClaim

    /// See `CacheClaim.resolutionEntry()`.
    func resolutionEntry() -> UUID? { claim.resolutionEntry() }
}

/// A leaf taken by **Leaf Handoff** under a Leaf Lease: the tree's grant,
/// the leaf's own cache objects as the live cache, and the recurrent backup
/// exact rewind needs. References only; the token path stays in the tree.
/// The Cache Claim holds one per handoff. The bounded-cache parity bench
/// takes and rewinds a leaf through this same code, outside any claim.
nonisolated struct CheckedOutLeaf: @unchecked Sendable {
    let grant: PrefixCacheManager.LeafLeaseGrant
    let live: FinalGenerationCache
    let rewind: LeafRewind

    /// Move `snapshot`'s cache objects out under `grant`, keeping the
    /// recurrent backup: the check-out's only array allocation.
    static func take(
        _ snapshot: HybridCacheSnapshot, under grant: PrefixCacheManager.LeafLeaseGrant
    ) -> CheckedOutLeaf {
        guard let (cache, kinds) = snapshot.takeMovingCache() else {
            preconditionFailure("an eligible leased leaf must own cache objects")
        }
        return CheckedOutLeaf(
            grant: grant, live: FinalGenerationCache(cache),
            rewind: LeafRewind(cache: cache, kinds: kinds, offset: grant.lease.offset))
    }

    /// **Leaf Rewind**: trim and rebuild the live cache to the leased
    /// offset, compact the attention capacity it grew into, and move it back
    /// under the leased node, ending the lease. Inside the Model Session,
    /// after generation has quiesced. Returns the compaction and the
    /// rewound cache's facts, read before the move takes the objects.
    func returnByRewind() async -> (
        compaction: AttentionCapacityCompaction.Outcome, facts: [String: String]
    ) {
        live.rewind(with: rewind)
        // The rewound cache's attention arrays keep the capacity the aborted
        // generation grew into; the leaf inherits it (#501 measured, #534
        // compacts above the threshold).
        let compaction = AttentionCapacityCompaction.compactIfNeeded(live.cache)
        let facts = RequestMemoryTelemetry.cacheFacts(live.cache)
        guard let body = live.moveSnapshot(offset: grant.lease.offset) else {
            preconditionFailure("a checked-out cache must remain capturable")
        }
        let refusal = await MainActor.run { grant.rewind(body) }
        precondition(refusal == nil, "the lease must accept its own leaf back on rewind")
        return (compaction, facts)
    }
}

/// What a tripped claim held when it tripped: its lane and its pins, which
/// the tripwire released, and a Leaf Lease it could not return.
nonisolated struct CacheClaimTripwireEvent: PrefixCacheDiagnostics.Payload {
    let violation: CacheClaim.Violation
    let owner: String
    let claimRequestID: UUID
    let releasedLane: Bool
    let releasedPins: Int
    let heldLease: LeafLease?
    let eventName = "cacheClaimTripwire"

    var fields: [(String, String)] {
        var fields = [
            ("violation", violation.rawValue), ("owner", owner),
            ("claimRequestID", claimRequestID.uuidString), ("lane", "\(releasedLane)"),
            ("pins", "\(releasedPins)"),
        ]
        if let heldLease {
            fields += [
                ("heldLeaseID", heldLease.id.uuidString),
                ("heldLeaseOffset", "\(heldLease.offset)"),
                ("heldLeaseBytes", "\(heldLease.bytes)"),
            ]
        } else {
            fields.append(("heldLeaseID", "none"))
        }
        return fields
    }
}
