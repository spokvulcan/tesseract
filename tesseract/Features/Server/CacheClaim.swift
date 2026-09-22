import Foundation
import os

/// **Cache Claim** — one keyed request's whole hold on the prefix cache
/// (ADR-0069): its lane in the **Active-Inference Reserve** and its
/// **Restore Pins**. Snapshot Resolution adds them; the claim concludes
/// exactly once, inside the request's GPU lease, and only a claim lets go
/// of what it holds.
///
/// A claim is held only inside an owner scope, one owner at a time. `start`
/// owns a request's claim inside `withRequestClaim` and hands it to the
/// drive, which redeems the hand-over with `HandOver.withClaim`. A
/// **Speculative Canonical Prefill** pass owns a copy-only claim of its own
/// (`withCopyOnlyClaim`). Each scope concludes on exit, whether it returns
/// or throws: it lets go of the pins, then the lane, in one MainActor hop.
/// The conclusion never reads task cancellation (stream termination cancels
/// the drive task on a normal finish too) and runs in a detached task the
/// scope awaits, so cancellation cannot cut it short.
///
/// Enter an owner scope outside the Model Session. Debug builds trap when a
/// scope opens inside one.
///
/// What the types cannot prevent trips the tripwire: a claim dropped
/// without concluding (a hand-over never redeemed), a hand-over redeemed
/// twice, a step taken while the claim is between owners or after it
/// concluded. Debug builds trap. Release builds log an error, emit a
/// `cacheClaimTripwire` event naming the request and what it held, and let
/// go of the pins and the lane on the MainActor.
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
            let redeemed = claim.redeem()
            defer { if redeemed { await claim.conclude() } }
            return try await body(claim)
        }
    }

    /// What a debug build traps on and a release build reports.
    enum Violation: String, Sendable {
        case droppedUnconcluded, handOverRedeemedTwice, concludedTwice
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
    private let memory: RequestMemoryTelemetry?
    private let tripwire: Tripwire
    private let lock = NSLock()
    private var owner: Owner

    private init(
        requestID: UUID, owner: Owner, context: PrefixCacheDiagnostics.Context,
        prefixCache: PrefixCacheManager, memory: RequestMemoryTelemetry?, tripwire: Tripwire
    ) {
        self.requestID = requestID
        self.owner = owner
        self.context = context
        self.prefixCache = prefixCache
        self.memory = memory
        self.tripwire = tripwire
    }

    deinit {
        let owner = lock.withLock { self.owner }
        guard owner != .concluded else { return }
        Self.trip(
            .droppedUnconcluded, owner: owner, requestID: requestID, context: context,
            prefixCache: prefixCache, tripwire: tripwire)
    }

    // MARK: - Owner scopes

    /// Hold a request's claim for `body`, the request's `start`. A normal
    /// return hands the claim over; a throw concludes it here. The claim's
    /// request is `context.requestID`.
    static func withRequestClaim<R>(
        context: PrefixCacheDiagnostics.Context,
        prefixCache: PrefixCacheManager,
        memory: RequestMemoryTelemetry?,
        tripwire: Tripwire = .standard,
        _ body: (CacheClaim) async throws -> R
    ) async rethrows -> (value: R, handOver: HandOver) {
        assert(!ModelSessionScope.isInside, "open a Cache Claim outside the Model Session")
        let claim = CacheClaim(
            requestID: context.requestID, owner: .start, context: context,
            prefixCache: prefixCache, memory: memory, tripwire: tripwire)
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
            memory: nil, tripwire: tripwire)
        defer { await claim.conclude() }
        return try await body(CopyOnlyClaim(claim: claim))
    }

    // MARK: - Steps

    /// The request Snapshot Resolution adds this claim's lane and pins
    /// under, or `nil` when the claim is between owners or has concluded
    /// (the tripwire fires and resolution pins nothing).
    func resolutionEntry() -> UUID? {
        let owner = lock.withLock { self.owner }
        switch owner {
        case .start, .drive, .copyOnly:
            return requestID
        case .inTransit:
            trip(.stepBetweenOwners, owner: owner)
            return nil
        case .concluded:
            trip(.stepAfterConclusion, owner: owner)
            return nil
        }
    }

    // MARK: - Ownership

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

    /// Exactly once, by an owner scope: let go of the pins, then the lane.
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
        let prefixCache = prefixCache
        let memory = memory
        await Task.detached {
            memory?.mark(.releasingRequest)
            let facts = await MainActor.run {
                prefixCache.release(token)
                return prefixCache.memoryTelemetryFacts()
            }
            memory?.mark(.releasingRequest, facts: facts)
        }.value
    }

    // MARK: - Tripwire

    private func trip(_ violation: Violation, owner: Owner) {
        Self.trip(
            violation, owner: owner, requestID: requestID, context: context,
            prefixCache: prefixCache, tripwire: tripwire)
    }

    private static func trip(
        _ violation: Violation, owner: Owner, requestID: UUID,
        context: PrefixCacheDiagnostics.Context, prefixCache: PrefixCacheManager,
        tripwire: Tripwire
    ) {
        let message =
            "Cache Claim tripwire: \(violation.rawValue) — request \(requestID.uuidString) "
            + "(\(owner.rawValue)); letting go of its pins and lane (ADR-0069)"
        Log.server.error("\(message)")
        let token = Release(requestID: requestID)
        Task { @MainActor in
            let released = prefixCache.release(token)
            context.log(
                CacheClaimTripwireEvent(
                    violation: violation, owner: owner.rawValue, claimRequestID: requestID,
                    releasedLane: released.lane, releasedPins: released.pins),
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

/// What a tripped claim held when it tripped: its lane and its pins, which
/// the tripwire released.
nonisolated struct CacheClaimTripwireEvent: PrefixCacheDiagnostics.Payload {
    let violation: CacheClaim.Violation
    let owner: String
    let claimRequestID: UUID
    let releasedLane: Bool
    let releasedPins: Int
    let eventName = "cacheClaimTripwire"

    var fields: [(String, String)] {
        [
            ("violation", violation.rawValue), ("owner", owner),
            ("claimRequestID", claimRequestID.uuidString), ("lane", "\(releasedLane)"),
            ("pins", "\(releasedPins)"),
        ]
    }
}

/// Whether the current task is inside a Model Session. Both session
/// providers set it, so an owner scope opened inside a session — whose
/// conclusion may need the session again, and the session's lock is not
/// reentrant — traps in debug builds instead of deadlocking.
nonisolated enum ModelSessionScope {
    @TaskLocal static var isInside = false
}
