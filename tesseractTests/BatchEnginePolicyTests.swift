import Foundation
import Testing

@testable import Tesseract_Agent

/// The **Batch Engine**'s pure scheduling core (PRD #173, ADR-0022): the
/// per-iteration decision — admit whom / prefill chunk or decode step /
/// yield / freeze — computed as a plain value from a snapshot of queue,
/// lanes, and lease waiters. Prior art: `PrefillPlannerTests`,
/// `LeafAdmissionBuilderTests`.
@Suite struct BatchEnginePolicyTests {

    // MARK: - Helpers

    private func entry(
        _ arrivalOrder: Int,
        waited: TimeInterval = 0,
        match: Int = 0,
        exclusive: Bool = false,
        satisfied: Bool = true
    ) -> BatchEngineSnapshot.QueueEntry {
        BatchEngineSnapshot.QueueEntry(
            requestID: UUID(
                uuidString: String(
                    format: "00000000-0000-0000-0000-%012d", arrivalOrder)
            )!,
            arrivalOrder: arrivalOrder,
            waitedSeconds: waited,
            matchedPrefixLength: match,
            requiresExclusivePool: exclusive,
            demandSatisfiedByLoadedModel: satisfied
        )
    }

    // MARK: - Queue order: longest prefix match, aged to strict FIFO

    @Test func freshEntriesOrderByLongestMatchThenArrival() {
        let ordered = BatchEnginePolicy.orderedQueue([
            entry(0, match: 100),
            entry(1, match: 4_000),
            entry(2, match: 4_000),
            entry(3, match: 900),
        ])
        #expect(ordered.map(\.arrivalOrder) == [1, 2, 3, 0])
    }

    @Test func agedEntriesPrecedeFreshInArrivalOrder() {
        // Entry 2 has the longest match but entries 0 and 1 have aged past
        // the threshold — cache-aware ordering must never starve a cold
        // prefix into the admission timeout (user story 9).
        let ordered = BatchEnginePolicy.orderedQueue([
            entry(0, waited: 11, match: 0),
            entry(1, waited: 12, match: 50),
            entry(2, waited: 1, match: 9_000),
        ])
        #expect(ordered.map(\.arrivalOrder) == [0, 1, 2])
    }

    @Test func entryAtExactAgingThresholdCountsAsAged() {
        let ordered = BatchEnginePolicy.orderedQueue([
            entry(0, waited: BatchEnginePolicy.agingThresholdSeconds, match: 0),
            entry(1, waited: 0, match: 5_000),
        ])
        #expect(ordered.map(\.arrivalOrder) == [0, 1])
    }

    @Test func emptyQueueOrdersToEmpty() {
        #expect(BatchEnginePolicy.orderedQueue([]).isEmpty)
    }

    // MARK: - Lane Admission and Admission Freeze

    private func laneID(_ order: Int) -> UUID {
        UUID(uuidString: String(format: "00000000-0000-0000-0001-%012d", order))!
    }

    private func lane(
        _ admissionOrder: Int,
        phase: BatchEngineSnapshot.LanePhase,
        pending: Bool = true,
        chunk: Int = 1_024,
        exclusive: Bool = false
    ) -> BatchEngineSnapshot.Lane {
        BatchEngineSnapshot.Lane(
            requestID: laneID(admissionOrder),
            admissionOrder: admissionOrder,
            phase: phase,
            hasPendingStep: pending,
            preferredPrefillChunk: chunk,
            isExclusive: exclusive
        )
    }

    private func snapshot(
        queue: [BatchEngineSnapshot.QueueEntry] = [],
        lanes: [BatchEngineSnapshot.Lane] = [],
        capacity: Int = 4,
        waiters: Bool = false,
        lastGrantWasPrefillClass: Bool = false
    ) -> BatchEngineSnapshot {
        BatchEngineSnapshot(
            queue: queue,
            lanes: lanes,
            laneCapacity: capacity,
            leaseHasWaiters: waiters,
            lastGrantWasPrefillClass: lastGrantWasPrefillClass
        )
    }

    @Test func admitsTheOrderedHeadNotTheArrivalHead() {
        let decision = BatchEnginePolicy.decide(
            snapshot(queue: [entry(0, match: 0), entry(1, match: 500)])
        )
        #expect(decision == .admit(entry(1, match: 500).requestID))
    }

    @Test func admissionStopsAtCapacity() {
        let decision = BatchEnginePolicy.decide(
            snapshot(
                queue: [entry(0)],
                lanes: [lane(0, phase: .decoding), lane(1, phase: .decoding)],
                capacity: 2
            ))
        #expect(decision == .decodeRound([laneID(0), laneID(1)]))
    }

    @Test func exclusiveHeadFreezesAdmissionsWhileLanesLive() {
        // The model-switch / image request has aged to the head: nothing may
        // be admitted — not even a compatible later entry — while the pool
        // drains by attrition (ADR-0022 Admission Freeze).
        let decision = BatchEnginePolicy.decide(
            snapshot(
                queue: [
                    entry(0, waited: 11, exclusive: true),
                    entry(1, match: 9_000),
                ],
                lanes: [lane(0, phase: .decoding)]
            ))
        #expect(decision == .decodeRound([laneID(0)]))
    }

    @Test func exclusiveHeadAdmittedAloneOncePoolIsEmpty() {
        let decision = BatchEnginePolicy.decide(
            snapshot(queue: [entry(0, exclusive: true)])
        )
        #expect(decision == .admitExclusive(entry(0, exclusive: true).requestID))
    }

    @Test func exclusiveLaneBlocksAllAdmissions() {
        let decision = BatchEnginePolicy.decide(
            snapshot(
                queue: [entry(0, match: 5_000)],
                lanes: [lane(0, phase: .decoding, exclusive: true)]
            ))
        #expect(decision == .decodeRound([laneID(0)]))
    }

    @Test func unsatisfiedHeadFreezesAdmissionsWhileLanesLive() {
        // A request naming a different model (or demanding a vision upgrade)
        // at the head: the pool must drain before the arbiter can reload —
        // no admissions meanwhile, not even compatible later entries.
        let decision = BatchEnginePolicy.decide(
            snapshot(
                queue: [
                    entry(0, waited: 11, satisfied: false),
                    entry(1, match: 9_000),
                ],
                lanes: [lane(0, phase: .decoding)]
            ))
        #expect(decision == .decodeRound([laneID(0)]))
    }

    @Test func unsatisfiedHeadTriggersPoolSwitchOncePoolIsEmpty() {
        // Empty pool: the engine must release the lease and re-acquire with
        // this entry's demand (the arbiter reloads), then admit normally —
        // a burst of same-model requests pools after ONE drain.
        let decision = BatchEnginePolicy.decide(
            snapshot(queue: [entry(0, satisfied: false)])
        )
        #expect(decision == .switchPool(entry(0).requestID))
    }

    @Test func unsatisfiedDemandOutranksExclusivePoolRequirement() {
        // An image-bearing request for a different model needs the reload
        // first, then the solo run.
        let decision = BatchEnginePolicy.decide(
            snapshot(queue: [entry(0, exclusive: true, satisfied: false)])
        )
        #expect(decision == .switchPool(entry(0).requestID))
    }

    // MARK: - Boundary Yield

    @Test func yieldBeatsEveryOtherAction() {
        let decision = BatchEnginePolicy.decide(
            snapshot(
                queue: [entry(0)],
                lanes: [lane(0, phase: .decoding)],
                waiters: true
            ))
        #expect(decision == .yieldLease)
    }

    @Test func neverYieldsWhileAnExclusiveLaneIsLive() {
        // An exclusive lane's generation runs today's monolithic path — its
        // Metal work is not inside granted steps, so there is no safe step
        // boundary to yield at. The waiter gets the lease FIFO once the lane
        // drains, exactly today's semantics.
        let decision = BatchEnginePolicy.decide(
            snapshot(
                lanes: [lane(0, phase: .decoding, pending: false, exclusive: true)],
                waiters: true
            ))
        #expect(decision == .idle)
    }

    // MARK: - Step loop: one prefill chunk alternates with decode steps

    @Test func startupGrantedInAdmissionOrder() {
        let decision = BatchEnginePolicy.decide(
            snapshot(lanes: [lane(1, phase: .starting), lane(0, phase: .starting)])
        )
        #expect(decision == .grant(laneID(0), .startup))
    }

    @Test func finishPrecedesOtherPrefillClassWork() {
        // Finishing frees a lane (and its pages) — capacity returns to the
        // pool before new heavy work starts.
        let decision = BatchEnginePolicy.decide(
            snapshot(lanes: [lane(0, phase: .starting), lane(1, phase: .finishing)])
        )
        #expect(decision == .grant(laneID(1), .finish))
    }

    @Test func onePrefillAtATimeInAdmissionOrder() {
        let decision = BatchEnginePolicy.decide(
            snapshot(lanes: [lane(0, phase: .prefilling), lane(1, phase: .prefilling)])
        )
        #expect(decision == .grant(laneID(0), .prefillChunk(tokens: 1_024)))
    }

    @Test func prefillChunkShrinksWhileAnyLaneDecodes() {
        // The decoding sibling need not have a step pending — its next stall
        // is what the smaller chunk bounds (ADR-0022: ~512 while decoding).
        let decision = BatchEnginePolicy.decide(
            snapshot(
                lanes: [
                    lane(0, phase: .decoding, pending: false),
                    lane(1, phase: .prefilling),
                ],
                lastGrantWasPrefillClass: false
            ))
        #expect(decision == .grant(laneID(1), .prefillChunk(tokens: 512)))
    }

    @Test func prefillChunkNeverExceedsLanePreference() {
        let decision = BatchEnginePolicy.decide(
            snapshot(
                lanes: [
                    lane(0, phase: .decoding, pending: false),
                    lane(1, phase: .prefilling, chunk: 256),
                ]
            ))
        #expect(decision == .grant(laneID(1), .prefillChunk(tokens: 256)))
    }

    @Test func decodeRoundAlternatesAfterPrefillClassGrant() {
        let decision = BatchEnginePolicy.decide(
            snapshot(
                lanes: [
                    lane(0, phase: .decoding),
                    lane(1, phase: .prefilling),
                ],
                lastGrantWasPrefillClass: true
            ))
        #expect(decision == .decodeRound([laneID(0)]))
    }

    @Test func prefillResumesAfterDecodeRound() {
        let decision = BatchEnginePolicy.decide(
            snapshot(
                lanes: [
                    lane(0, phase: .decoding),
                    lane(1, phase: .prefilling),
                ],
                lastGrantWasPrefillClass: false
            ))
        #expect(decision == .grant(laneID(1), .prefillChunk(tokens: 512)))
    }

    @Test func prefillContinuesWhenNoDecodeIsPending() {
        let decision = BatchEnginePolicy.decide(
            snapshot(
                lanes: [lane(0, phase: .prefilling)],
                lastGrantWasPrefillClass: true
            ))
        #expect(decision == .grant(laneID(0), .prefillChunk(tokens: 1_024)))
    }

    @Test func lanesWithoutPendingStepAreSkippedInDecodeRound() {
        let decision = BatchEnginePolicy.decide(
            snapshot(
                lanes: [
                    lane(0, phase: .decoding, pending: false),
                    lane(1, phase: .decoding),
                ]
            ))
        #expect(decision == .decodeRound([laneID(1)]))
    }

    @Test func idleWhenNothingIsPending() {
        let decision = BatchEnginePolicy.decide(
            snapshot(lanes: [lane(0, phase: .decoding, pending: false)])
        )
        #expect(decision == .idle)
    }

    // MARK: - Headroom lanes: serialization by arithmetic, not policy

    private let gib = 1 << 30

    @Test func abundantHeadroomClampsToTheHardCap() {
        let lanes = BatchEnginePolicy.admittableLanes(
            headroomBytes: 100 * gib,
            evictableCacheBytes: 0,
            perLaneBytes: 4 * gib
        )
        #expect(lanes == 4)
    }

    @Test func bigModelHeadroomSerializesToOneLane() {
        // A model that leaves no working-set headroom degenerates to N=1 by
        // arithmetic — batching can never reintroduce a swap incident
        // (user story 12).
        let lanes = BatchEnginePolicy.admittableLanes(
            headroomBytes: 1 * gib,
            evictableCacheBytes: 0,
            perLaneBytes: 4 * gib
        )
        #expect(lanes == 1)
    }

    @Test func evictableCacheCountsAsClaimableRoom() {
        // 5 GiB headroom claims 4 GiB (the ceiling's 0.8 damping); alone
        // that is one 4 GiB lane, but 4 GiB of evictable cache — which a
        // drain can demote — funds the second.
        let alone = BatchEnginePolicy.admittableLanes(
            headroomBytes: 5 * gib,
            evictableCacheBytes: 0,
            perLaneBytes: 4 * gib
        )
        let withCache = BatchEnginePolicy.admittableLanes(
            headroomBytes: 5 * gib,
            evictableCacheBytes: 4 * gib,
            perLaneBytes: 4 * gib
        )
        #expect(alone == 1)
        #expect(withCache == 2)
    }

    @Test func benchOverrideCapIsRespected() {
        let lanes = BatchEnginePolicy.admittableLanes(
            headroomBytes: 50 * gib,
            evictableCacheBytes: 0,
            perLaneBytes: 4 * gib,
            hardCap: 8
        )
        #expect(lanes == 8)
    }
}
