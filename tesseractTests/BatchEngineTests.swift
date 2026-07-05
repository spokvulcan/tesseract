import Foundation
import Testing
import os

@testable import Tesseract_Agent

/// The **Batch Engine**'s observable contract (PRD #173, ADR-0022), driven
/// with fake step bodies over a real `GPULeaseQueue`-backed lease: admission
/// respects cap and headroom, LPM order with aging, decode progresses during
/// a sibling's chunked prefill, Boundary Yield admits a waiting
/// slot-preserving consumer, Admission Freeze drains before an exclusive or
/// different-model run, cancellation frees a lane for a queued request, and
/// the 60 s busy contract survives as an admission timeout.
@MainActor
@Suite(.serialized) struct BatchEngineTests {

    // MARK: - Rig

    /// Records ordered string events from any isolation.
    actor StepRecorder {
        private(set) var events: [String] = []
        func add(_ event: String) { events.append(event) }
        func index(of event: String) -> Int? { events.firstIndex(of: event) }
    }

    /// A real lease (FIFO, atomic handoff) plus the waiter signal, standing
    /// in for the arbiter. Records every acquisition's demand and lets a
    /// test mutate the "loaded model" the compatibility oracle sees.
    @MainActor
    final class TestLease {
        let queue = GPULeaseQueue()
        let signal = LeaseWaiterSignal()
        private(set) var acquisitions: [BatchModelDemand] = []
        /// The pretend-loaded model; the lease runner "reloads" by assigning
        /// the acquisition's override, mirroring `ensureLoaded`.
        private(set) var loadedModelID: String?

        func runner() -> BatchLeaseRunner {
            { [self] override, vision, body in
                try await queue.withExclusive {
                    acquisitions.append(
                        BatchModelDemand(modelIDOverride: override, vision: vision))
                    if let override { loadedModelID = override }
                    await body()
                }
            }
        }

        /// A slot-preserving consumer's acquisition, signal choreography
        /// mirroring `InferenceArbiter.withExclusiveGPU(.tts)`.
        func runSlotPreserving(_ body: @escaping () async -> Void) async throws {
            signal.increment()
            var acquired = false
            do {
                try await queue.withExclusive {
                    acquired = true
                    signal.decrement()
                    await body()
                }
            } catch {
                if !acquired { signal.decrement() }
                throw error
            }
        }
    }

    private static let generousBudget = BatchLaneBudget(
        headroomBytes: 100 << 30, evictableCacheBytes: 0, perLaneBytes: 4 << 30)
    private static let singleLaneBudget = BatchLaneBudget(
        headroomBytes: 1 << 30, evictableCacheBytes: 0, perLaneBytes: 4 << 30)

    private func makeEngine(
        lease: TestLease,
        budget: BatchLaneBudget = generousBudget,
        satisfied: (@Sendable @MainActor (BatchModelDemand) -> Bool)? = nil
    ) -> BatchEngine {
        let defaultOracle: @Sendable @MainActor (BatchModelDemand) -> Bool = { demand in
            demand.modelIDOverride == nil
                || demand.modelIDOverride == lease.loadedModelID
        }
        return BatchEngine(
            leaseRunner: lease.runner(),
            leaseWaiters: lease.signal,
            demandSatisfied: satisfied ?? defaultOracle,
            laneBudget: { budget }
        )
    }

    private func submission(
        _ id: UUID,
        match: Int = 0,
        images: Bool = false,
        monolithic: Bool = false,
        modelID: String? = nil,
        timeout: Duration = .seconds(60)
    ) -> BatchSubmission {
        BatchSubmission(
            requestID: id,
            demand: BatchModelDemand(modelIDOverride: modelID, vision: .fromSettings),
            bearsImages: images,
            runsMonolithic: monolithic,
            matchedPrefixLength: match,
            admissionTimeout: timeout
        )
    }

    private func eventually(
        timeout: Duration = .seconds(5),
        _ condition: @escaping () async -> Bool
    ) async -> Bool {
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: timeout)
        while clock.now < deadline {
            if await condition() { return true }
            try? await Task.sleep(for: .milliseconds(5))
        }
        return await condition()
    }

    // MARK: - Basic lane lifecycle

    @Test func submittedCompletionRunsItsStepsInOrderAndDrains() async throws {
        let lease = TestLease()
        let engine = makeEngine(lease: lease)
        let recorder = StepRecorder()
        let laneID = UUID()

        let admission = try await engine.submit(submission(laneID))
        #expect(!admission.isExclusive)

        _ = try await engine.step(lane: laneID, kind: .startup) { _ in
            await recorder.add("startup")
        }
        for index in 0..<3 {
            _ = try await engine.step(lane: laneID, kind: .decode) { _ in
                await recorder.add("decode\(index)")
            }
        }
        await engine.laneFinished(laneID)

        #expect(await recorder.events == ["startup", "decode0", "decode1", "decode2"])
        #expect(lease.acquisitions.count == 1)
        // The engine releases the lease once the last lane drains.
        let released = await eventually { await !lease.queue.isLeased }
        #expect(released)
    }

    // MARK: - Admission: capacity and order

    @Test func headroomCapacityHoldsSecondSubmissionUntilLaneFinishes() async throws {
        let lease = TestLease()
        let engine = makeEngine(lease: lease, budget: Self.singleLaneBudget)
        let laneA = UUID()
        let laneB = UUID()
        let recorder = StepRecorder()

        _ = try await engine.submit(submission(laneA))
        let second = Task {
            _ = try await engine.submit(submission(laneB))
            await recorder.add("B admitted")
            await engine.laneFinished(laneB)
        }
        // Give the engine room to (incorrectly) admit B.
        for _ in 0..<50 { await Task.yield() }
        #expect(await recorder.events.isEmpty)

        await engine.laneFinished(laneA)
        try await second.value
        #expect(await recorder.events == ["B admitted"])
    }

    @Test func queuedSubmissionsAdmitInLongestPrefixMatchOrder() async throws {
        let lease = TestLease()
        let engine = makeEngine(lease: lease, budget: Self.singleLaneBudget)
        let laneA = UUID()
        let shortMatch = UUID()
        let longMatch = UUID()
        let recorder = StepRecorder()

        _ = try await engine.submit(submission(laneA))
        let short = Task {
            _ = try await engine.submit(submission(shortMatch, match: 10))
            await recorder.add("short")
            await engine.laneFinished(shortMatch)
        }
        let long = Task {
            _ = try await engine.submit(submission(longMatch, match: 5_000))
            await recorder.add("long")
            await engine.laneFinished(longMatch)
        }
        // Both queued behind the capacity-1 pool.
        for _ in 0..<50 { await Task.yield() }
        await engine.laneFinished(laneA)

        try await short.value
        try await long.value
        #expect(await recorder.events == ["long", "short"])
    }

    // MARK: - Interleaving: decode streams during a sibling's prefill

    @Test func decodeStepsProgressDuringSiblingChunkedPrefill() async throws {
        let lease = TestLease()
        let engine = makeEngine(lease: lease)
        let decoder = UUID()
        let prefiller = UUID()
        let recorder = StepRecorder()

        _ = try await engine.submit(submission(decoder))
        _ = try await engine.step(lane: decoder, kind: .startup) { _ in
            await recorder.add("A.startup")
        }
        let decodeDrive = Task {
            for index in 0..<30 {
                _ = try await engine.step(lane: decoder, kind: .decode) { _ in
                    await recorder.add("A.decode\(index)")
                    try? await Task.sleep(for: .milliseconds(2))
                }
            }
            await engine.laneFinished(decoder)
        }

        var chunkGrants: [Int] = []
        _ = try await engine.submit(submission(prefiller))
        _ = try await engine.step(lane: prefiller, kind: .startup) { _ in
            await recorder.add("B.startup")
        }
        for index in 0..<4 {
            let granted = try await engine.step(lane: prefiller, kind: .prefillChunk) {
                grant in
                await recorder.add("B.chunk\(index)")
                try? await Task.sleep(for: .milliseconds(2))
                return grant.prefillChunkTokens
            }
            chunkGrants.append(granted)
        }
        await engine.laneFinished(prefiller)
        try await decodeDrive.value

        // Chunks shrank because a sibling was decoding (ADR-0022 ~512).
        #expect(chunkGrants.allSatisfy { $0 == 512 })

        // The decode stream did not freeze for the prefill's duration:
        // decode steps landed between the first and last prefill chunk.
        let events = await recorder.events
        let firstChunk = try #require(events.firstIndex(of: "B.chunk0"))
        let lastChunk = try #require(events.firstIndex(of: "B.chunk3"))
        let decodesBetween = events[firstChunk...lastChunk].filter {
            $0.hasPrefix("A.decode")
        }
        #expect(!decodesBetween.isEmpty)
    }

    // MARK: - Boundary Yield

    @Test func boundaryYieldAdmitsWaitingSlotPreservingConsumer() async throws {
        let lease = TestLease()
        let engine = makeEngine(lease: lease)
        let laneID = UUID()
        let recorder = StepRecorder()

        _ = try await engine.submit(submission(laneID))
        let drive = Task {
            for index in 0..<40 {
                _ = try await engine.step(lane: laneID, kind: .decode) { _ in
                    await recorder.add("decode\(index)")
                    try? await Task.sleep(for: .milliseconds(2))
                }
            }
            await engine.laneFinished(laneID)
        }
        // Let the lane get going.
        _ = await eventually { await recorder.index(of: "decode3") != nil }

        try await lease.runSlotPreserving {
            await recorder.add("tts")
        }

        try await drive.value
        let events = await recorder.events
        let tts = try #require(events.firstIndex(of: "tts"))
        // The consumer ran within the lane's stream — not after pool idle —
        // and the lane kept decoding afterward.
        #expect(events.last != "tts")
        #expect(tts < events.count - 1)
        #expect(events.firstIndex(of: "decode39")! > tts)
    }

    @Test func fullDrainReleasesLeaseBeforeServingTheNextQueuedEntry() async throws {
        let lease = TestLease()
        let engine = makeEngine(lease: lease, budget: Self.singleLaneBudget)
        let first = UUID()
        let second = UUID()
        let recorder = StepRecorder()

        _ = try await engine.submit(submission(first))
        // A non-slot-preserving consumer (`reloadLLMIfNeeded`, a direct
        // lease user) queues FIFO. The engine never yields to it mid-batch —
        // but a fully drained pool is a tenure boundary, so it must get its
        // turn before the engine's next admission, not starve behind a
        // stream of back-to-back completions.
        let reload = Task {
            try await lease.queue.withExclusive {
                await recorder.add("reload")
            }
        }
        for _ in 0..<50 { await Task.yield() }

        let successor = Task {
            _ = try await engine.submit(submission(second))
            await recorder.add("second.run")
            await engine.laneFinished(second)
        }
        for _ in 0..<50 { await Task.yield() }

        await engine.laneFinished(first)
        try await reload.value
        try await successor.value

        let events = await recorder.events
        let reloadIndex = try #require(events.firstIndex(of: "reload"))
        let secondIndex = try #require(events.firstIndex(of: "second.run"))
        #expect(reloadIndex < secondIndex)
    }

    // MARK: - Admission Freeze: exclusive requests drain the pool

    @Test func exclusiveRequestDrainsPoolRunsAloneThenPoolResumes() async throws {
        let lease = TestLease()
        let engine = makeEngine(lease: lease)
        let poolLane = UUID()
        let imageLane = UUID()
        let lateLane = UUID()
        let recorder = StepRecorder()

        _ = try await engine.submit(submission(poolLane))
        let poolDrive = Task {
            for index in 0..<10 {
                _ = try await engine.step(lane: poolLane, kind: .decode) { _ in
                    await recorder.add("pool.decode\(index)")
                    try? await Task.sleep(for: .milliseconds(2))
                }
            }
            await recorder.add("pool.done")
            await engine.laneFinished(poolLane)
        }
        _ = await eventually { await recorder.index(of: "pool.decode1") != nil }

        let imageRun = Task {
            let admission = try await engine.submit(
                submission(imageLane, images: true))
            await recorder.add("image.run")
            #expect(admission.isExclusive)
            // Hold the exclusive lane until the late entry is queued
            // behind it, so the freeze-during-exclusive phase is observable.
            _ = await eventually {
                await recorder.index(of: "late.submitted") != nil
            }
            await engine.laneFinished(imageLane)
        }

        // The exclusive request admits only once the pool has drained.
        try await poolDrive.value
        _ = await eventually { await recorder.index(of: "image.run") != nil }

        let lateRun = Task {
            // Same match, later arrival — stays behind the exclusive head.
            // (A *higher* match may legitimately jump a fresh exclusive
            // entry: queue order is LPM-first, aged to strict FIFO —
            // policy-tested.)
            _ = try await engine.submit(submission(lateLane))
            await recorder.add("late.run")
            await engine.laneFinished(lateLane)
        }
        // While the exclusive lane is live, admissions stay frozen.
        for _ in 0..<50 { await Task.yield() }
        #expect(await recorder.index(of: "late.run") == nil)
        await recorder.add("late.submitted")

        try await imageRun.value
        try await lateRun.value

        let events = await recorder.events
        let poolDone = try #require(events.firstIndex(of: "pool.done"))
        let image = try #require(events.firstIndex(of: "image.run"))
        let late = try #require(events.firstIndex(of: "late.run"))
        #expect(poolDone < image)
        #expect(image < late)
    }

    // MARK: - Model switch: drain, re-acquire under the new demand, pool

    @Test func differentModelRequestDrainsThenReacquiresAndPoolsNormally() async throws {
        let lease = TestLease()
        let engine = makeEngine(lease: lease)
        let poolLane = UUID()
        let modelB = UUID()
        let recorder = StepRecorder()

        _ = try await engine.submit(submission(poolLane))
        let poolDrive = Task {
            for index in 0..<8 {
                _ = try await engine.step(lane: poolLane, kind: .decode) { _ in
                    await recorder.add("pool.decode\(index)")
                    try? await Task.sleep(for: .milliseconds(2))
                }
            }
            await recorder.add("pool.done")
            await engine.laneFinished(poolLane)
        }
        _ = await eventually { await recorder.index(of: "pool.decode1") != nil }

        let switchRun = Task {
            let admission = try await engine.submit(
                submission(modelB, modelID: "model-b"))
            await recorder.add("b.run")
            // After the reload the request pools normally — not exclusive.
            #expect(!admission.isExclusive)
            await engine.laneFinished(modelB)
        }

        try await poolDrive.value
        try await switchRun.value

        let events = await recorder.events
        let poolDone = try #require(events.firstIndex(of: "pool.done"))
        let bRun = try #require(events.firstIndex(of: "b.run"))
        #expect(poolDone < bRun)
        // Second acquisition carried the new demand for the arbiter reload.
        #expect(lease.acquisitions.map(\.modelIDOverride) == [nil, "model-b"])
        #expect(lease.loadedModelID == "model-b")
    }

    // MARK: - Cancellation and timeout

    @Test func cancelledLaneFreesCapacityForQueuedRequest() async throws {
        let lease = TestLease()
        let engine = makeEngine(lease: lease, budget: Self.singleLaneBudget)
        let cancelled = UUID()
        let queued = UUID()
        let recorder = StepRecorder()

        _ = try await engine.submit(submission(cancelled))
        let doomed = Task {
            do {
                while true {
                    _ = try await engine.step(lane: cancelled, kind: .decode) { _ in
                        await recorder.add("doomed.decode")
                        try? await Task.sleep(for: .milliseconds(2))
                    }
                }
            } catch {
                await recorder.add("doomed.cancelled")
                await engine.laneFinished(cancelled)
            }
        }
        _ = await eventually { await recorder.index(of: "doomed.decode") != nil }

        let successor = Task {
            _ = try await engine.submit(submission(queued))
            await recorder.add("queued.run")
            await engine.laneFinished(queued)
        }
        for _ in 0..<50 { await Task.yield() }
        #expect(await recorder.index(of: "queued.run") == nil)

        doomed.cancel()
        await doomed.value
        try await successor.value

        let events = await recorder.events
        let torn = try #require(events.firstIndex(of: "doomed.cancelled"))
        let ran = try #require(events.firstIndex(of: "queued.run"))
        #expect(torn < ran)
    }

    @Test func admissionDeadlinePreservesTheBusyContract() async throws {
        let lease = TestLease()
        let engine = makeEngine(lease: lease, budget: Self.singleLaneBudget)
        let blocker = UUID()
        let starved = UUID()

        _ = try await engine.submit(submission(blocker))
        await #expect(throws: LeaseTimeoutError.self) {
            _ = try await engine.submit(
                submission(starved, timeout: .milliseconds(50)))
        }
        await engine.laneFinished(blocker)
    }

    @Test func disconnectWhileQueuedRemovesTheEntry() async throws {
        let lease = TestLease()
        let engine = makeEngine(lease: lease, budget: Self.singleLaneBudget)
        let blocker = UUID()
        let leaver = UUID()
        let follower = UUID()
        let recorder = StepRecorder()

        _ = try await engine.submit(submission(blocker))
        let leavingTask = Task {
            do {
                _ = try await engine.submit(submission(leaver))
                await engine.laneFinished(leaver)
                return false
            } catch {
                return error is CancellationError
            }
        }
        for _ in 0..<50 { await Task.yield() }
        leavingTask.cancel()
        #expect(await leavingTask.value)

        // The abandoned entry is gone: the next submission is served, not
        // queued behind a ghost.
        let followerTask = Task {
            _ = try await engine.submit(submission(follower))
            await recorder.add("follower.run")
            await engine.laneFinished(follower)
        }
        await engine.laneFinished(blocker)
        try await followerTask.value
        #expect(await recorder.events == ["follower.run"])
    }

    // MARK: - Lane lifecycle diagnostics (PRD #173 user story 23)

    /// Lifecycle events keyed by request identity flow to the diagnostics
    /// sink: queued → admitted → firstToken → drained, with the match length
    /// on `queued` (the LPM-vs-FIFO evidence) and durations on the later
    /// phases so per-lane TTFT and queue wait derive from the record.
    @Test func laneLifecycleEventsReachTheDiagnosticsSink() async throws {
        let lease = TestLease()
        let engine = makeEngine(lease: lease)
        let laneID = UUID()

        let lines = OSAllocatedUnfairLock(initialState: [String]())
        let sink = PrefixCacheDiagnostics.addTestSink { line in
            lines.withLock { $0.append(line) }
        }
        defer { PrefixCacheDiagnostics.removeTestSink(sink) }

        _ = try await engine.submit(submission(laneID, match: 128))
        _ = try await engine.step(lane: laneID, kind: .startup) { _ in }
        _ = try await engine.step(lane: laneID, kind: .decode) { _ in }
        _ = try await engine.step(lane: laneID, kind: .decode) { _ in }
        await engine.laneFinished(laneID)

        let laneLines = lines.withLock { $0 }.filter {
            $0.contains("event=lane") && $0.contains("requestID=\(laneID.uuidString)")
        }
        let phases = laneLines.compactMap { line in
            line.split(separator: " ")
                .first(where: { $0.hasPrefix("phase=") })?
                .dropFirst("phase=".count)
        }.map(String.init)
        #expect(phases == ["queued", "admitted", "firstToken", "drained"])
        let queued = try #require(laneLines.first)
        #expect(queued.contains("matchedPrefixLength=128"))
        #expect(queued.contains("exclusive=false"))
        let admitted = try #require(laneLines.dropFirst().first)
        #expect(admitted.contains("queueWaitMs="))
        let firstToken = try #require(laneLines.dropFirst(2).first)
        #expect(firstToken.contains("sinceAdmissionMs="))
    }

    /// An exclusive (monolithic) lane never runs decode steps, so its record
    /// is queued → admitted → drained — no phantom firstToken.
    @Test func exclusiveLaneLifecycleSkipsFirstToken() async throws {
        let lease = TestLease()
        let engine = makeEngine(lease: lease)
        let laneID = UUID()

        let lines = OSAllocatedUnfairLock(initialState: [String]())
        let sink = PrefixCacheDiagnostics.addTestSink { line in
            lines.withLock { $0.append(line) }
        }
        defer { PrefixCacheDiagnostics.removeTestSink(sink) }

        _ = try await engine.submit(submission(laneID, monolithic: true))
        await engine.laneFinished(laneID)

        let phases = lines.withLock { $0 }
            .filter {
                $0.contains("event=lane")
                    && $0.contains("requestID=\(laneID.uuidString)")
            }
            .compactMap { line in
                line.split(separator: " ")
                    .first(where: { $0.hasPrefix("phase=") })?
                    .dropFirst("phase=".count)
            }.map(String.init)
        #expect(phases == ["queued", "admitted", "drained"])
        let queuedLine = lines.withLock { $0 }.first {
            $0.contains("requestID=\(laneID.uuidString)") && $0.contains("phase=queued")
        }
        #expect(queuedLine?.contains("exclusive=true") == true)
    }
}
