import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct PromptCacheTelemetryStoreTests {

    @Test func aggregateComputesHitRateTokenReuseAndLatency() {
        let events = [
            event(
                "lookup",
                fields: [
                    ("reason", "hit"),
                    ("promptTokens", "100"),
                    ("skippedPrefillTokens", "80"),
                    ("checkpointType", "system"),
                    ("lookupMs", "10"),
                    ("restoreMs", "3"),
                ]),
            event(
                "lookup",
                fields: [
                    ("reason", "ssdHit"),
                    ("promptTokens", "200"),
                    ("skippedPrefillTokens", "150"),
                    ("checkpointType", "leaf"),
                    ("lookupMs", "20"),
                    ("restoreMs", "8"),
                ]),
            event(
                "lookup",
                fields: [
                    ("reason", "missNoSnapshotInPrefix"),
                    ("promptTokens", "50"),
                    ("skippedPrefillTokens", "0"),
                    ("checkpointType", "nil"),
                    ("lookupMs", "5"),
                    ("restoreMs", "0"),
                ]),
            event(
                "ttft",
                fields: [
                    ("lookupMs", "15"),
                    ("restoreMs", "4"),
                    ("prefillMs", "44"),
                    ("ttftMs", "63"),
                ]),
            event("ssdAdmit", fields: [("outcome", "droppedSystemProtectionWins")]),
            event("eviction", fields: [("freedBytes", "4096")]),
        ]

        let aggregate = PromptCacheTelemetryAggregate.from(events: events)

        #expect(aggregate.lookupCount == 3)
        #expect(aggregate.hitCount == 1)
        #expect(aggregate.ssdHitCount == 1)
        #expect(aggregate.missCount == 1)
        #expect(aggregate.hitRate == 2.0 / 3.0)
        #expect(aggregate.tokenReuseRate == 230.0 / 350.0)
        #expect(aggregate.restoredCheckpointCounts["system"] == 1)
        #expect(aggregate.restoredCheckpointCounts["leaf"] == 1)
        #expect(aggregate.averageLookupMs == 12.5)
        #expect(aggregate.averageRestoreMs == 3.75)
        #expect(aggregate.averagePrefillMs == 44)
        #expect(aggregate.averageTTFTMs == 63)
        #expect(aggregate.admissionCount == 1)
        #expect(aggregate.admissionDropCount == 1)
        #expect(aggregate.evictionCount == 1)
    }

    @Test func aggregateCreditsRewindFromProductionRewrittenChainPrefixHit() {
        // In production a chain-prefix hit's reason is rewritten to "hit"
        // before the lookup event is logged, so the rewind signal rides the
        // explicit `chainPrefixRestore` field, not the reason. The floor
        // (snapshotOffset) sitting below the divergence (sharedPrefixLength) is
        // a served Think-Strip Rewind (issue #101). Before the fix this branch
        // keyed on `reason == "chainPrefixHit"`, which production never emits,
        // so the KPI stayed permanently 0.
        let events = [
            event(
                "lookup",
                fields: [
                    ("reason", "hit"),
                    ("hydratedFromSSD", "true"),
                    ("chainPrefixRestore", "true"),
                    ("sharedPrefixLength", "900"),
                    ("snapshotOffset", "640"),
                    ("promptTokens", "1000"),
                    ("skippedPrefillTokens", "640"),
                ]),
            // A restore that landed exactly at the divergence is a hit, not a
            // rewind — it must not inflate the count.
            event(
                "lookup",
                fields: [
                    ("reason", "hit"),
                    ("hydratedFromSSD", "true"),
                    ("chainPrefixRestore", "true"),
                    ("sharedPrefixLength", "512"),
                    ("snapshotOffset", "512"),
                    ("promptTokens", "600"),
                    ("skippedPrefillTokens", "512"),
                ]),
        ]

        let aggregate = PromptCacheTelemetryAggregate.from(events: events)

        #expect(aggregate.rewindEventCount == 1)
        #expect(aggregate.rewindTokens == 260)
        // Both are SSD-tier hits despite the reason being rewritten to "hit".
        #expect(aggregate.ssdHitCount == 2)
        #expect(aggregate.hitCount == 0)
    }

    @Test func recordForTestingBoundsEventAndSampleBuffers() {
        let store = PromptCacheTelemetryStore(registerDiagnosticsSink: false)
        let events = (0..<(PromptCacheTelemetryStore.maxEvents + 5)).map { index in
            event(
                "lookup",
                fields: [
                    ("index", "\(index)"),
                    ("reason", "missNoEntries"),
                    ("promptTokens", "1"),
                    ("skippedPrefillTokens", "0"),
                ])
        }

        store.recordForTesting(events)

        #expect(store.events.count == PromptCacheTelemetryStore.maxEvents)
        #expect(store.events.first?.field("index") == "5")
        #expect(store.aggregate.lookupCount == PromptCacheTelemetryStore.maxEvents)
        #expect(store.metricSamples.count == 1)
    }

    @Test func pauseBuffersDiagnosticsUntilResume() async {
        let store = PromptCacheTelemetryStore(registerDiagnosticsSink: true)
        let requestID = UUID()
        store.toggleLiveUpdates()

        PrefixCacheDiagnostics.forwardTelemetryEvent(
            event(
                "lookup", requestID: requestID,
                fields: [
                    ("reason", "hit"),
                    ("promptTokens", "32"),
                    ("skippedPrefillTokens", "16"),
                ]))

        try? await Task.sleep(nanoseconds: 50_000_000)
        #expect(events(in: store, matching: requestID).isEmpty)

        store.toggleLiveUpdates()
        try? await waitUntil { !events(in: store, matching: requestID).isEmpty }

        let matchingEvents = events(in: store, matching: requestID)
        #expect(matchingEvents.count == 1)
        #expect(PromptCacheTelemetryAggregate.from(events: matchingEvents).hitRate == 1)
    }

    @Test func exportJSONContainsSnapshotAggregateAndEvents() throws {
        let store = PromptCacheTelemetryStore(registerDiagnosticsSink: false)
        store.recordForTesting([
            event(
                "lookup",
                fields: [
                    ("reason", "hit"),
                    ("promptTokens", "10"),
                    ("skippedPrefillTokens", "7"),
                ])
        ])

        let data = try #require(store.exportJSONString().data(using: .utf8))
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let payload = try decoder.decode(PromptCacheExportPayload.self, from: data)

        #expect(payload.snapshot == nil)
        #expect(payload.events.count == 1)
        #expect(payload.aggregate.lookupCount == 1)
        #expect(payload.aggregate.hitRate == 1)
    }

    @Test func compactEventDisplaySummarizesRows() {
        let lookup = event(
            "lookup",
            fields: [
                ("reason", "ssdHit"),
                ("promptTokens", "128"),
                ("skippedPrefillTokens", "96"),
            ])
        let eviction = event(
            "eviction",
            fields: [
                ("freedBytes", "1048576")
            ])

        #expect(PromptCacheEventDisplay.symbol(for: lookup.eventName) == "magnifyingglass")
        #expect(PromptCacheEventDisplay.reason(for: lookup) == "ssdHit")
        #expect(PromptCacheEventDisplay.tokenSummary(lookup) == "96/128")
        #expect(PromptCacheEventDisplay.bytesSummary(eviction) == "1 MB")
        #expect(PromptCacheEventDisplay.requestSummary(lookup) == "11111111")
    }

    // MARK: - Tree filtering
    //
    // Fixture topology (offsets in parentheses):
    //
    //   root(0, empty)
    //    ├─ a(100, empty)
    //    │   └─ b(200, leaf, ramOnly)
    //    │       └─ c(300, empty)            ← empty tail
    //    └─ d(150, branchPoint, ramOnly)
    //        └─ e(400, ssdOnly)

    @Test func emptyNodesAreHiddenByDefaultAndSurvivorsReparent() throws {
        let store = storeWithFixtureTree()

        let tree = try #require(store.selectedTree)
        let ids = Set(tree.nodes.map(\.id))
        #expect(ids == ["root", "b", "d", "e"])

        // b's empty parent `a` is gone — b re-parents to root with an
        // edge spanning the hidden run.
        let b = tree.nodes.first { $0.id == "b" }
        #expect(b?.parentID == "root")
        let rootToB = tree.edges.first { $0.childID == "b" }
        #expect(rootToB?.parentID == "root")
        #expect(rootToB?.tokenCount == 200)

        // The empty tail `c` is simply gone.
        #expect(!tree.edges.contains { $0.childID == "c" })
    }

    @Test func hidingACheckpointTypeContractsPastIt() throws {
        let store = storeWithFixtureTree()
        store.visibleCheckpointTypes.remove("branchPoint")

        let tree = try #require(store.selectedTree)
        // Before the contraction fix, `d` was pulled back in as an
        // ancestor of `e`, making the toggle a no-op for interior nodes.
        #expect(!tree.nodes.contains { $0.id == "d" })
        let e = tree.nodes.first { $0.id == "e" }
        #expect(e?.parentID == "root")
        #expect(tree.edges.first { $0.childID == "e" }?.tokenCount == 400)
    }

    @Test func showingAllStorageStatesRestoresTheFullTopology() throws {
        let store = storeWithFixtureTree()
        store.visibleStorageStates = Set(PromptCacheStorageState.allCases)

        let tree = try #require(store.selectedTree)
        #expect(tree.nodes.count == 6)
        #expect(tree.nodes.first { $0.id == "b" }?.parentID == "a")
        // Direct edges keep their original spans.
        #expect(tree.edges.first { $0.childID == "b" }?.tokenCount == 100)
    }

    @Test func searchKeepsMatchesTheirVisibleAncestorsAndTheRoot() throws {
        let store = storeWithFixtureTree()
        store.searchText = "200"

        let tree = try #require(store.selectedTree)
        let ids = Set(tree.nodes.map(\.id))
        // b matches by offset; root anchors; d/e don't match and drop out.
        #expect(ids == ["root", "b"])
    }

    @Test func resetFiltersRestoresTheEmptyHiddenDefault() {
        let store = storeWithFixtureTree()
        store.visibleStorageStates = Set(PromptCacheStorageState.allCases)
        store.visibleCheckpointTypes = []
        store.searchText = "junk"

        store.resetFilters()

        #expect(store.searchText.isEmpty)
        #expect(!store.visibleStorageStates.contains(.empty))
        #expect(store.visibleCheckpointTypes == ["system", "leaf", "branchPoint"])
    }

    @Test func normalizeSelectionPreservesFirstAvailableAndDeselection() {
        let store = storeWithFixtureTree()
        // "First available" (nil) partition and no node selection must
        // survive a refresh — not snap to a concrete partition/root.
        #expect(store.selectedPartitionID == nil)
        #expect(store.selectedNodeID == nil)

        // A valid selection survives; a stale one is dropped.
        store.selectNode("b")
        store.replaceSnapshotForTesting(fixtureSnapshot())
        #expect(store.selectedNodeID == "b")
        store.selectNode("no-such-node")
        store.replaceSnapshotForTesting(fixtureSnapshot())
        #expect(store.selectedNodeID == nil)
    }

    private func storeWithFixtureTree() -> PromptCacheTelemetryStore {
        let store = PromptCacheTelemetryStore(registerDiagnosticsSink: false)
        store.replaceSnapshotForTesting(fixtureSnapshot())
        return store
    }

    private func fixtureSnapshot() -> PromptCacheTelemetrySnapshot {
        let nodes = [
            treeNode("root", parent: nil, offset: 0),
            treeNode("a", parent: "root", offset: 100),
            treeNode("b", parent: "a", offset: 200, storage: .ramOnly, checkpoint: "leaf"),
            treeNode("c", parent: "b", offset: 300),
            treeNode(
                "d", parent: "root", offset: 150, storage: .ramOnly, checkpoint: "branchPoint"),
            treeNode("e", parent: "d", offset: 400, storage: .ssdOnly),
        ]
        let nodesByID = Dictionary(uniqueKeysWithValues: nodes.map { ($0.id, $0) })
        let edges: [PromptCacheTreeEdgeSnapshot] = nodes.compactMap { node in
            guard let parentID = node.parentID, let parent = nodesByID[parentID] else { return nil }
            return PromptCacheTreeEdgeSnapshot(
                id: "\(parentID)->\(node.id)",
                parentID: parentID,
                childID: node.id,
                tokenCount: node.tokenOffset - parent.tokenOffset
            )
        }
        let tree = PromptCacheTreeSnapshot(
            id: "partition-0",
            partitionDigest: "partition-0",
            partitionSummary: "fixture",
            nodeCount: nodes.count,
            totalSnapshotBytes: 2048,
            snapshotCount: 2,
            snapshotsByType: ["leaf": 1, "branchPoint": 1],
            nodes: nodes,
            edges: edges
        )
        return PromptCacheTelemetrySnapshot(
            capturedAt: Date(timeIntervalSince1970: 100),
            memoryBudgetBytes: 0,
            budgetCeilingBytes: 0,
            budgetFloorBytes: 0,
            residentSnapshotBytes: 0,
            partitionCount: 1,
            totalNodeCount: nodes.count,
            snapshotCount: 2,
            snapshotsByType: ["leaf": 1, "branchPoint": 1],
            ssd: .disabled,
            tuner: .unavailable,
            counters: PromptCacheCumulativeCounters(),
            estimates: MeasuredSecondsEstimates(),
            trees: [tree]
        )
    }

    private func treeNode(
        _ id: String,
        parent: String?,
        offset: Int,
        storage: PromptCacheStorageState = .empty,
        checkpoint: String? = nil
    ) -> PromptCacheTreeNodeSnapshot {
        PromptCacheTreeNodeSnapshot(
            id: id,
            parentID: parent,
            pathHash: "hash-\(id)",
            tokenOffset: offset,
            pathTokenCount: offset,
            edgeTokenCount: 0,
            childCount: 0,
            depth: 0,
            hasSnapshot: storage == .ramOnly,
            checkpointType: checkpoint,
            snapshotBytes: storage == .ramOnly ? 1024 : 0,
            storageState: storage,
            snapshotRefID: nil,
            storageBytes: 0,
            lastAccessAgeSeconds: 0,
            normalizedRecency: nil,
            normalizedFlopEfficiency: nil,
            utility: nil
        )
    }

    private func event(
        _ name: String,
        requestID: UUID = UUID(uuidString: "11111111-1111-1111-1111-111111111111")!,
        fields: [(String, String)] = []
    ) -> PromptCacheTelemetryEvent {
        PromptCacheTelemetryEvent(
            timestamp: Date(timeIntervalSince1970: 100),
            scope: .request,
            eventName: name,
            requestID: requestID,
            modelID: "telemetry-test",
            kvBits: 8,
            kvGroupSize: 64,
            fields: fields
        )
    }

    private func events(
        in store: PromptCacheTelemetryStore,
        matching requestID: UUID
    ) -> [PromptCacheTelemetryEvent] {
        store.events.filter { $0.requestID == requestID }
    }

    private func waitUntil(
        timeout: Duration = .seconds(3),
        _ condition: @MainActor @Sendable () -> Bool
    ) async throws {
        let deadline = ContinuousClock.now + timeout
        while !condition() {
            try await Task.sleep(for: .milliseconds(10))
            if ContinuousClock.now >= deadline {
                Issue.record("waitUntil timed out after \(timeout)")
                return
            }
        }
    }
}
