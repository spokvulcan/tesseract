import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct PromptCacheTelemetryStoreTests {

    @Test func aggregateComputesHitRateTokenReuseAndLatency() {
        let events = [
            event("lookup", fields: [
                ("reason", "hit"),
                ("promptTokens", "100"),
                ("skippedPrefillTokens", "80"),
                ("checkpointType", "system"),
                ("lookupMs", "10"),
                ("restoreMs", "3"),
            ]),
            event("lookup", fields: [
                ("reason", "ssdHit"),
                ("promptTokens", "200"),
                ("skippedPrefillTokens", "150"),
                ("checkpointType", "leaf"),
                ("lookupMs", "20"),
                ("restoreMs", "8"),
            ]),
            event("lookup", fields: [
                ("reason", "missNoSnapshotInPrefix"),
                ("promptTokens", "50"),
                ("skippedPrefillTokens", "0"),
                ("checkpointType", "nil"),
                ("lookupMs", "5"),
                ("restoreMs", "0"),
            ]),
            event("ttft", fields: [
                ("lookupMs", "15"),
                ("restoreMs", "4"),
                ("prefillMs", "44"),
                ("totalPromptMs", "63"),
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

    @Test func recordForTestingBoundsEventAndSampleBuffers() {
        let store = PromptCacheTelemetryStore(registerDiagnosticsSink: false)
        let events = (0..<(PromptCacheTelemetryStore.maxEvents + 5)).map { index in
            event("lookup", fields: [
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
        store.toggleLiveUpdates()

        PrefixCacheDiagnostics.forwardTelemetryEvent(event("lookup", fields: [
            ("reason", "hit"),
            ("promptTokens", "32"),
            ("skippedPrefillTokens", "16"),
        ]))

        try? await Task.sleep(nanoseconds: 50_000_000)
        #expect(store.events.isEmpty)

        store.toggleLiveUpdates()
        try? await Task.sleep(nanoseconds: 200_000_000)

        #expect(store.events.count == 1)
        #expect(store.aggregate.hitRate == 1)
    }

    @Test func exportJSONContainsSnapshotAggregateAndEvents() throws {
        let store = PromptCacheTelemetryStore(registerDiagnosticsSink: false)
        store.recordForTesting([
            event("lookup", fields: [
                ("reason", "hit"),
                ("promptTokens", "10"),
                ("skippedPrefillTokens", "7"),
            ]),
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
        let lookup = event("lookup", fields: [
            ("reason", "ssdHit"),
            ("promptTokens", "128"),
            ("skippedPrefillTokens", "96"),
        ])
        let eviction = event("eviction", fields: [
            ("freedBytes", "1048576"),
        ])

        #expect(PromptCacheEventDisplay.symbol(for: lookup.eventName) == "magnifyingglass")
        #expect(PromptCacheEventDisplay.reason(for: lookup) == "ssdHit")
        #expect(PromptCacheEventDisplay.tokenSummary(lookup) == "96/128")
        #expect(PromptCacheEventDisplay.bytesSummary(eviction) == "1 MB")
        #expect(PromptCacheEventDisplay.requestSummary(lookup) == "11111111")
    }

    private func event(
        _ name: String,
        fields: [(String, String)] = []
    ) -> PromptCacheTelemetryEvent {
        PromptCacheTelemetryEvent(
            timestamp: Date(timeIntervalSince1970: 100),
            scope: .request,
            eventName: name,
            requestID: UUID(uuidString: "11111111-1111-1111-1111-111111111111")!,
            modelID: "telemetry-test",
            kvBits: 8,
            kvGroupSize: 64,
            fields: fields
        )
    }
}
