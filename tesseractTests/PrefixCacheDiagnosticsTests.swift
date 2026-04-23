import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

struct PrefixCacheDiagnosticsTests {

    private let context = PrefixCacheDiagnostics.Context(
        requestID: UUID(uuidString: "00000000-0000-0000-0000-000000000001")!,
        modelID: "qwen3.5",
        kvBits: 8,
        kvGroupSize: 64
    )

    @Test func lookupHitRendersDeterministically() {
        let event = PrefixCacheDiagnostics.LookupEvent(
            reason: .hit(snapshotOffset: 768, totalTokens: 1024, type: .system),
            promptTokens: 1024,
            sharedPrefixLength: 900,
            skippedPrefillTokens: 768,
            newTokensToPrefill: 256,
            lookupMs: 0.012,
            restoreMs: 0.003,
            plannedCheckpoints: [(offset: 900, type: .branchPoint)]
        )

        #expect(context.render(event) ==
            "event=lookup requestID=00000000-0000-0000-0000-000000000001 modelID=qwen3.5 kvBits=8 kvGroupSize=64 reason=hit promptTokens=1024 sharedPrefixLength=900 snapshotOffset=768 checkpointType=system skippedPrefillTokens=768 newTokensToPrefill=256 lookupMs=12.000 restoreMs=3.000 plannedCheckpoints=[900:branchPoint]")
    }

    @Test func lookupMissRendersTreeDepthAndNilSnapshotFields() {
        let event = PrefixCacheDiagnostics.LookupEvent(
            reason: .missNoSnapshotInPrefix,
            promptTokens: 700,
            sharedPrefixLength: 128,
            skippedPrefillTokens: 0,
            newTokensToPrefill: 700,
            lookupMs: 0.004,
            restoreMs: 0,
            plannedCheckpoints: []
        )

        #expect(context.render(event) ==
            "event=lookup requestID=00000000-0000-0000-0000-000000000001 modelID=qwen3.5 kvBits=8 kvGroupSize=64 reason=missNoSnapshotInPrefix promptTokens=700 sharedPrefixLength=128 snapshotOffset=nil checkpointType=nil skippedPrefillTokens=0 newTokensToPrefill=700 lookupMs=4.000 restoreMs=0.000 plannedCheckpoints=[]")
    }

    @Test func captureDistinguishesPrefillAndLeafSources() {
        let prefill = PrefixCacheDiagnostics.CaptureEvent(
            offset: 512,
            checkpointType: .system,
            bytes: 2048,
            duringPrefill: true,
            source: "prefill"
        )
        let leaf = PrefixCacheDiagnostics.CaptureEvent(
            offset: 1024,
            checkpointType: .leaf,
            bytes: 4096,
            duringPrefill: false,
            source: "leaf"
        )

        #expect(context.render(prefill).contains("duringPrefill=true source=prefill"))
        #expect(context.render(leaf).contains("duringPrefill=false source=leaf"))
    }

    @Test func leafModeRendersModeAndContinuation() {
        let event = PrefixCacheDiagnostics.LeafModeEvent(
            mode: HTTPLeafStoreMode.directToolLeaf.rawValue,
            continuation: HTTPLeafContinuationKind.toolResult.rawValue
        )

        #expect(context.render(event) ==
            "event=leafMode requestID=00000000-0000-0000-0000-000000000001 modelID=qwen3.5 kvBits=8 kvGroupSize=64 mode=directToolLeaf continuation=toolResult")
    }

    @Test func evictionRendersUtilityScoresOnlyWhenPresent() {
        let utility = PrefixCacheDiagnostics.EvictionEvent(.init(
            strategy: .utility,
            offset: 512,
            checkpointType: .leaf,
            freedBytes: 4096,
            budgetBytes: 2048,
            snapshotBytesAfter: 1024,
            normalizedRecency: 0.25,
            normalizedFlopEfficiency: 0.75,
            utility: 1.0,
            bodyDroppedStorageRefID: nil
        ))
        let fallback = PrefixCacheDiagnostics.EvictionEvent(.init(
            strategy: .fallback,
            offset: 256,
            checkpointType: .system,
            freedBytes: 2048,
            budgetBytes: 0,
            snapshotBytesAfter: 0,
            normalizedRecency: nil,
            normalizedFlopEfficiency: nil,
            utility: nil,
            bodyDroppedStorageRefID: nil
        ))

        let utilityLine = context.render(utility)
        let fallbackLine = context.render(fallback)

        #expect(utilityLine.contains("normalizedRecency=0.250000"))
        #expect(utilityLine.contains("normalizedFlopEfficiency=0.750000"))
        #expect(utilityLine.contains("utility=1.000000"))
        #expect(!fallbackLine.contains("normalizedRecency="))
        #expect(!fallbackLine.contains("normalizedFlopEfficiency="))
        #expect(!fallbackLine.contains("utility="))
    }

    @Test func ttftClampsNegativeFirstTokenTimeToZero() {
        let event = PrefixCacheDiagnostics.TTFTEvent(
            lookupMs: 0.001,
            restoreMs: 0.002,
            prefillMs: 0.020,
            totalPromptMs: 0.015
        )

        #expect(context.render(event) ==
            "event=ttft requestID=00000000-0000-0000-0000-000000000001 modelID=qwen3.5 kvBits=8 kvGroupSize=64 lookupMs=1.000 restoreMs=2.000 prefillMs=20.000 firstTokenMs=0.000 totalPromptMs=15.000")
    }

    @Test func memoryRendersCacheAndMlxCounters() {
        let stats = PrefixCacheManager.CacheStats(
            partitionCount: 2,
            totalNodeCount: 5,
            totalSnapshotBytes: 8192,
            snapshotsByType: [.system: 1, .leaf: 2, .branchPoint: 0]
        )
        let event = PrefixCacheDiagnostics.MemoryEvent(
            stats: stats,
            budgetBytes: 16384,
            modelWeightBytes: 123456,
            activeMlxBytes: 111,
            peakMlxBytes: 222,
            mlxCacheLimitBytes: 333
        )

        #expect(context.render(event) ==
            "event=memory requestID=00000000-0000-0000-0000-000000000001 modelID=qwen3.5 kvBits=8 kvGroupSize=64 snapshotCount=3 totalSnapshotBytes=8192 budgetBytes=16384 modelWeightBytes=123456 activeMlxBytes=111 peakMlxBytes=222 mlxCacheLimitBytes=333 partitionCount=2")
    }

    // MARK: - SSD-tier event renderers (Task 4.1.12)

    @Test func ssdAdmitAcceptedRendersInSystemFormat() {
        let event = PrefixCacheDiagnostics.SSDAdmitEvent(
            id: "snap-1",
            bytes: 4_096,
            outcome: .accepted
        )

        #expect(PrefixCacheDiagnostics.renderSystem(event) ==
            "event=ssdAdmit id=snap-1 bytes=4096 outcome=accepted")
    }

    @Test func ssdAdmitDroppedSystemProtectionWinsRendersOutcome() {
        let event = PrefixCacheDiagnostics.SSDAdmitEvent(
            id: "snap-2",
            bytes: 8_192,
            outcome: .droppedSystemProtectionWins
        )

        #expect(PrefixCacheDiagnostics.renderSystem(event) ==
            "event=ssdAdmit id=snap-2 bytes=8192 outcome=droppedSystemProtectionWins")
    }

    @Test func ssdEvictAtAdmissionPairsVictimAndIncoming() {
        let event = PrefixCacheDiagnostics.SSDEvictAtAdmissionEvent(
            victimID: "old-1",
            incomingID: "new-1"
        )

        #expect(PrefixCacheDiagnostics.renderSystem(event) ==
            "event=ssdEvictAtAdmission victimID=old-1 incomingID=new-1")
    }

    @Test func ssdHitRendersHydrateMs() {
        let event = PrefixCacheDiagnostics.SSDHitEvent(
            id: "snap-3",
            hydrateMs: 0.045
        )

        #expect(PrefixCacheDiagnostics.renderSystem(event) ==
            "event=ssdHit id=snap-3 hydrateMs=45.000")
    }

    @Test func ssdMissCarriesGranularReason() {
        let event = PrefixCacheDiagnostics.SSDMissEvent(
            id: "snap-4",
            reason: .fingerprintMismatch
        )

        #expect(PrefixCacheDiagnostics.renderSystem(event) ==
            "event=ssdMiss id=snap-4 reason=fingerprintMismatch")
    }

    @Test func ssdBodyDropCarriesIDOnly() {
        let event = PrefixCacheDiagnostics.SSDBodyDropEvent(id: "snap-5")

        #expect(PrefixCacheDiagnostics.renderSystem(event) ==
            "event=ssdBodyDrop id=snap-5")
    }

    @Test func leafSupersessionCarriesOffsetAndOptionalStorageRefID() {
        let withRef = PrefixCacheDiagnostics.LeafSupersessionEvent(
            offset: 512,
            storageRefID: "snap-old"
        )
        let withoutRef = PrefixCacheDiagnostics.LeafSupersessionEvent(
            offset: 256,
            storageRefID: nil
        )

        #expect(context.render(withRef) ==
            "event=leafSupersession requestID=00000000-0000-0000-0000-000000000001 modelID=qwen3.5 kvBits=8 kvGroupSize=64 offset=512 storageRefID=snap-old")
        #expect(context.render(withoutRef) ==
            "event=leafSupersession requestID=00000000-0000-0000-0000-000000000001 modelID=qwen3.5 kvBits=8 kvGroupSize=64 offset=256 storageRefID=nil")
    }

    @Test func ssdRecordHitCarriesIDOnly() {
        let event = PrefixCacheDiagnostics.SSDRecordHitEvent(id: "snap-6")

        #expect(PrefixCacheDiagnostics.renderSystem(event) ==
            "event=ssdRecordHit id=snap-6")
    }

    @Test func storageRefCommitCarriesIDOnly() {
        let event = PrefixCacheDiagnostics.StorageRefCommitEvent(id: "snap-7")

        #expect(PrefixCacheDiagnostics.renderSystem(event) ==
            "event=storageRefCommit id=snap-7")
    }

    @Test func storageRefDropCallbackCarriesReason() {
        for reason: SSDDropReason in [
            .backpressureOldest,
            .evictedByLRU,
            .systemProtectionWins,
            .exceedsBudget,
            .diskFull,
            .writerIOError,
            .hydrationFailure,
        ] {
            let event = PrefixCacheDiagnostics.StorageRefDropCallbackEvent(
                id: "snap-8",
                reason: reason
            )
            let line = PrefixCacheDiagnostics.renderSystem(event)
            #expect(
                line == "event=storageRefDropCallback id=snap-8 reason=\(PrefixCacheDiagnostics.ssdDropReasonString(reason))"
            )
        }
    }

    @Test func warmStartCompleteRendersStructuredFields() {
        let event = PrefixCacheDiagnostics.WarmStartCompleteEvent(
            partitionCount: 3,
            snapshotCount: 17,
            invalidatedPartitionCount: 1,
            durationSeconds: 0.012
        )

        #expect(PrefixCacheDiagnostics.renderSystem(event) ==
            "event=warmStartComplete partitionCount=3 snapshotCount=17 invalidatedPartitionCount=1 durationMs=12.000")
    }

    @Test func fingerprintMismatchCarriesPartitionDigest() {
        let event = PrefixCacheDiagnostics.FingerprintMismatchEvent(partition: "abcd1234")

        #expect(PrefixCacheDiagnostics.renderSystem(event) ==
            "event=fingerprintMismatch partition=abcd1234")
    }

    @Test func testSinkCapturesContextfulAndSystemEvents() {
        // Install a recording sink, emit one event from each path,
        // then drain. The sink must see both lines; uninstalling
        // clears it for subsequent tests. Tests run in parallel and
        // share the registry, so we filter by a per-test unique ID
        // (`diag-snap-9` / `diag-snap-10`) to ignore cross-test
        // leakage from sibling tests.
        let sink = RecordingSink()
        let handle = PrefixCacheDiagnostics.addTestSink(sink.handler)
        defer { PrefixCacheDiagnostics.removeTestSink(handle) }

        context.log(PrefixCacheDiagnostics.SSDRecordHitEvent(id: "diag-snap-9"))
        PrefixCacheDiagnostics.logSystem(
            PrefixCacheDiagnostics.StorageRefCommitEvent(id: "diag-snap-10")
        )

        let mine = sink.drain().filter {
            $0.contains("diag-snap-9") || $0.contains("diag-snap-10")
        }
        #expect(mine.count == 2)

        let recordHit = mine.first { $0.contains("event=ssdRecordHit") }
        #expect(recordHit != nil)
        #expect(recordHit?.contains("requestID=00000000-0000-0000-0000-000000000001") == true)

        let commit = mine.first { $0.contains("event=storageRefCommit") }
        #expect(commit == "event=storageRefCommit id=diag-snap-10")
    }

    @Test func telemetrySinkCapturesStructuredFieldsWithoutChangingRenderedLogs() {
        let sink = TelemetryRecordingSink()
        let handle = PrefixCacheDiagnostics.addTelemetrySink(sink.handler)
        defer { PrefixCacheDiagnostics.removeTelemetrySink(handle) }

        let lookup = PrefixCacheDiagnostics.LookupEvent(
            reason: .hit(snapshotOffset: 768, totalTokens: 1024, type: .system),
            promptTokens: 1024,
            sharedPrefixLength: 900,
            skippedPrefillTokens: 768,
            newTokensToPrefill: 256,
            lookupMs: 0.012,
            restoreMs: 0.003,
            plannedCheckpoints: [(offset: 900, type: .branchPoint)]
        )
        let rendered = context.render(lookup)
        #expect(rendered ==
            "event=lookup requestID=00000000-0000-0000-0000-000000000001 modelID=qwen3.5 kvBits=8 kvGroupSize=64 reason=hit promptTokens=1024 sharedPrefixLength=900 snapshotOffset=768 checkpointType=system skippedPrefillTokens=768 newTokensToPrefill=256 lookupMs=12.000 restoreMs=3.000 plannedCheckpoints=[900:branchPoint]")

        let runID = UUID()
        let localContext = PrefixCacheDiagnostics.Context(
            requestID: runID,
            modelID: "qwen3.5",
            kvBits: 8,
            kvGroupSize: 64
        )
        let systemID = "telemetry-\(runID.uuidString)"

        localContext.log(lookup)
        PrefixCacheDiagnostics.logSystem(
            PrefixCacheDiagnostics.SSDAdmitEvent(
                id: systemID,
                bytes: 4096,
                outcome: .accepted
            )
        )

        let mine = sink.drain().filter {
            $0.requestID == runID || $0.field("id") == systemID
        }
        #expect(mine.count == 2)

        let structuredLookup = mine.first { $0.eventName == "lookup" }
        #expect(structuredLookup?.scope == .request)
        #expect(structuredLookup?.requestID == runID)
        #expect(structuredLookup?.modelID == "qwen3.5")
        #expect(structuredLookup?.kvBits == 8)
        #expect(structuredLookup?.kvGroupSize == 64)
        #expect(structuredLookup?.field("reason") == "hit")
        #expect(structuredLookup?.field("plannedCheckpoints") == "[900:branchPoint]")

        let structuredSystem = mine.first { $0.eventName == "ssdAdmit" }
        #expect(structuredSystem?.scope == .system)
        #expect(structuredSystem?.requestID == nil)
        #expect(structuredSystem?.field("id") == systemID)
        #expect(structuredSystem?.field("outcome") == "accepted")
    }

    private final class RecordingSink: @unchecked Sendable {
        private let lock = NSLock()
        private var lines: [String] = []

        var handler: @Sendable (String) -> Void {
            { [weak self] line in
                guard let self else { return }
                self.lock.lock()
                self.lines.append(line)
                self.lock.unlock()
            }
        }

        func drain() -> [String] {
            lock.lock()
            defer { lock.unlock() }
            let copy = lines
            lines.removeAll()
            return copy
        }
    }

    private final class TelemetryRecordingSink: @unchecked Sendable {
        private let lock = NSLock()
        private var events: [PromptCacheTelemetryEvent] = []

        var handler: @Sendable (PromptCacheTelemetryEvent) -> Void {
            { [weak self] event in
                guard let self else { return }
                self.lock.lock()
                self.events.append(event)
                self.lock.unlock()
            }
        }

        func drain() -> [PromptCacheTelemetryEvent] {
            lock.lock()
            defer { lock.unlock() }
            let copy = events
            events.removeAll()
            return copy
        }
    }
}
