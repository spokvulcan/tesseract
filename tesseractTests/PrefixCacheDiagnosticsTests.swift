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
            utility: 1.0
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
            utility: nil
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
}
