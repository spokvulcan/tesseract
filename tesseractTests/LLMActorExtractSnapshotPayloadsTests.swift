//
//  LLMActorExtractSnapshotPayloadsTests.swift
//  tesseractTests
//
//  Unit tests for `LLMActor.extractSnapshotPayloads` — the helper that
//  converts a set of live Metal-resident `HybridCacheSnapshot`
//  instances into pure `Sendable` `SnapshotPayload` value types via
//  `MLXArray.asData()`. Also covers the downstream plumbing contract
//  on `PrefixCacheManager.storeSnapshots` / `storeLeaf`.
//
//  The helper itself is `nonisolated static` and operates on pure
//  fixture snapshots, so none of these tests require a loaded MLX
//  model.
//

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

struct LLMActorExtractSnapshotPayloadsTests {

    // MARK: - Fixture builders

    /// Build a single-layer `KVCacheSimple` snapshot whose arrays have
    /// deterministic content so round-trip equality is byte-exact.
    private func makeSimpleKVSnapshot(
        tokenOffset: Int = 5,
        type: HybridCacheSnapshot.CheckpointType = .system
    ) -> HybridCacheSnapshot {
        let kv = KVCacheSimple()
        // `.ones` picks a non-zero value so a failed asData() call
        // (returning an empty `Data`) fails loudly on a size check.
        kv.state = [
            MLXArray.ones([1, 2, 4, 8]),
            MLXArray.ones([1, 2, 4, 8]),
        ]
        return HybridCacheSnapshot.capture(
            cache: [kv], offset: tokenOffset, type: type
        )!
    }

    /// Build a mixed-layer snapshot (KV + Mamba + Quantized) to
    /// exercise the per-layer `className` + `metaState` plumbing.
    private func makeMixedSnapshot(
        tokenOffset: Int = 32
    ) -> HybridCacheSnapshot {
        let kv = KVCacheSimple()
        kv.state = [
            MLXArray.zeros([1, 1, 4, 64]),
            MLXArray.zeros([1, 1, 4, 64]),
        ]

        let mamba = MambaCache()
        mamba.state = [
            MLXArray.zeros([1, 3, 128]),
            MLXArray.zeros([1, 8, 16, 32]),
        ]

        let quantized = QuantizedKVCache(groupSize: 64, bits: 8)

        return HybridCacheSnapshot.capture(
            cache: [kv, mamba, quantized], offset: tokenOffset, type: .leaf
        )!
    }

    // MARK: - SSD gate

    @Test
    func returnsEmptyWhenSSDDisabled() {
        let snapshot = makeSimpleKVSnapshot()
        let payloads = LLMActor.extractSnapshotPayloads(
            [snapshot], ssdEnabled: false
        )
        #expect(payloads.isEmpty)
    }

    @Test
    func returnsEmptyWhenSSDDisabledForMultipleSnapshots() {
        // The gate must short-circuit regardless of input size — a
        // non-empty input array with SSD disabled still yields [].
        let snapshots = [
            makeSimpleKVSnapshot(tokenOffset: 5),
            makeSimpleKVSnapshot(tokenOffset: 9),
            makeMixedSnapshot(tokenOffset: 40),
        ]
        let payloads = LLMActor.extractSnapshotPayloads(
            snapshots, ssdEnabled: false
        )
        #expect(payloads.isEmpty)
    }

    @Test
    func returnsEmptyForEmptyInputEvenWhenEnabled() {
        let payloads = LLMActor.extractSnapshotPayloads(
            [], ssdEnabled: true
        )
        #expect(payloads.isEmpty)
    }

    // MARK: - Structural equivalence

    @Test
    func preservesTokenOffsetAndCheckpointType() throws {
        let snapshot = makeSimpleKVSnapshot(tokenOffset: 17, type: .branchPoint)
        let payloads = LLMActor.extractSnapshotPayloads(
            [snapshot], ssdEnabled: true
        )
        try #require(payloads.count == 1)
        let payload = payloads[0]
        #expect(payload.tokenOffset == 17)
        #expect(payload.checkpointType == .branchPoint)
    }

    @Test
    func preservesPerLayerMetadata() throws {
        let snapshot = makeMixedSnapshot(tokenOffset: 64)
        let payloads = LLMActor.extractSnapshotPayloads(
            [snapshot], ssdEnabled: true
        )
        try #require(payloads.count == 1)
        let payload = payloads[0]

        #expect(payload.layers.count == snapshot.layers.count)
        for (layerIdx, layer) in payload.layers.enumerated() {
            let source = snapshot.layers[layerIdx]
            #expect(layer.className == source.className)
            #expect(layer.metaState == source.metaState)
            #expect(layer.offset == source.offset)
            #expect(layer.state.count == source.state.count)
        }
    }

    @Test
    func payloadIsPositionallyAlignedWithInput() throws {
        let snapshots = [
            makeSimpleKVSnapshot(tokenOffset: 5),
            makeSimpleKVSnapshot(tokenOffset: 9),
            makeSimpleKVSnapshot(tokenOffset: 13),
        ]
        let payloads = LLMActor.extractSnapshotPayloads(
            snapshots, ssdEnabled: true
        )
        try #require(payloads.count == snapshots.count)
        for (i, payload) in payloads.enumerated() {
            #expect(payload.tokenOffset == snapshots[i].tokenOffset)
            #expect(payload.checkpointType == snapshots[i].checkpointType)
        }
    }

    // MARK: - Byte round-trip

    @Test
    func byteRoundTripMatchesSourceArrays() throws {
        let snapshot = makeSimpleKVSnapshot()
        let payloads = LLMActor.extractSnapshotPayloads(
            [snapshot], ssdEnabled: true
        )
        try #require(payloads.count == 1)
        let payload = payloads[0]

        // The per-layer state arrays must serialize to the same bytes
        // as the snapshot's own MLX-resident deep copies. This is the
        // core correctness assertion — a downstream `savePromptCache`
        // call on either the live snapshot or the extracted payload
        // must produce byte-identical files. `MLXArray.ones` defaults
        // to `.float32` at `Vendor/.../mlx-swift/Source/MLX/Factory.swift:115`,
        // so the dtype literal is pinned to match — a vendor default
        // change would surface as a test failure at this line, not as
        // a silent wire-format drift.
        for (layerIdx, layer) in payload.layers.enumerated() {
            let source = snapshot.layers[layerIdx]
            for (arrayIdx, payloadArray) in layer.state.enumerated() {
                let reference = source.state[arrayIdx].asData()
                #expect(payloadArray.data == reference.data,
                        "layer \(layerIdx) array \(arrayIdx) bytes mismatch")
                #expect(payloadArray.shape == reference.shape,
                        "layer \(layerIdx) array \(arrayIdx) shape mismatch")
                #expect(payloadArray.dtype == "float32",
                        "layer \(layerIdx) array \(arrayIdx) dtype — expected the pinned wire-format literal \"float32\"")
            }
        }
    }

    /// Literal pinning of the SSD on-disk wire-format contract. These
    /// strings are written verbatim into the snapshot header at
    /// `SSDSnapshotStore.encodePlaceholderContainer`, so a typo, a
    /// case rename, or any divergence from this table silently
    /// corrupts cache files that a future reader would reject.
    /// Assertions are literal — NOT sourced from
    /// `LLMActor.dtypeWireString` — so self-consistency cannot mask a
    /// contract drift. If a new `DType` case is added, extend this
    /// table in lockstep with the production helper.
    @Test
    func dtypeWireStringsArePinned() {
        #expect(LLMActor.dtypeWireString(.bool) == "bool")
        #expect(LLMActor.dtypeWireString(.uint8) == "uint8")
        #expect(LLMActor.dtypeWireString(.uint16) == "uint16")
        #expect(LLMActor.dtypeWireString(.uint32) == "uint32")
        #expect(LLMActor.dtypeWireString(.uint64) == "uint64")
        #expect(LLMActor.dtypeWireString(.int8) == "int8")
        #expect(LLMActor.dtypeWireString(.int16) == "int16")
        #expect(LLMActor.dtypeWireString(.int32) == "int32")
        #expect(LLMActor.dtypeWireString(.int64) == "int64")
        #expect(LLMActor.dtypeWireString(.float16) == "float16")
        #expect(LLMActor.dtypeWireString(.float32) == "float32")
        #expect(LLMActor.dtypeWireString(.bfloat16) == "bfloat16")
        #expect(LLMActor.dtypeWireString(.complex64) == "complex64")
        #expect(LLMActor.dtypeWireString(.float64) == "float64")
    }

    @Test
    func totalBytesMatchesSumOfArrayNbytes() throws {
        let snapshot = makeMixedSnapshot()
        let payloads = LLMActor.extractSnapshotPayloads(
            [snapshot], ssdEnabled: true
        )
        try #require(payloads.count == 1)
        let payload = payloads[0]

        // `SnapshotPayload.totalBytes` is the SSD front-door's byte
        // accounting. For arrays materialized via `asData(access: .copy)`
        // that value must equal the sum of each source MLXArray's
        // `nbytes`, which is the same byte count the radix-tree
        // eviction path uses.
        let expected = snapshot.layers.reduce(0) { acc, layer in
            acc + layer.state.reduce(0) { $0 + $1.nbytes }
        }
        #expect(payload.totalBytes == expected)
    }

    // MARK: - PrefixCacheManager signature plumbing

    @MainActor
    @Test
    func storeSnapshotsAcceptsNewPayloadsParameter() {
        // Accepting the new `snapshotPayloads` arg must not break the
        // RAM-only behavior — the manager still stores the snapshot,
        // returns `StoreDiagnostics`, and applies eviction pressure.
        let manager = PrefixCacheManager(
            memoryBudgetBytes: 16 * 1024 * 1024
        )
        let snapshot = makeSimpleKVSnapshot(tokenOffset: 4)
        let payloads = LLMActor.extractSnapshotPayloads(
            [snapshot], ssdEnabled: true
        )
        let partitionKey = CachePartitionKey(
            modelID: "test-model", kvBits: nil, kvGroupSize: 64
        )
        let diagnostics = manager.storeSnapshots(
            promptTokens: [1, 2, 3, 4],
            capturedSnapshots: [snapshot],
            snapshotPayloads: payloads,
            partitionKey: partitionKey,
            requestID: UUID()
        )
        #expect(diagnostics.evictions.isEmpty)
        #expect(manager.stats.snapshotCount == 1)
    }

    @MainActor
    @Test
    func storeLeafAcceptsNewPayloadParameter() {
        let manager = PrefixCacheManager(
            memoryBudgetBytes: 16 * 1024 * 1024
        )
        let snapshot = makeSimpleKVSnapshot(tokenOffset: 6)
        let payload = LLMActor.extractSnapshotPayloads(
            [snapshot], ssdEnabled: true
        ).first
        let partitionKey = CachePartitionKey(
            modelID: "test-model", kvBits: nil, kvGroupSize: 64
        )
        let diagnostics = manager.storeLeaf(
            storedTokens: [10, 20, 30, 40, 50, 60],
            leafSnapshot: snapshot,
            leafPayload: payload,
            partitionKey: partitionKey,
            requestID: UUID()
        )
        #expect(diagnostics.evictions.isEmpty)
        #expect(manager.stats.snapshotCount == 1)
    }

    @MainActor
    @Test
    func storeSnapshotsToleratesEmptyPayloadsForRAMOnlyPath() {
        // When `ssdConfig?.enabled` is off, the LLMActor helper returns
        // `[]` for `snapshotPayloads`. The manager must accept that
        // zero-length array without mis-attributing it as a "no
        // snapshots to store" signal — the snapshots still need to
        // land in the radix tree.
        let manager = PrefixCacheManager(
            memoryBudgetBytes: 16 * 1024 * 1024
        )
        let snapshot = makeSimpleKVSnapshot(tokenOffset: 5)
        let partitionKey = CachePartitionKey(
            modelID: "test-model", kvBits: nil, kvGroupSize: 64
        )
        let diagnostics = manager.storeSnapshots(
            promptTokens: [7, 8, 9, 10, 11],
            capturedSnapshots: [snapshot],
            snapshotPayloads: [],
            partitionKey: partitionKey,
            requestID: UUID()
        )
        #expect(diagnostics.evictions.isEmpty)
        #expect(manager.stats.snapshotCount == 1)
    }

    // MARK: - Call-site wiring regression coverage
    //
    // The three asymmetric LLMActor call sites (mid-prefill,
    // unstripped leaf, stripped leaf) cannot be exercised by unit
    // tests without a loaded MLX model — the full wiring is gated
    // behind `container.perform` on a real `ModelContainer`. These
    // source-grep tests compensate by pinning the literal wiring
    // strings, so a refactor that drops any of the four load-bearing
    // lines fails here with a clear pointer at the missing site.
    // Mirrors the `threadAffinityContractDocCommentIsPinned` pattern
    // at `HybridCacheSnapshotTests.swift:539`.

    private func readLLMActorSource() throws -> String {
        let testFile = URL(fileURLWithPath: #filePath)
        let projectRoot = testFile
            .deletingLastPathComponent()   // tesseractTests
            .deletingLastPathComponent()   // project root
        let sourceFile = projectRoot
            .appendingPathComponent("tesseract")
            .appendingPathComponent("Features")
            .appendingPathComponent("Agent")
            .appendingPathComponent("LLMActor.swift")
        return try String(contentsOf: sourceFile, encoding: .utf8)
    }

    @Test
    func makeHTTPPrefixCacheGenerationPopulatesCapturedPayloads() throws {
        // `HTTPPrefixCacheGeneration` is constructed at the tail of
        // `makeHTTPPrefixCacheGeneration` with `capturedPayloads:
        // capturedPayloads` — where the local `capturedPayloads` was
        // produced by `Self.extractSnapshotPayloads(capturedSnapshots,
        // ssdEnabled: ssdEnabled)` inside the enclosing
        // `container.perform`. Dropping either of those two lines
        // would leave the field silently empty while the build and
        // the helper's own unit tests still pass.
        let source = try readLLMActorSource()

        let extractCall = "Self.extractSnapshotPayloads(\n                capturedSnapshots,\n                ssdEnabled: ssdEnabled\n            )"
        let structInit = "capturedPayloads: capturedPayloads,"

        #expect(
            source.contains(extractCall),
            "makeHTTPPrefixCacheGeneration must call Self.extractSnapshotPayloads(capturedSnapshots, ssdEnabled:) inside its container.perform block"
        )
        #expect(
            source.contains(structInit),
            "HTTPPrefixCacheGeneration must be constructed with capturedPayloads: capturedPayloads"
        )
    }

    @Test
    func midPrefillCallSiteForwardsCapturedPayloadsToStoreSnapshots() throws {
        // The `prefixCache.storeSnapshots(...)` call inside the post-
        // generation MainActor hop must pass `snapshotPayloads:
        // mlxStart.capturedPayloads`. A refactor that drops this arg
        // falls back to the `= []` default and silently disables SSD
        // persistence for every mid-prefill snapshot.
        let source = try readLLMActorSource()
        #expect(
            source.contains("snapshotPayloads: mlxStart.capturedPayloads"),
            "Mid-prefill storeSnapshots call must forward mlxStart.capturedPayloads"
        )
    }

    @Test
    func unstrippedLeafCallSiteForwardsLeafPayloadToStoreLeaf() throws {
        // The unstripped leaf `storeLeaf(...)` call must pass
        // `leafPayload: leafPayload`, where `leafPayload` is the
        // locally-bound optional produced alongside the leaf capture
        // inside the `container.perform(nonSendable: finalCache)`
        // block. Regression guard against accidentally dropping the
        // arg or inlining `nil`.
        let source = try readLLMActorSource()
        #expect(
            source.contains("leafPayload: leafPayload"),
            "Unstripped leaf storeLeaf call must forward the locally-bound leafPayload"
        )
    }

    @Test
    func structuredLeafHelperForwardsSharedLeafPayloadToStoreLeaf() throws {
        // The tool-loop direct leaf and canonical user leaf now share
        // one `captureStructuredLeafFromBoundary(...)` helper. That
        // helper must still extract a local `leafPayload` and forward
        // it into `prefixCache.storeLeaf(...)`; the older dedicated
        // `strippedLeafPayload` path no longer exists under the
        // single-leaf policy.
        let source = try readLLMActorSource()
        #expect(
            source.contains("private static func captureStructuredLeafFromBoundary("),
            "Structured leaf helper must exist so direct-tool and canonical-user modes share one leaf admission path"
        )
        #expect(
            source.contains("leafPayload: leafPayload"),
            "Structured leaf helper storeLeaf call must forward the extracted leafPayload"
        )
        #expect(
            !source.contains("leafPayload: strippedLeafPayload"),
            "Single-leaf policy should not retain the removed strippedLeafPayload store path"
        )
    }

    @Test
    func mainActorRunClosuresAroundPrefixCacheStoresAreNonSuspending() throws {
        // The three `MainActor.run` closures wrapping
        // `prefixCache.storeSnapshots` / `storeLeaf` must stay
        // synchronous. `SSDSnapshotStore.tryEnqueue` is nonisolated
        // under an `NSLock`; an `await` inside the closure would
        // force the HTTP hot path to suspend mid-admission and break
        // the ordering the pending-ref map was designed around.
        let source = try readLLMActorSource()
        let bodies = extractMainActorRunBodies(
            source: source,
            containing: ["prefixCache.storeSnapshots", "prefixCache.storeLeaf"]
        )
        #expect(
            bodies.count == 3,
            "Expected exactly 3 MainActor.run closures calling prefixCache.store*; found \(bodies.count). A refactor may have moved, collapsed, or duplicated one of the mid-prefill / direct-leaf / structured-leaf-helper sites — review before updating this assertion."
        )
        for (index, body) in bodies.enumerated() {
            #expect(
                !body.contains("await"),
                "MainActor.run closure #\(index) calling prefixCache.store* contains an `await` — non-suspending admission is broken. Body:\n\(body)"
            )
        }
    }

    /// Scan `source` for every `MainActor.run { ... }` trailing-closure
    /// body and return the ones whose body contains at least one of
    /// `anchors`. Uses naive brace matching — adequate because the
    /// closures of interest in `LLMActor.swift` contain no string
    /// literals with braces, no block comments, and no nested closures
    /// wider than the enclosing `MainActor.run`. If that changes, the
    /// call-site count assertion above will start failing and the
    /// matcher can be upgraded then.
    private func extractMainActorRunBodies(
        source: String,
        containing anchors: [String]
    ) -> [String] {
        var result: [String] = []
        var cursor = source.startIndex
        let needle = "MainActor.run"
        while let runRange = source.range(of: needle, range: cursor..<source.endIndex) {
            guard let braceOpen = source[runRange.upperBound...].firstIndex(of: "{") else {
                break
            }
            var depth = 0
            var scanner = braceOpen
            var braceClose: String.Index?
            while scanner < source.endIndex {
                let ch = source[scanner]
                if ch == "{" {
                    depth += 1
                } else if ch == "}" {
                    depth -= 1
                    if depth == 0 {
                        braceClose = scanner
                        break
                    }
                }
                scanner = source.index(after: scanner)
            }
            guard let braceClose else { break }
            let bodyStart = source.index(after: braceOpen)
            let body = String(source[bodyStart..<braceClose])
            if anchors.contains(where: { body.contains($0) }) {
                result.append(body)
            }
            cursor = source.index(after: braceClose)
        }
        return result
    }
}
