//
//  LLMActorExtractSnapshotPayloadsTests.swift
//  tesseractTests
//
//  Unit tests for the LLMActor snapshot extraction edge: attaching
//  RAM-only or SSD-backed storage to Snapshot Admission entries while
//  converting live Metal-resident `HybridCacheSnapshot` instances into
//  pure `Sendable` `SnapshotPayload` values via `MLXArray.asData()`.
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

    private func checkpointCandidates(
        for snapshots: [HybridCacheSnapshot],
        ssdEnabled: Bool = true
    ) -> [SnapshotAdmission.CheckpointCandidate] {
        LLMActor.extractCheckpointAdmissionCandidates(
            snapshots,
            ssdEnabled: ssdEnabled
        )
    }

    private func checkpointPayload(
        for snapshot: HybridCacheSnapshot
    ) throws -> SnapshotPayload {
        let payloads = checkpointPayloads(for: [snapshot])
        try #require(payloads.count == 1)
        return payloads[0]
    }

    private func checkpointPayloads(
        for snapshots: [HybridCacheSnapshot],
        ssdEnabled: Bool = true
    ) -> [SnapshotPayload] {
        checkpointCandidates(
            for: snapshots,
            ssdEnabled: ssdEnabled
        ).compactMap { candidate in
            if case .ramAndSSD(let payload) = candidate.storage {
                return payload
            }
            return nil
        }
    }

    private func expectRAMOnly(
        _ candidates: [SnapshotAdmission.CheckpointCandidate]
    ) {
        for candidate in candidates {
            if case .ramOnly = candidate.storage {
                // expected
            } else {
                #expect(Bool(false), "Expected candidate to be RAM-only")
            }
        }
    }

    // MARK: - SSD gate

    @Test
    func candidatesAreRAMOnlyWhenSSDDisabled() {
        let snapshot = makeSimpleKVSnapshot()
        let candidates = checkpointCandidates(for: [snapshot], ssdEnabled: false)

        #expect(candidates.count == 1)
        #expect(candidates.first?.snapshot.tokenOffset == snapshot.tokenOffset)
        expectRAMOnly(candidates)
    }

    @Test
    func candidatesAreRAMOnlyWhenSSDDisabledForMultipleSnapshots() {
        // The gate must preserve every snapshot while marking each
        // candidate RAM-only. SSD-disabled is no longer represented as
        // an empty payload array.
        let snapshots = [
            makeSimpleKVSnapshot(tokenOffset: 5),
            makeSimpleKVSnapshot(tokenOffset: 9),
            makeMixedSnapshot(tokenOffset: 40),
        ]
        let candidates = checkpointCandidates(for: snapshots, ssdEnabled: false)

        #expect(candidates.map(\.snapshot.tokenOffset) == snapshots.map(\.tokenOffset))
        expectRAMOnly(candidates)
    }

    @Test
    func returnsNoCandidatesForEmptyInputEvenWhenEnabled() {
        let candidates = checkpointCandidates(for: [], ssdEnabled: true)

        #expect(candidates.isEmpty)
    }

    @Test
    func checkpointCandidatesCarryPerEntryStorageAtExtractionEdge() throws {
        let snapshots = [
            makeSimpleKVSnapshot(tokenOffset: 5),
            makeSimpleKVSnapshot(tokenOffset: 9, type: .branchPoint),
        ]

        let ssdCandidates = checkpointCandidates(for: snapshots)

        #expect(ssdCandidates.count == snapshots.count)
        for index in snapshots.indices {
            #expect(ssdCandidates[index].snapshot.tokenOffset == snapshots[index].tokenOffset)
            if case .ramAndSSD(let payload) = ssdCandidates[index].storage {
                #expect(payload.tokenOffset == snapshots[index].tokenOffset)
                #expect(payload.checkpointType == snapshots[index].checkpointType)
            } else {
                #expect(Bool(false), "Expected SSD-enabled candidate to carry its payload")
            }
        }

        let ramOnlyCandidates = checkpointCandidates(for: snapshots, ssdEnabled: false)

        #expect(ramOnlyCandidates.count == snapshots.count)
        expectRAMOnly(ramOnlyCandidates)
    }

    // MARK: - Structural equivalence

    @Test
    func preservesTokenOffsetAndCheckpointType() throws {
        let snapshot = makeSimpleKVSnapshot(tokenOffset: 17, type: .branchPoint)
        let payload = try checkpointPayload(for: snapshot)
        #expect(payload.tokenOffset == 17)
        #expect(payload.checkpointType == .branchPoint)
    }

    @Test
    func preservesPerLayerMetadata() throws {
        let snapshot = makeMixedSnapshot(tokenOffset: 64)
        let payload = try checkpointPayload(for: snapshot)

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
        let payloads = checkpointPayloads(for: snapshots)
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
        let payload = try checkpointPayload(for: snapshot)

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
    /// `encodePlaceholderContainer` (in `PlaceholderContainer.swift`),
    /// so a typo, a case rename, or any divergence from this table
    /// silently corrupts cache files that a future reader would reject.
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
        let payload = try checkpointPayload(for: snapshot)

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

    // MARK: - PrefixCacheManager admission plumbing

    @MainActor
    @Test
    func admitAcceptsSnapshotAdmissionPayloads() throws {
        // Snapshot Admission must carry extracted payloads into the
        // cache manager without breaking RAM insertion or diagnostics.
        let manager = PrefixCacheManager(
            memoryBudgetBytes: 16 * 1024 * 1024
        )
        let snapshot = makeSimpleKVSnapshot(tokenOffset: 4)
        let payload = try checkpointPayload(for: snapshot)
        let partitionKey = CachePartitionKey(
            modelID: "test-model", kvBits: nil, kvGroupSize: 64
        )
        let admission = try #require(SnapshotAdmission.checkpoints(
            fullPromptTokens: [1, 2, 3, 4],
            candidates: [
                SnapshotAdmission.CheckpointCandidate(
                    snapshot: snapshot,
                    storage: .ramAndSSD(payload)
                )
            ],
            partitionKey: partitionKey,
            requestID: UUID()
        ))
        let diagnostics = manager.admit(admission)

        #expect(diagnostics.evictions.isEmpty)
        #expect(manager.stats.snapshotCount == 1)
    }

    @MainActor
    @Test
    func snapshotAdmissionSupportsRAMOnlyCheckpointEntries() throws {
        // When `ssdConfig?.enabled` is off, the LLMActor helper returns
        // `[]` for payload extraction. The extraction edge represents
        // that as a RAM-only admission entry, not as "no snapshots".
        let manager = PrefixCacheManager(
            memoryBudgetBytes: 16 * 1024 * 1024
        )
        let snapshot = makeSimpleKVSnapshot(tokenOffset: 5)
        let partitionKey = CachePartitionKey(
            modelID: "test-model", kvBits: nil, kvGroupSize: 64
        )
        let admission = try #require(SnapshotAdmission.checkpoints(
            fullPromptTokens: [7, 8, 9, 10, 11],
            candidates: [
                SnapshotAdmission.CheckpointCandidate(
                    snapshot: snapshot,
                    storage: .ramOnly
                )
            ],
            partitionKey: partitionKey,
            requestID: UUID()
        ))
        let diagnostics = manager.admit(admission)

        #expect(diagnostics.evictions.isEmpty)
        #expect(manager.stats.snapshotCount == 1)
    }

    // MARK: - Call-site wiring regression coverage
    //
    // The leaf LLMActor call sites cannot be exercised by unit tests
    // without a loaded MLX model — the full wiring is gated behind
    // `container.perform` on a real `ModelContainer`. These source
    // checks pin shared structured leaf capture and the synchronous
    // cache admission invariant.
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
    func structuredLeafHelperKeepsLeafPayloadCaptureShared() throws {
        // The tool-loop direct leaf and canonical user leaf now share
        // one `captureStructuredLeafFromBoundary(...)` helper. That
        // helper must still extract a local `leafPayload` and pair it
        // with the leaf snapshot in one admission value; the older dedicated
        // `strippedLeafPayload` path no longer exists under the
        // single-leaf policy.
        let source = try readLLMActorSource()
        #expect(
            source.contains("private static func captureStructuredLeafFromBoundary("),
            "Structured leaf helper must exist so direct-tool and canonical-user modes share one leaf admission path"
        )
        #expect(
            source.components(separatedBy: "let leafAdmission = SnapshotAdmission.leaf(").count - 1 == 2,
            "Both production leaf paths must construct leaf Snapshot Admission values"
        )
        #expect(
            source.components(separatedBy: "prefixCache.admit(leafAdmission)").count - 1 == 2,
            "Both production leaf paths must mutate the cache through admit"
        )
        #expect(
            !source.contains("leafPayload: strippedLeafPayload"),
            "Single-leaf policy should not retain the removed strippedLeafPayload store path"
        )
    }

    @Test
    func mainActorRunClosuresAroundPrefixCacheAdmissionsAreNonSuspending() throws {
        // The three `MainActor.run` closures wrapping
        // `prefixCache.admit` must stay
        // synchronous. `SSDSnapshotStore.tryEnqueue` is nonisolated
        // under an `NSLock`; an `await` inside the closure would
        // force the HTTP hot path to suspend mid-admission and break
        // the ordering the pending-ref map was designed around.
        let source = try readLLMActorSource()
        let bodies = extractMainActorRunBodies(
            source: source,
            containing: ["prefixCache.admit"]
        )
        #expect(
            bodies.count == 3,
            "Expected exactly 3 MainActor.run closures calling prefixCache admission APIs; found \(bodies.count). A refactor may have moved, collapsed, or duplicated one of the mid-prefill / direct-leaf / structured-leaf-helper sites — review before updating this assertion."
        )
        for (index, body) in bodies.enumerated() {
            #expect(
                !body.contains("await"),
                "MainActor.run closure #\(index) calling prefixCache admission APIs contains an `await` — non-suspending admission is broken. Body:\n\(body)"
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
