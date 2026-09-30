//
//  SnapshotPayloadTests.swift
//  tesseractTests
//
//  The **Snapshot Payload** (ADR-0078): a snapshot's SSD byte form, built by
//  **Deferred Payload Extraction**. These pin what the SSD writer receives:
//  the byte total fixed at the extraction edge, the host copy deferred to the
//  writer's task, borrowed storage that outlives the snapshot and is released
//  layer by layer, a view payload detached from both its sources, and the
//  dtype names that are part of the on-disk contract. Pure fixtures, no model.
//

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

struct SnapshotPayloadTests {

    @Test func viewPayloadDetachesEveryArrayAndDefersTheHostCopy() async throws {
        let attention = KVCacheSimple()
        attention.state = [
            MLXArray((0..<512).map(Float.init)).reshaped([1, 1, 8, 64]),
            MLXArray.ones([1, 1, 8, 64]),
        ]
        let recurrent = MambaCache()
        recurrent.state = [MLXArray([Float(42), 43, 44])]
        attention.trim(4)
        let view = try #require(
            HybridCacheSnapshot.capture(
                cache: [attention, recurrent], offset: 4, type: .branchPoint, prefixView: true))
        attention.state = [
            MLXArray((0..<512).map(Float.init)).reshaped([1, 1, 8, 64]),
            MLXArray.ones([1, 1, 8, 64]),
        ]
        recurrent.state = [MLXArray([Float(99), 100, 101])]
        let leaf = try #require(
            HybridCacheSnapshot.capture(cache: [attention, recurrent], offset: 8, type: .leaf))
        let (payload, owed) = try SnapshotPayload.deferred(for: view, backingLeaf: leaf)
        #expect(payload.extending == nil)
        #expect(payload.totalBytes == 2060)  // two 4×64 float32 arrays + three recurrent values
        #expect(!payload.isMaterialized)
        #expect(owed.retainedArrays.count == 3)
        let originalAddresses = Set(
            (leaf.layers + view.layers).flatMap(\.state).map(backingAddress))
        #expect(owed.retainedArrays.allSatisfy { !originalAddresses.contains(backingAddress($0)) })
        // No host data exists at the extraction edge. The eventual reader
        // runs the host copy; production supplies the SSD writer's task.
        let layers = await Task.detached { payload.layers }.value
        #expect(payload.isMaterialized)
        #expect(owed.retainedArrays.isEmpty)
        #expect(SnapshotPayload.byteCount(of: layers) == 2060)
        #expect(layers[0].state[0].shape == [1, 1, 4, 64])
        #expect(
            layers[0].state[0].data
                == MLXArray((0..<256).map(Float.init)).asData(access: .copy).data)
        #expect(
            layers[1].state[0].data == MLXArray([Float(42), 43, 44]).asData(access: .copy).data)
        #expect(layers.allSatisfy { $0.suffixBaseOffset == nil })
    }

    @Test func deferredBytesBorrowEvaluatedStorageAndOutliveTheSnapshot() throws {
        weak var backing: MLXArray?
        let (payload, address, expected): (SnapshotPayload, UInt, Data) = {
            let array = MLXArray(Array(0..<256).map(Float.init))
            eval(array)
            backing = array
            let snapshot = HybridCacheSnapshot(
                tokenOffset: 1,
                layers: [.init(className: "ArraysCache", state: [array], metaState: [], offset: 1)],
                checkpointType: .leaf, memoryBytes: array.nbytes, createdAt: .now)
            return (
                SnapshotPayload.extract(snapshot), backingAddress(array),
                array.asData(access: .copy).data
            )
        }()
        let bytes = try #require(payload.layers.first?.state.first?.data)
        #expect(backing != nil, "the borrowed Data must retain its evaluated MLX owner")
        #expect(bytes == expected)
        bytes.withUnsafeBytes { (buffer: UnsafeRawBufferPointer) in
            #expect(UInt(bitPattern: buffer.baseAddress) == address)
        }
    }

    @Test func streamingReleasesBorrowedArraysAfterEachLayer() throws {
        weak var first: MLXArray?
        weak var second: MLXArray?
        let payload: SnapshotPayload = {
            let arrays = [MLXArray.ones([256]), MLXArray.ones([256])]
            eval(arrays)
            first = arrays[0]
            second = arrays[1]
            let snapshot = HybridCacheSnapshot(
                tokenOffset: 1,
                layers: arrays.map {
                    .init(className: "ArraysCache", state: [$0], metaState: [], offset: 1)
                },
                checkpointType: .leaf, memoryBytes: 2048, createdAt: .now)
            return SnapshotPayload.extract(snapshot)
        }()
        let descriptor = PersistedSnapshotDescriptor(
            snapshotID: "borrowed", partitionDigest: "abcd1234",
            pathFromRoot: [1], tokenOffset: 1, checkpointType: "leaf", bytes: 2048,
            createdAt: 100, lastAccessAt: 100, fileRelativePath: "borrowed.safetensors",
            schemaVersion: SnapshotManifestSchema.currentVersion)
        let encoding = try PlaceholderContainerEncoding(payload: payload, descriptor: descriptor)
        #expect(payload.retainsBodyArrays)
        var consumed = 0
        encoding.withChunks(maximumBytes: 256, releasingLayers: true) { chunk in
            if consumed < encoding.headerByteCount + 1024 {
                #expect(first != nil)
            } else {
                #expect(first == nil, "the first layer must release before the second writes")
            }
            #expect(second != nil)
            consumed += chunk.count
        }
        #expect(consumed == encoding.byteCount)
        #expect(first == nil)
        #expect(second == nil)
        #expect(!payload.retainsBodyArrays)
    }

    @Test func deferredEmptyArraysProduceEmptyBytes() throws {
        let array = MLXArray.zeros([0])
        let snapshot = HybridCacheSnapshot(
            tokenOffset: 1,
            layers: [.init(className: "ArraysCache", state: [array], metaState: [], offset: 1)],
            checkpointType: .leaf, memoryBytes: 0, createdAt: .now)
        let payload = SnapshotPayload.extract(snapshot)
        #expect(payload.layers[0].state[0].data.isEmpty)
        #expect(payload.layers[0].state[0].shape == [0])
    }

    // MARK: - Structural equivalence

    @Test
    func preservesTokenOffsetAndCheckpointType() throws {
        let snapshot = PrefixCacheTestFixtures.makeSimpleKVSnapshot(
            tokenOffset: 17, type: .branchPoint)
        let payload = SnapshotPayload.extract(snapshot)
        #expect(payload.tokenOffset == 17)
        #expect(payload.checkpointType == .branchPoint)
    }

    @Test
    func preservesPerLayerMetadata() throws {
        let snapshot = PrefixCacheTestFixtures.makeMixedSnapshot(tokenOffset: 64)
        let payload = SnapshotPayload.extract(snapshot)

        #expect(payload.layers.count == snapshot.layers.count)
        for (layerIdx, layer) in payload.layers.enumerated() {
            let source = snapshot.layers[layerIdx]
            #expect(layer.className == source.className)
            #expect(layer.metaState == source.metaState)
            #expect(layer.offset == source.offset)
            #expect(layer.state.count == source.state.count)
        }
    }

    // MARK: - Byte round-trip

    @Test
    func byteRoundTripMatchesSourceArrays() throws {
        let snapshot = PrefixCacheTestFixtures.makeSimpleKVSnapshot()
        let payload = SnapshotPayload.extract(snapshot)

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
                #expect(
                    payloadArray.data == reference.data,
                    "layer \(layerIdx) array \(arrayIdx) bytes mismatch")
                #expect(
                    payloadArray.shape == reference.shape,
                    "layer \(layerIdx) array \(arrayIdx) shape mismatch")
                #expect(
                    payloadArray.dtype == "float32",
                    "layer \(layerIdx) array \(arrayIdx) dtype — expected the pinned wire-format literal \"float32\""
                )
            }
        }
    }

    // MARK: - Deferred Payload Extraction

    @Test
    func extractionDefersTheHostCopyAndFixesTheByteTotal() throws {
        let snapshot = PrefixCacheTestFixtures.makeMixedSnapshot(tokenOffset: 64)
        let payload = SnapshotPayload.extract(snapshot)

        // The extraction edge reads shapes only; the memcpy belongs to the
        // SSD writer's task (a demotion used to run it on the MainActor).
        #expect(!payload.isMaterialized)
        #expect(payload.totalBytes == snapshot.memoryBytes)

        let layers = payload.layers
        #expect(payload.isMaterialized)
        #expect(SnapshotPayload.byteCount(of: layers) == payload.totalBytes)
        #expect(layers.count == snapshot.layers.count)
    }

    /// Literal pinning of the SSD on-disk wire-format contract. These
    /// strings are written verbatim into the snapshot header at
    /// `encodePlaceholderContainer` (in `PlaceholderContainer.swift`),
    /// so a typo, a case rename, or any divergence from this table
    /// silently corrupts cache files that a future reader would reject.
    /// Assertions are literal — NOT sourced from
    /// `SnapshotPayload.dtypeWireString` — so self-consistency cannot mask a
    /// contract drift. If a new `DType` case is added, extend this
    /// table in lockstep with the production helper.
    @Test
    func dtypeWireStringsArePinned() {
        #expect(SnapshotPayload.dtypeWireString(.bool) == "bool")
        #expect(SnapshotPayload.dtypeWireString(.uint8) == "uint8")
        #expect(SnapshotPayload.dtypeWireString(.uint16) == "uint16")
        #expect(SnapshotPayload.dtypeWireString(.uint32) == "uint32")
        #expect(SnapshotPayload.dtypeWireString(.uint64) == "uint64")
        #expect(SnapshotPayload.dtypeWireString(.int8) == "int8")
        #expect(SnapshotPayload.dtypeWireString(.int16) == "int16")
        #expect(SnapshotPayload.dtypeWireString(.int32) == "int32")
        #expect(SnapshotPayload.dtypeWireString(.int64) == "int64")
        #expect(SnapshotPayload.dtypeWireString(.float16) == "float16")
        #expect(SnapshotPayload.dtypeWireString(.float32) == "float32")
        #expect(SnapshotPayload.dtypeWireString(.bfloat16) == "bfloat16")
        #expect(SnapshotPayload.dtypeWireString(.complex64) == "complex64")
        #expect(SnapshotPayload.dtypeWireString(.float64) == "float64")
    }

    @Test
    func totalBytesMatchesSumOfArrayNbytes() throws {
        let snapshot = PrefixCacheTestFixtures.makeMixedSnapshot()
        let payload = SnapshotPayload.extract(snapshot)

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
}
