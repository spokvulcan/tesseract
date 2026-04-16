//
//  SnapshotManifestTests.swift
//  tesseractTests
//
//  Round-trip tests for the SSD prefix-cache data model. The types
//  under test are pure value types — no store, no writer, no I/O.
//  These tests pin the Codable wire format, the non-Codable
//  transport shapes (`SnapshotStorageRef`, `SnapshotPayload`), and
//  the `CheckpointType` wire-string mapping. Downstream code
//  (`SSDSnapshotStore`, `TieredSnapshotStore`, warm start) builds
//  on these guarantees.
//

import Foundation
import Testing
import MLXLMCommon

@testable import Tesseract_Agent

struct SnapshotManifestTests {

    // MARK: - Fixture builders

    private func makeDescriptor(
        id: String = "11111111-2222-3333-4444-555555555555",
        partition: String = "abcd1234",
        type: String = "system",
        bytes: Int = 209_715_200
    ) -> PersistedSnapshotDescriptor {
        PersistedSnapshotDescriptor(
            snapshotID: id,
            partitionDigest: partition,
            pathFromRoot: [151_643, 9_707, 27, 42],
            tokenOffset: 4_096,
            checkpointType: type,
            bytes: bytes,
            createdAt: 800_000.5,
            lastAccessAt: 800_010.25,
            fileRelativePath: "partitions/\(partition)/snapshots/1/\(id).safetensors",
            schemaVersion: SnapshotManifestSchema.currentVersion
        )
    }

    private func makePartitionMeta(
        fingerprint: String = String(repeating: "a", count: 64)
    ) -> PartitionMeta {
        PartitionMeta(
            modelID: "mlx-community/Qwen3-4B-4bit",
            modelFingerprint: fingerprint,
            kvBits: 8,
            kvGroupSize: 64,
            createdAt: 799_999.0,
            schemaVersion: SnapshotManifestSchema.currentVersion
        )
    }

    /// JSON encoder/decoder pair with a deterministic output shape so
    /// round-trip equality is not sensitive to dictionary iteration
    /// order or pretty-printing.
    private func jsonCodecs() -> (JSONEncoder, JSONDecoder) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        let decoder = JSONDecoder()
        return (encoder, decoder)
    }

    // MARK: - Schema version

    @Test
    func schemaVersionIsFour() {
        // Bumping invalidates every existing manifest. Pinned so an
        // accidental bump trips this test.
        #expect(SnapshotManifestSchema.currentVersion == 4)
    }

    @Test
    func emptyManifestUsesCurrentSchemaVersion() {
        let manifest = SnapshotManifest.empty()
        #expect(manifest.schemaVersion == SnapshotManifestSchema.currentVersion)
        #expect(manifest.partitions.isEmpty)
        #expect(manifest.snapshots.isEmpty)
    }

    // MARK: - PersistedSnapshotDescriptor round-trip

    @Test
    func persistedSnapshotDescriptorRoundTripsViaJSON() throws {
        let original = makeDescriptor()
        let (encoder, decoder) = jsonCodecs()

        let data = try encoder.encode(original)
        let decoded = try decoder.decode(PersistedSnapshotDescriptor.self, from: data)

        #expect(decoded == original)
    }

    @Test
    func persistedSnapshotDescriptorPreservesEveryField() throws {
        let original = makeDescriptor(
            id: "deadbeef-0000-1111-2222-333344445555",
            partition: "cafef00d",
            type: "branchPoint",
            bytes: 123_456_789
        )
        let (encoder, decoder) = jsonCodecs()

        let data = try encoder.encode(original)
        let decoded = try decoder.decode(PersistedSnapshotDescriptor.self, from: data)

        #expect(decoded.snapshotID == original.snapshotID)
        #expect(decoded.partitionDigest == original.partitionDigest)
        #expect(decoded.pathFromRoot == original.pathFromRoot)
        #expect(decoded.tokenOffset == original.tokenOffset)
        #expect(decoded.checkpointType == original.checkpointType)
        #expect(decoded.bytes == original.bytes)
        #expect(decoded.createdAt == original.createdAt)
        #expect(decoded.lastAccessAt == original.lastAccessAt)
        #expect(decoded.fileRelativePath == original.fileRelativePath)
        #expect(decoded.schemaVersion == original.schemaVersion)
    }

    @Test
    func persistedSnapshotDescriptorLastAccessAtIsMutable() {
        // `lastAccessAt` is the single `var` field — the store bumps
        // it on every hit. A compile-level assertion: mutating in
        // place should compile, and the value should change.
        var descriptor = makeDescriptor()
        let before = descriptor.lastAccessAt
        descriptor.lastAccessAt = before + 100
        #expect(descriptor.lastAccessAt == before + 100)
    }

    @Test
    func persistedSnapshotDescriptorWireKeysMatchFieldNames() throws {
        // The JSON keys are the Swift field names (synthesized
        // CodingKeys default). The on-disk manifest relies on
        // these exact key names, so a custom `CodingKeys` enum
        // that renames any field silently breaks every previously
        // persisted manifest. Pin the key set here.
        let original = makeDescriptor()
        let data = try JSONEncoder().encode(original)
        let wire = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        let expectedKeys: Set<String> = [
            "snapshotID",
            "partitionDigest",
            "pathFromRoot",
            "tokenOffset",
            "checkpointType",
            "bytes",
            "createdAt",
            "lastAccessAt",
            "fileRelativePath",
            "schemaVersion",
        ]
        #expect(Set(wire.keys) == expectedKeys)
    }

    // MARK: - PartitionMeta round-trip

    @Test
    func partitionMetaRoundTripsViaJSON() throws {
        let original = makePartitionMeta()
        let (encoder, decoder) = jsonCodecs()

        let data = try encoder.encode(original)
        let decoded = try decoder.decode(PartitionMeta.self, from: data)

        #expect(decoded == original)
    }

    @Test
    func partitionMetaHandlesNilOptionalFields() throws {
        // `kvBits` is optional; it must round-trip cleanly as `null`
        // in JSON and come back as `nil`.
        let original = PartitionMeta(
            modelID: "mlx-community/Qwen3-4B-4bit",
            modelFingerprint: String(repeating: "b", count: 64),
            kvBits: nil,
            kvGroupSize: 128,
            createdAt: 0,
            schemaVersion: SnapshotManifestSchema.currentVersion
        )
        let (encoder, decoder) = jsonCodecs()

        let data = try encoder.encode(original)
        let decoded = try decoder.decode(PartitionMeta.self, from: data)

        #expect(decoded == original)
        #expect(decoded.kvBits == nil)
    }

    // MARK: - SnapshotManifest round-trip

    @Test
    func snapshotManifestEmptyRoundTripsViaJSON() throws {
        let original = SnapshotManifest.empty()
        let (encoder, decoder) = jsonCodecs()

        let data = try encoder.encode(original)
        let decoded = try decoder.decode(SnapshotManifest.self, from: data)

        #expect(decoded == original)
        #expect(decoded.schemaVersion == SnapshotManifestSchema.currentVersion)
        #expect(decoded.partitions.isEmpty)
        #expect(decoded.snapshots.isEmpty)
    }

    @Test
    func snapshotManifestPopulatedRoundTripsViaJSON() throws {
        // Build a realistic manifest with two partitions and three
        // snapshots, then round-trip it through JSON and verify the
        // whole structure survives. Uses the deterministic codec so
        // dictionary order is not a problem.
        let partA = makePartitionMeta(fingerprint: String(repeating: "a", count: 64))
        let partB = PartitionMeta(
            modelID: "mlx-community/Qwen3-9B-8bit",
            modelFingerprint: String(repeating: "b", count: 64),
            kvBits: 4,
            kvGroupSize: 32,
            createdAt: 900_000,
            schemaVersion: SnapshotManifestSchema.currentVersion
        )
        let descA = makeDescriptor(
            id: "aaaaaaaa-0000-1111-2222-333344445555",
            partition: "11111111",
            type: "system",
            bytes: 200_000_000
        )
        let descB = makeDescriptor(
            id: "bbbbbbbb-0000-1111-2222-333344445555",
            partition: "11111111",
            type: "leaf",
            bytes: 50_000_000
        )
        let descC = makeDescriptor(
            id: "cccccccc-0000-1111-2222-333344445555",
            partition: "22222222",
            type: "branchPoint",
            bytes: 75_000_000
        )

        let original = SnapshotManifest(
            schemaVersion: SnapshotManifestSchema.currentVersion,
            partitions: ["11111111": partA, "22222222": partB],
            snapshots: [
                descA.snapshotID: descA,
                descB.snapshotID: descB,
                descC.snapshotID: descC,
            ]
        )
        let (encoder, decoder) = jsonCodecs()

        let data = try encoder.encode(original)
        let decoded = try decoder.decode(SnapshotManifest.self, from: data)

        #expect(decoded == original)
        #expect(decoded.partitions.count == 2)
        #expect(decoded.snapshots.count == 3)
        #expect(decoded.partitions["11111111"] == partA)
        #expect(decoded.partitions["22222222"] == partB)
        #expect(decoded.snapshots[descA.snapshotID] == descA)
        #expect(decoded.snapshots[descB.snapshotID] == descB)
        #expect(decoded.snapshots[descC.snapshotID] == descC)
    }

    @Test
    func snapshotManifestMutationsAreAllowedOnVarFields() {
        // The manifest's `partitions` and `snapshots` dictionaries are
        // `var` so the store can mutate them in place. Compile-level
        // proof: mutation should work without reassigning the whole
        // manifest.
        var manifest = SnapshotManifest.empty()
        manifest.partitions["aaa"] = makePartitionMeta()
        manifest.snapshots[makeDescriptor().snapshotID] = makeDescriptor()
        manifest.schemaVersion = SnapshotManifestSchema.currentVersion + 1

        #expect(manifest.partitions.count == 1)
        #expect(manifest.snapshots.count == 1)
        #expect(manifest.schemaVersion == SnapshotManifestSchema.currentVersion + 1)
    }

    // MARK: - SnapshotStorageRef (non-Codable value type)

    @Test
    func snapshotStorageRefConstructsAllFields() {
        let ref = SnapshotStorageRef(
            snapshotID: "ref-id-123",
            partitionDigest: "abcd1234",
            tokenOffset: 4_096,
            checkpointType: .system,
            bytesOnDisk: 200_000_000,
            lastAccessTime: .now,
            committed: false
        )
        #expect(ref.snapshotID == "ref-id-123")
        #expect(ref.partitionDigest == "abcd1234")
        #expect(ref.tokenOffset == 4_096)
        #expect(ref.checkpointType == .system)
        #expect(ref.bytesOnDisk == 200_000_000)
        #expect(ref.committed == false)
    }

    @Test
    func snapshotStorageRefCommittedIsMutable() {
        // `committed` flips from false → true when the writer loop
        // fsyncs the file. Make sure the in-place mutation compiles
        // and behaves.
        var ref = SnapshotStorageRef(
            snapshotID: "flip-test",
            partitionDigest: "abcd1234",
            tokenOffset: 0,
            checkpointType: .leaf,
            bytesOnDisk: 1,
            lastAccessTime: .now,
            committed: false
        )
        #expect(ref.committed == false)
        ref.committed = true
        #expect(ref.committed == true)
    }

    @Test
    func snapshotStorageRefLastAccessTimeIsMutable() {
        // `lastAccessTime` moves forward on every hit. Confirm the
        // field is a `var` in practice by mutating it.
        var ref = SnapshotStorageRef(
            snapshotID: "bump-test",
            partitionDigest: "abcd1234",
            tokenOffset: 0,
            checkpointType: .leaf,
            bytesOnDisk: 1,
            lastAccessTime: .now,
            committed: true
        )
        let before = ref.lastAccessTime
        ref.lastAccessTime = before.advanced(by: .milliseconds(50))
        #expect(ref.lastAccessTime > before)
    }

    // MARK: - SnapshotPayload aggregation

    @Test
    func snapshotPayloadTotalBytesSumsEveryArray() {
        let layer0 = SnapshotPayload.LayerPayload(
            className: "KVCache",
            state: [
                SnapshotPayload.ArrayPayload(
                    data: Data(repeating: 0xAA, count: 1_024),
                    dtype: "bfloat16",
                    shape: [1, 32, 1_024, 128]
                ),
                SnapshotPayload.ArrayPayload(
                    data: Data(repeating: 0xBB, count: 2_048),
                    dtype: "bfloat16",
                    shape: [1, 32, 2_048, 128]
                ),
            ],
            metaState: [],
            offset: 4_096
        )
        let layer1 = SnapshotPayload.LayerPayload(
            className: "KVCache",
            state: [
                SnapshotPayload.ArrayPayload(
                    data: Data(repeating: 0xCC, count: 512),
                    dtype: "bfloat16",
                    shape: [1, 32, 512, 128]
                )
            ],
            metaState: [],
            offset: 4_096
        )
        let payload = SnapshotPayload(
            tokenOffset: 4_096,
            checkpointType: .system,
            layers: [layer0, layer1]
        )

        #expect(payload.totalBytes == 1_024 + 2_048 + 512)
        #expect(payload.checkpointType == .system)
        #expect(payload.tokenOffset == 4_096)
        #expect(payload.layers.count == 2)
    }

    @Test
    func snapshotPayloadTotalBytesIsZeroForEmptyLayers() {
        // Defensive: a payload with zero layers or zero arrays per
        // layer should return 0, not crash. The writer's byte-budget
        // check relies on this.
        let empty = SnapshotPayload(
            tokenOffset: 0,
            checkpointType: .leaf,
            layers: []
        )
        #expect(empty.totalBytes == 0)

        let emptyLayers = SnapshotPayload(
            tokenOffset: 0,
            checkpointType: .leaf,
            layers: [
                SnapshotPayload.LayerPayload(
                    className: "KVCache",
                    state: [],
                    metaState: [],
                    offset: 0
                )
            ]
        )
        #expect(emptyLayers.totalBytes == 0)
    }

    @Test
    func snapshotPayloadPreservesLayerMetaState() {
        // metaState strings round-trip through the payload; they are
        // required for KVCache reconstruction on hydration.
        let layer = SnapshotPayload.LayerPayload(
            className: "QuantizedKVCache",
            state: [],
            metaState: ["64", "4", "start-pos:0"],
            offset: 128
        )
        let payload = SnapshotPayload(
            tokenOffset: 128,
            checkpointType: .branchPoint,
            layers: [layer]
        )
        #expect(payload.layers[0].metaState == ["64", "4", "start-pos:0"])
        #expect(payload.layers[0].className == "QuantizedKVCache")
        #expect(payload.layers[0].offset == 128)
    }

    // MARK: - CheckpointType wire format

    @Test
    func checkpointTypeWireStringRoundTripsForEveryCase() {
        let cases: [HybridCacheSnapshot.CheckpointType] = [
            .system,
            .leaf,
            .branchPoint,
        ]
        for original in cases {
            let wire = original.wireString
            let decoded = HybridCacheSnapshot.CheckpointType(wireString: wire)
            #expect(decoded == original, "round-trip failed for \(original)")
        }
    }

    @Test
    func checkpointTypeWireStringMatchesExpectedNames() {
        // The wire form is part of the on-disk schema — pin the exact
        // strings so a future rename does not silently break existing
        // manifests.
        #expect(HybridCacheSnapshot.CheckpointType.system.wireString == "system")
        #expect(HybridCacheSnapshot.CheckpointType.leaf.wireString == "leaf")
        #expect(HybridCacheSnapshot.CheckpointType.branchPoint.wireString == "branchPoint")
    }

    @Test
    func checkpointTypeInitWireStringReturnsNilForUnknownInput() {
        // Warm start relies on `nil` → "drop this descriptor" to
        // tolerate future enum-case additions. Pin the behavior.
        #expect(HybridCacheSnapshot.CheckpointType(wireString: "") == nil)
        #expect(HybridCacheSnapshot.CheckpointType(wireString: "SYSTEM") == nil)
        #expect(HybridCacheSnapshot.CheckpointType(wireString: "bogus") == nil)
        #expect(HybridCacheSnapshot.CheckpointType(wireString: "lastMessageBoundary") == nil)
        #expect(HybridCacheSnapshot.CheckpointType(wireString: "system ") == nil)
    }

    // MARK: - Equatable behavior

    @Test
    func descriptorEqualityIsFieldWise() {
        let a = makeDescriptor()
        let b = makeDescriptor()
        #expect(a == b)

        var c = a
        c.lastAccessAt = a.lastAccessAt + 1
        #expect(a != c)
    }

    @Test
    func partitionMetaEqualityIsFieldWise() {
        let a = makePartitionMeta()
        var b = a
        #expect(a == b)
        b = PartitionMeta(
            modelID: a.modelID,
            modelFingerprint: a.modelFingerprint,
            kvBits: a.kvBits,
            kvGroupSize: a.kvGroupSize + 1,
            createdAt: a.createdAt,
            schemaVersion: a.schemaVersion
        )
        #expect(a != b)
    }

    // MARK: - Partition digest (wire contract)

    @Test
    func partitionDigestIsEightLowercaseHexChars() {
        // Contract: the digest is exactly the low 8 hex chars of an
        // FNV-1a 32-bit hash. Any deviation (longer, uppercase,
        // non-hex) breaks the on-disk directory layout.
        let key = CachePartitionKey(
            modelID: "m",
            kvBits: nil,
            kvGroupSize: 64,
            modelFingerprint: nil
        )
        let digest = key.partitionDigest
        #expect(digest.count == 8)
        #expect(digest.allSatisfy { $0.isHexDigit && !$0.isUppercase })
    }

    @Test
    func partitionDigestIsDeterministic() {
        // Two keys built from identical inputs must produce identical
        // digests on every call. Any randomness or external-state
        // dependency in the canonicalization/hash would break the
        // wire contract the moment the writer and warm-start paths
        // run in different processes.
        let a = CachePartitionKey(
            modelID: "mlx-community/Qwen3-4B-4bit",
            kvBits: 8,
            kvGroupSize: 64,
            modelFingerprint: "deadbeef"
        )
        let b = CachePartitionKey(
            modelID: "mlx-community/Qwen3-4B-4bit",
            kvBits: 8,
            kvGroupSize: 64,
            modelFingerprint: "deadbeef"
        )
        #expect(a.partitionDigest == b.partitionDigest)
        #expect(a.partitionDigest == a.partitionDigest)
    }

    @Test
    func partitionDigestStableWirePin() {
        // **Load-bearing pin.** Any change to the canonicalization
        // rule, the separator, the FNV algorithm, or the nil
        // sentinel invalidates every previously persisted snapshot.
        // The writer and warm-start paths both compute this digest,
        // so a silent drift makes snapshots unreachable or partitions
        // collapse together.
        //
        // If this test breaks because of an intentional schema
        // change, bump `SnapshotManifestSchema.currentVersion` and
        // the warm-start path will wipe old manifests on next boot.
        // Do NOT just update the expected value — that's the bug
        // this test is designed to catch.
        let key = CachePartitionKey(
            modelID: "mlx-community/Qwen3-4B-4bit",
            kvBits: 8,
            kvGroupSize: 64,
            modelFingerprint: "deadbeef"
        )
        #expect(key.partitionDigest == "ecfce886")
    }

    @Test
    func partitionDigestDistinguishesEveryField() {
        // Every key field must feed into the digest — if any field
        // is dropped from canonicalization, two logically distinct
        // partitions would collide on disk and cross-contaminate.
        let base = CachePartitionKey(
            modelID: "m", kvBits: 8, kvGroupSize: 64,
            modelFingerprint: "f"
        )
        let diffs: [(label: String, key: CachePartitionKey)] = [
            ("modelID", CachePartitionKey(
                modelID: "m2", kvBits: 8, kvGroupSize: 64,
                modelFingerprint: "f"
            )),
            ("kvBits", CachePartitionKey(
                modelID: "m", kvBits: 4, kvGroupSize: 64,
                modelFingerprint: "f"
            )),
            ("kvGroupSize", CachePartitionKey(
                modelID: "m", kvBits: 8, kvGroupSize: 32,
                modelFingerprint: "f"
            )),
            ("modelFingerprint", CachePartitionKey(
                modelID: "m", kvBits: 8, kvGroupSize: 64,
                modelFingerprint: "f2"
            )),
        ]
        for diff in diffs {
            #expect(
                base.partitionDigest != diff.key.partitionDigest,
                "\(diff.label) does not affect the digest"
            )
        }
    }

    @Test
    func partitionDigestNilFieldsStable() {
        // A key with all-nil optionals must produce a stable digest
        // regardless of how many times it's computed. The nil case
        // is the most common warm-start scenario for RAM-only test
        // fixtures and benchmark runs that pass no fingerprint.
        let a = CachePartitionKey(
            modelID: "m",
            kvBits: nil,
            kvGroupSize: 64,
            modelFingerprint: nil
        )
        let b = CachePartitionKey(
            modelID: "m",
            kvBits: nil,
            kvGroupSize: 64,
            modelFingerprint: nil
        )
        #expect(a.partitionDigest == b.partitionDigest)
        // Distinct from a key with the same fields under an explicit
        // fingerprint — the nil branch must not collapse into the
        // Some branch.
        let withFingerprint = CachePartitionKey(
            modelID: "m",
            kvBits: nil,
            kvGroupSize: 64,
            modelFingerprint: "abc"
        )
        #expect(a.partitionDigest != withFingerprint.partitionDigest)
    }

    @Test
    func partitionDigestFingerprintNilDistinguishedFromLiteralSentinelStrings() {
        // **Regression trap for a real correctness bug.** A bare
        // sentinel like `"none"` for nil would collide with an
        // explicit modelFingerprint value of `"none"`. The structural
        // presence-tag encoding must keep every (nil, string) pair
        // distinct, including the obvious sentinel strings
        // (`"none"`, `"N"`, `""`) and the presence-tag prefix (`"S"`).
        let probes: [(label: String, value: String)] = [
            ("none", "none"),
            ("N", "N"),
            ("S", "S"),
            ("empty", ""),
            ("Sanything", "Sanything"),
        ]
        for probe in probes {
            let nilKey = CachePartitionKey(
                modelID: "m",
                kvBits: 8,
                kvGroupSize: 64,
                modelFingerprint: nil
            )
            let valueKey = CachePartitionKey(
                modelID: "m",
                kvBits: 8,
                kvGroupSize: 64,
                modelFingerprint: probe.value
            )
            #expect(
                nilKey.partitionDigest != valueKey.partitionDigest,
                "nil modelFingerprint collides with literal \"\(probe.label)\""
            )
        }
    }

    // MARK: - TriAttention partition identity

    private func makeTriAttentionIdentity(
        budget: Int = 12_000,
        artifact: String? = "aaa",
        impl: TriAttentionImplementationVersion = .v1
    ) -> TriAttentionPartitionIdentity {
        .triAttention(
            budgetTokens: budget,
            calibrationArtifactIdentity: artifact.map {
                TriAttentionCalibrationArtifactIdentity(rawValue: $0)
            },
            implementationVersion: impl
        )
    }

    private func makeKey(
        modelID: String = "mlx-community/Qwen3-4B-4bit",
        fingerprint: String? = "deadbeef",
        triAttention: TriAttentionPartitionIdentity = .dense
    ) -> CachePartitionKey {
        CachePartitionKey(
            modelID: modelID,
            kvBits: 8,
            kvGroupSize: 64,
            modelFingerprint: fingerprint,
            triAttention: triAttention
        )
    }

    /// `.dense` must digest to the same value the pre-TriAttention
    /// canonicalization produced. Any drift silently orphans every
    /// dense snapshot persisted before TriAttention support landed.
    @Test
    func partitionDigestDenseMatchesPreTriAttentionWirePin() {
        let explicitDense = makeKey(triAttention: .dense)
        let defaulted = makeKey()
        #expect(explicitDense.partitionDigest == "ecfce886")
        #expect(defaulted.partitionDigest == "ecfce886")
        #expect(explicitDense == defaulted)
    }

    @Test
    func partitionDigestTriAttentionSplitsFromDense() {
        let base = makeKey()
        let triAttention = makeKey(
            triAttention: makeTriAttentionIdentity(artifact: "cafef00d")
        )
        #expect(base.partitionDigest != triAttention.partitionDigest)
        #expect(base != triAttention)
    }

    @Test
    func partitionDigestDistinguishesTriAttentionFields() {
        let base = makeKey(
            modelID: "m",
            fingerprint: "f",
            triAttention: makeTriAttentionIdentity()
        )
        let diffs: [(label: String, identity: TriAttentionPartitionIdentity)] = [
            ("budgetTokens", makeTriAttentionIdentity(budget: 8_000)),
            ("calibrationArtifactIdentity", makeTriAttentionIdentity(artifact: "bbb")),
            ("calibrationArtifactIdentity nil vs Some", makeTriAttentionIdentity(artifact: nil)),
        ]
        for diff in diffs {
            let other = makeKey(
                modelID: "m",
                fingerprint: "f",
                triAttention: diff.identity
            )
            #expect(
                base.partitionDigest != other.partitionDigest,
                "\(diff.label) does not affect the TriAttention digest"
            )
            #expect(base != other, "\(diff.label) does not affect equality")
        }
    }

    @Test
    func partitionKeySortsDenseBeforeTriAttention() {
        let dense = makeKey(modelID: "m", fingerprint: "f")
        let triAttention = makeKey(
            modelID: "m", fingerprint: "f",
            triAttention: makeTriAttentionIdentity()
        )
        #expect(dense < triAttention)
        #expect(!(triAttention < dense))

        let smallerBudget = makeKey(
            modelID: "m", fingerprint: "f",
            triAttention: makeTriAttentionIdentity(budget: 8_000)
        )
        #expect(smallerBudget < triAttention)
    }

    /// `enabled == false` must collapse onto `.dense` regardless of
    /// the carried budget/artifact/impl fields; otherwise dense
    /// partitions fragment whenever the runtime selector forwards a
    /// non-default budget through a disabled configuration.
    @Test
    func triAttentionPartitionIdentityFromConfigurationCollapsesDisabled() {
        #expect(TriAttentionPartitionIdentity.from(.v1Disabled) == .dense)

        let disabledWithFields = TriAttentionPartitionIdentity.from(
            TriAttentionConfiguration(
                enabled: false,
                budgetTokens: 8_000,
                calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity(
                    rawValue: "cafebabe"
                ),
                implementationVersion: .v1
            )
        )
        #expect(disabledWithFields == .dense)

        let artifact = TriAttentionCalibrationArtifactIdentity(rawValue: "cafebabe")
        let enabled = TriAttentionPartitionIdentity.from(
            TriAttentionConfiguration(
                enabled: true,
                budgetTokens: 12_000,
                calibrationArtifactIdentity: artifact,
                implementationVersion: .v1
            )
        )
        #expect(
            enabled == .triAttention(
                budgetTokens: 12_000,
                calibrationArtifactIdentity: artifact,
                implementationVersion: .v1
            )
        )
    }

    // MARK: - Schema-version compatibility

    @Test
    func manifestIsSchemaCompatibleAtCurrentVersion() {
        var manifest = SnapshotManifest.empty()
        #expect(manifest.isSchemaCompatible)

        manifest.schemaVersion = SnapshotManifestSchema.currentVersion + 1
        #expect(!manifest.isSchemaCompatible)

        manifest.schemaVersion = 0
        #expect(!manifest.isSchemaCompatible)

        manifest.schemaVersion = SnapshotManifestSchema.currentVersion
        #expect(manifest.isSchemaCompatible)
    }

    @Test
    func manifestDecodesOlderSchemaWithoutThrowing() throws {
        // Codable must not throw on a schema mismatch. Warm start
        // is responsible for the wipe/rebuild decision; the data
        // model's job is to expose the mismatch as a structured
        // signal (`isSchemaCompatible`), not to fail deserialization.
        // Throwing here would force warm start to wrap every
        // decode in a try/catch and rely on the error message to
        // distinguish "corrupt JSON" from "old schema", which is
        // fragile.
        let oldJSON = #"{"schemaVersion":0,"partitions":{},"snapshots":{}}"#
            .data(using: .utf8)!
        let decoded = try JSONDecoder().decode(SnapshotManifest.self, from: oldJSON)
        #expect(decoded.schemaVersion == 0)
        #expect(!decoded.isSchemaCompatible)
        #expect(decoded.partitions.isEmpty)
        #expect(decoded.snapshots.isEmpty)
    }

    @Test
    func manifestDecodesFutureSchemaWithoutThrowing() throws {
        // Forward-compat: a manifest written by a newer schema must
        // still decode cleanly so warm start can detect and wipe it,
        // rather than throwing an opaque decode error that the
        // warm-start path has to pattern-match.
        let futureJSON = #"{"schemaVersion":999,"partitions":{},"snapshots":{}}"#
            .data(using: .utf8)!
        let decoded = try JSONDecoder().decode(SnapshotManifest.self, from: futureJSON)
        #expect(decoded.schemaVersion == 999)
        #expect(!decoded.isSchemaCompatible)
    }

    @Test
    func manifestRejectsCorruptStructureNotVersionMismatch() throws {
        // Structural corruption IS allowed to throw — this is what
        // separates "old schema, wipe it" (caught by decode +
        // isSchemaCompatible==false) from "bytes are garbage, also
        // wipe it but log differently" (caught by a decode throw).
        // Warm start distinguishes these two cases to tell the
        // operator whether the manifest was written by a different
        // version or whether something trampled the file.
        let brokenJSON = #"{"schemaVersion":"not-an-int","partitions":{},"snapshots":{}}"#
            .data(using: .utf8)!
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(SnapshotManifest.self, from: brokenJSON)
        }
    }

    @Test
    func storageRefEqualityIgnoresNothing() {
        // All fields participate in `==` because `SnapshotStorageRef`
        // gets synthesized Equatable. Confirm by flipping `committed`
        // on two otherwise-identical values.
        let instant: ContinuousClock.Instant = .now
        let a = SnapshotStorageRef(
            snapshotID: "x",
            partitionDigest: "y",
            tokenOffset: 0,
            checkpointType: .leaf,
            bytesOnDisk: 1,
            lastAccessTime: instant,
            committed: false
        )
        let b = SnapshotStorageRef(
            snapshotID: "x",
            partitionDigest: "y",
            tokenOffset: 0,
            checkpointType: .leaf,
            bytesOnDisk: 1,
            lastAccessTime: instant,
            committed: false
        )
        #expect(a == b)

        let c = SnapshotStorageRef(
            snapshotID: "x",
            partitionDigest: "y",
            tokenOffset: 0,
            checkpointType: .leaf,
            bytesOnDisk: 1,
            lastAccessTime: instant,
            committed: true  // differs
        )
        #expect(a != c)
    }
}
