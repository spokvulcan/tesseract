import CryptoKit
import Foundation
import MLX
import MLXLMCommon

/// A fixed 2K, unquantized-KV gate. The full default correctness matrix and
/// speculative HTTP replay remain separate acceptance checks.
nonisolated enum BoundedCacheParity {
    private struct Continuation {
        let state: CacheStateBytes
        let logits: Data
    }

    static func run(context: ModelContext, diskRoot: URL) async throws
        -> [BenchmarkHarness.CheckResult]
    {
        let tokens = BenchmarkHarness.promptTokens(targetTokens: 2048, tokenizer: context.tokenizer)
        let offset = 1024
        var cache = try context.model.newCache(parameters: nil)
        try prefill(context, tokens: Array(tokens.prefix(offset)), base: 0, cache: cache)
        let prefix = CacheStateBytes(cache)
        guard let copied = HybridCacheSnapshot.capture(cache: cache, offset: offset, type: .leaf),
            let moved = HybridCacheSnapshot.captureMoving(cache: &cache, offset: offset)
        else { throw HybridCacheCorrectnessError.snapshotCaptureFailed }

        let cold = try continuation(
            context, tokens: tokens, base: 0,
            cache: context.model.newCache(parameters: nil))
        var checks: [BenchmarkHarness.CheckResult] = []
        func checkState(_ name: String, _ actual: CacheStateBytes, _ expected: CacheStateBytes) {
            checks.append(
                .init(
                    name: name, passed: expected.byteCount > 0 && actual == expected,
                    detail: "layers=\(actual.layers.count) stateBytes=\(actual.byteCount) "
                        + "expectedBytes=\(expected.byteCount) metadataAndBytesEqual=\(actual == expected)"
                ))
        }
        func checkContinuation(_ name: String, _ actual: Continuation, _ expected: Continuation) {
            checkState(name + "State", actual.state, expected.state)
            checks.append(
                .init(
                    name: name + "Logits",
                    passed: !actual.logits.isEmpty && actual.logits == expected.logits,
                    detail:
                        "logitsBytes=\(actual.logits.count) bitwiseEqual=\(actual.logits == expected.logits) "
                        + "sha256=\(SHA256.hash(data: actual.logits).map { String(format: "%02x", $0) }.joined())"
                ))
        }
        checkState("copiedPrefix", CacheStateBytes(try copied.restore()), prefix)
        checkState("movedPrefixRestoredByCopy", CacheStateBytes(try moved.restore()), prefix)
        checkContinuation(
            "copiedContinuation",
            try continuation(
                context, tokens: tokens, base: offset, cache: copied.restore()), cold)

        let (manager, checkedOut, key) = try await checkout(moved, tokens: tokens, offset: offset)
        let owner = checkedOut.live
        checks.append(
            .init(
                name: "checkoutOwnership", passed: cache.isEmpty && moved.layers.isEmpty,
                detail: "sourceLayers=\(cache.count) treeBodyLayers=\(moved.layers.count)"))
        checkState("checkedOutPrefix", CacheStateBytes(owner.cache), prefix)
        checkContinuation(
            "handoffContinuation",
            try continuation(
                context, tokens: tokens, base: offset, cache: owner.cache), cold)
        guard
            let head = HybridCacheSnapshot.capture(
                cache: owner.cache, offset: tokens.count, type: .leaf)
        else { throw HybridCacheCorrectnessError.snapshotCaptureFailed }
        // The production rewind, compaction included (the Cache Claim's own).
        _ = await checkedOut.returnByRewind()
        guard let rewound = await manager.lookup(tokens: tokens, partitionKey: key).snapshot
        else { throw HybridCacheCorrectnessError.snapshotCaptureFailed }
        let leaseCount = await MainActor.run { checkedOut.grant.tree.leaseCount }
        checks.append(
            .init(
                name: "rewindOwnership",
                passed: owner.cache.isEmpty && leaseCount == 0,
                detail: "requestLayers=\(owner.cache.count) treeLeases=\(leaseCount)"))
        checkState("rewoundPrefix", CacheStateBytes(try rewound.restore()), prefix)
        checkContinuation(
            "rewoundContinuation",
            try continuation(
                context, tokens: tokens, base: offset, cache: rewound.restore()), cold)

        try await withScratchStore(diskRoot: diskRoot) { store in
            let fingerprint = String(repeating: "a", count: 64)
            let digest = "bounded-parity"
            store.registerPartition(
                .init(
                    modelID: key.modelID, modelFingerprint: fingerprint, kvBits: nil,
                    kvGroupSize: 64,
                    createdAt: 100, schemaVersion: SnapshotManifestSchema.currentVersion),
                digest: digest)
            let full = try await roundTrip(
                store, snapshot: copied, id: "base", tokens: tokens,
                digest: digest, fingerprint: fingerprint, extending: nil)
            checkState("ssdFullPrefix", CacheStateBytes(try full.restore()), prefix)
            checkContinuation(
                "ssdFullContinuation",
                try continuation(
                    context, tokens: tokens, base: offset, cache: full.restore()), cold)
            let extensionHead = try await roundTrip(
                store, snapshot: head, id: "head", tokens: tokens, digest: digest,
                fingerprint: fingerprint,
                extending: .init(baseSnapshotID: "base", baseOffset: offset))
            checkState(
                "ssdExtensionState", CacheStateBytes(try extensionHead.restore()), cold.state)
            let sentinelTokens = tokens + [tokens.last!]
            checkContinuation(
                "ssdExtensionSentinel",
                try continuation(
                    context, tokens: sentinelTokens, base: tokens.count,
                    cache: extensionHead.restore()),
                try continuation(
                    context, tokens: sentinelTokens, base: tokens.count, cache: head.restore()))
        }
        return checks
    }

    /// Own the temporary store through its final writer drain and directory removal.
    static func withScratchStore(
        diskRoot: URL, operation: (SSDSnapshotStore) async throws -> Void
    ) async throws {
        let store = SSDSnapshotStore(
            config: .init(
                enabled: true, rootURL: diskRoot, budgetBytes: 1 << 30, maxPendingBytes: 1 << 30))
        let outcome: Result<Void, Error>
        do {
            try await operation(store)
            outcome = .success(())
        } catch {
            outcome = .failure(error)
        }
        // A synchronous defer cannot await the writer. Capture failures first
        // so every cooperative exit drains before removing model cache tensors.
        await store.flushAsync()
        if FileManager.default.fileExists(atPath: diskRoot.path) {
            try FileManager.default.removeItem(at: diskRoot)
        }
        try outcome.get()
    }

    @MainActor
    private static func checkout(_ snapshot: HybridCacheSnapshot, tokens: [Int], offset: Int)
        async throws
        -> (PrefixCacheManager, CheckedOutLeaf, CachePartitionKey)
    {
        let manager = PrefixCacheManager(memoryBudgetBytes: 1 << 30)
        let key = CachePartitionKey(modelID: "bounded-parity", kvBits: nil, kvGroupSize: 64)
        guard
            let admission = SnapshotAdmission.leaf(
                storedTokens: Array(tokens.prefix(offset)), snapshot: snapshot, storage: .ramOnly,
                partitionKey: key)
        else { throw HybridCacheCorrectnessError.snapshotCaptureFailed }
        manager.admit(admission)
        guard let resolved = manager.lookup(tokens: tokens, partitionKey: key).snapshot else {
            throw HybridCacheCorrectnessError.snapshotCaptureFailed
        }
        let outcome = manager.leaseLeaf(
            snapshot: resolved, tokens: tokens, partitionKey: key,
            bodyRefusal: resolved.checkoutRefusal(maximumAdvance: tokens.count - offset + 1),
            context: .init(requestID: UUID(), modelID: key.modelID, kvBits: nil, kvGroupSize: 64))
        guard case .leased(let grant) = outcome else {
            throw HybridCacheCorrectnessError.verificationFailed(
                failedChecks: ["checkout fell back: \(outcome)"])
        }
        // The production check-out's move, outside a Cache Claim: the bench
        // checks the tree and the move, not the claim's lifecycle.
        return (manager, CheckedOutLeaf.take(resolved, under: grant), key)
    }

    private static func prefill(
        _ context: ModelContext, tokens: [Int], base: Int, cache: [any KVCache]
    ) throws {
        guard !tokens.isEmpty else { return }
        _ = try PrefillExecutor.run(
            model: context.model, text: .init(tokens: MLXArray(tokens.map(Int32.init)), mask: nil),
            cache: cache, checkpoints: [:], checkpointBaseOffset: base,
            prefillStepSize: 512, consumeAll: true)
    }

    private static func continuation(
        _ context: ModelContext, tokens: [Int], base: Int,
        cache: [any KVCache]
    ) throws -> Continuation {
        try prefill(
            context, tokens: Array(tokens[base..<(tokens.count - 1)]), base: base, cache: cache)
        let input = MLXArray([Int32(tokens.last!)]).expandedDimensions(axis: 0)
        let logits = context.model(.init(tokens: input, mask: nil), cache: cache, state: nil)
            .logits[0, 0]
        eval(logits, cache)
        return Continuation(
            state: CacheStateBytes(cache), logits: logits.asData(access: .copy).data)
    }

    private static func roundTrip(
        _ store: SSDSnapshotStore, snapshot: HybridCacheSnapshot,
        id: String, tokens: [Int], digest: String, fingerprint: String,
        extending: SnapshotExtension?
    ) async throws -> HybridCacheSnapshot {
        let payload = ServerCompletion.extractSnapshotPayload(snapshot, extending: extending)
        guard payload.extending == extending else {
            throw HybridCacheCorrectnessError.verificationFailed(failedChecks: [
                "extension degraded to full"
            ])
        }
        let descriptor = PersistedSnapshotDescriptor(
            snapshotID: id, partitionDigest: digest,
            pathFromRoot: Array(tokens.prefix(snapshot.tokenOffset)),
            tokenOffset: snapshot.tokenOffset, checkpointType: "leaf", bytes: payload.totalBytes,
            segmentBaseOffset: extending?.baseOffset ?? 0, inheritedSegments: [], createdAt: 100,
            lastAccessAt: 100,
            fileRelativePath: PersistedSnapshotDescriptor.relativeFilePath(
                snapshotID: id, partitionDigest: digest),
            schemaVersion: SnapshotManifestSchema.currentVersion)
        guard case .accepted = store.tryEnqueue(payload: payload, descriptor: descriptor) else {
            throw HybridCacheCorrectnessError.verificationFailed(failedChecks: [
                "SSD enqueue rejected"
            ])
        }
        await store.flushAsync()
        guard let committed = store.residency().descriptor(id: id),
            let hydrated = store.loadSync(
                snapshotRef: .init(
                    snapshotID: id, partitionDigest: digest, tokenOffset: snapshot.tokenOffset,
                    checkpointType: .leaf, bytesOnDisk: committed.totalBytes),
                expectedFingerprint: fingerprint)
        else {
            throw HybridCacheCorrectnessError.verificationFailed(failedChecks: [
                "SSD commit/restore failed"
            ])
        }
        return hydrated
    }
}
