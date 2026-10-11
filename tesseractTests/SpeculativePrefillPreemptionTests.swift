import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// PR B follow-up (PRD #137, user story 10): the **Speculative Canonical
/// Prefill** preemption interleaving, pinned over the toy model. A real
/// pass — probe → Snapshot Resolution → restore → chunked extension
/// prefill — is cancelled mid-span at a deterministic chunk boundary; the
/// settle must admit the partial progress (past the 2,048-token capture
/// threshold) as a RAM-only leaf, and a subsequent Snapshot Resolution —
/// what a preempting request performs — must surface that admission instead
/// of the shallower boundary it restored from.
@MainActor
@Suite struct SpeculativePrefillPreemptionTests {

    @Test(arguments: ["planned", "transient", "leased", "departed"], [false, true])
    func speculativeViewRestoresOrReprefillsAfterBackingLeafDeparture(
        backingState: String, ramOnlySpine: Bool
    )
        async throws
    {
        let tokenizer = ToySequencingTokenizer()
        let stored = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [
                .init(role: .user, content: String(repeating: "a", count: 2200)),
                .assistant(content: "Done"),
            ])
        let render = ConversationRender.uncached(tokenizer: tokenizer)
        let future = try #require(
            try LeafAdmissionBuilder.futureSharedPrefix(
                storedConversation: stored, keySpace: .identity(keyPath: []), render: render)?.get()
        )
        let path = try #require(
            SpeculativeCanonicalPrefill.admitPath(
                futureSharedPrefix: future, canonicalLeafOffset: 0))
        let provider = ToyModelSessionProvider(
            model: ToyLanguageModel(script: [0]), tokenizer: tokenizer)
        let prefix = Array(path.prefix(4))
        let backerPath = prefix + [42, 42, 42, 42]
        let (system, view, leaf) = try await provider.withSession { session in
            let cache = try session.newCache(parameters: GenerateParameters())
            _ = try session.prefill(
                text: .init(tokens: MLXArray(prefix.prefix(2).map(Int32.init))),
                cache: cache, checkpoints: [:], checkpointBaseOffset: 0, prefillStepSize: 2,
                consumeAll: true, initialState: nil, evalPolicy: .pipelined, storedForm: nil)
            let system = try #require(
                session.captureSnapshot(cache: cache, offset: 2, type: .system, storedForm: nil))
            _ = try session.prefill(
                text: .init(tokens: MLXArray(prefix.suffix(2).map(Int32.init))),
                cache: cache, checkpoints: [:], checkpointBaseOffset: 2, prefillStepSize: 2,
                consumeAll: true, initialState: nil, evalPolicy: .pipelined, storedForm: nil)
            let view = try #require(
                session.captureSnapshot(
                    cache: cache, offset: 4, type: .branchPoint, storedForm: nil))
            _ = try session.prefill(
                text: .init(tokens: MLXArray([Int32(42), 42, 42, 42])),
                cache: cache, checkpoints: [:], checkpointBaseOffset: 4, prefillStepSize: 4,
                consumeAll: true, initialState: nil, evalPolicy: .pipelined, storedForm: nil)
            return (
                system, view,
                try #require(
                    session.captureSnapshot(cache: cache, offset: 8, type: .leaf, storedForm: nil))
            )
        }
        let store = TieredSnapshotStore(ssdConfig: nil)
        let manager = PrefixCacheManager(memoryBudgetBytes: 1 << 20, tieredStore: store)
        let key = CachePartitionKey(
            modelID: "toy/view-speculation-\(UUID())", kvBits: nil, kvGroupSize: 64)
        let telemetry = TelemetryCapture(modelID: key.modelID)
        defer { telemetry.stop() }
        manager.restoreSnapshot(
            path: Array(prefix.prefix(2)), snapshot: system,
            partitionKey: key, lastAccessTime: .now)
        if backingState == "planned" {
            manager.restoreSnapshot(
                path: prefix, snapshot: view, partitionKey: key, lastAccessTime: .now)
        }
        manager.restoreSnapshot(
            path: backerPath, snapshot: leaf, partitionKey: key, lastAccessTime: .now)
        let diagnostics = PrefixCacheDiagnostics.Context(
            requestID: UUID(), modelID: key.modelID, kvBits: nil, kvGroupSize: 64)
        let tree = try #require(store.tree(for: key))
        let backer = try #require(tree.findBestSnapshot(tokens: backerPath)?.node)
        var lease: LeafLease?
        if backingState == "leased" {
            let claim = try #require(tree.beginLeafLease(on: backer, context: diagnostics))
            lease = claim
            #expect(tree.takeLeasedBody(claim, on: backer) != nil)
        } else if backingState == "departed" {
            _ = tree.dropBody(node: backer)
        }
        // The request whose answer `stored` records, keyed as Server
        // Completion keys it: its key path begins with the view's `prefix`.
        let request = try await ToyRequestKeying.keyedRequest(
            provider: provider,
            conversation: HTTPPrefixCacheConversation(
                systemPrompt: nil, messages: Array(stored.messages.dropLast())),
            modelID: key.modelID, prefillStepSize: 256)
        #expect(Array(request.keySpace.keyPath.prefix(prefix.count)) == prefix)
        let seed = SpeculativeCanonicalPrefill.makeSeed(
            storedConversation: stored, request: request, canonicalLeafOffset: 0,
            transientBoundary: backingState == "planned" ? nil : view,
            ramOnlySpine: ramOnlySpine, diagnostics: diagnostics)
        await SpeculativeCanonicalPrefill.run(
            seed: seed, sessions: provider, prefixCache: manager)
        let unavailable = backingState == "leased" || backingState == "departed"
        let events = telemetry.drain()
        let admitted = try #require(events.last { $0.eventName == "speculativePrefill" })
        #expect(admitted.intField("boundaryOffset") == (unavailable ? 2 : 4))
        #expect(admitted.intField("targetOffset") == path.count)
        let fallback = events.first { $0.field("fallback") == "boundaryReprefill" }
        #expect((fallback != nil) == unavailable)
        if unavailable {
            #expect(fallback?.field("reason") == "no-backing-leaf")
            #expect(fallback?.intField("requestedOffset") == 4)
            #expect(fallback?.intField("restoredOffset") == 2)
        }
        let restored = try #require(manager.lookup(tokens: path, partitionKey: key).snapshot)
        #expect(restored.tokenOffset == path.count)
        // Same admitted path and actual KV rows in both the view and fallback arms.
        let restoredBytes = restored.layers[0].state[0].asArray(Float.self)
        #expect(restoredBytes == path.flatMap { Array(repeating: Float($0), count: 4) })
        if let lease {
            #expect(
                tree.endLeafLease(
                    lease, on: backer, returning: leaf, tokens: backerPath, reason: .rewind))
        }
        _ = manager.clearRAMTier()
        #expect(manager.totalSnapshotBytes == 0)
    }

    @Test(.parksPoolThreads)
    func preemptedPassSettlesPartialLeafThatResolutionThenSurfaces() async throws {
        let tokenizer = ToySequencingTokenizer()
        let stored = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [
                HTTPPrefixCacheMessage(role: .user, content: String(repeating: "a", count: 4000)),
                .assistant(content: "Done"),
            ]
        )

        // The pass's own probe machinery computes the future shared path.
        let probed = try LeafAdmissionBuilder.futureSharedPrefix(
            storedConversation: stored,
            keySpace: .identity(keyPath: []),
            render: ConversationRender.uncached(tokenizer: tokenizer)
        )
        let futurePrefix = try #require(try probed?.get())
        let admitPath = try #require(
            SpeculativeCanonicalPrefill.admitPath(
                futureSharedPrefix: futurePrefix,
                canonicalLeafOffset: 0
            )
        )
        #expect(admitPath.count > 3072)

        // Pause the pass's second extension chunk (restore boundary 1024 +
        // one completed 1024-token chunk ⇒ the blocked forward starts at
        // offset 2048), so the cancel lands mid-span with exactly the
        // 2,048-token capture threshold consumed.
        let gate = ForwardGate(threshold: 2048)
        let model = ToyLanguageModel(script: [0], onForward: gate.onForward)
        let records = model.capacityRecords
        let provider = ToyModelSessionProvider(
            model: model,
            tokenizer: tokenizer
        )
        let manager = PrefixCacheManager(memoryBudgetBytes: 1 << 30)
        let partitionKey = CachePartitionKey(modelID: "toy/model", kvBits: nil, kvGroupSize: 64)
        let diagnostics = PrefixCacheDiagnostics.Context(
            requestID: UUID(), modelID: "toy/model", kvBits: nil, kvGroupSize: 64
        )

        // Arrange the boundary the pass restores from: real rows for the
        // path's first 1,024 tokens, captured through the session verbs and
        // admitted as a leaf.
        let boundaryTokens = Array(admitPath[0..<1024])
        let boundaryLeaf = try await provider.withSession { session -> HybridCacheSnapshot? in
            let cache = try session.newCache(parameters: GenerateParameters(temperature: 0))
            _ = try session.prefill(
                text: .init(tokens: MLXArray(boundaryTokens.map(Int32.init)), mask: nil),
                cache: cache,
                checkpoints: [:],
                checkpointBaseOffset: 0,
                prefillStepSize: 1024,
                consumeAll: true,
                initialState: nil,
                evalPolicy: .pipelined,
                storedForm: nil
            )
            return session.captureSnapshot(
                cache: cache, offset: boundaryTokens.count, type: .leaf, storedForm: nil)
        }
        let boundarySnapshot = try #require(boundaryLeaf)
        let boundaryAdmission = try #require(
            SnapshotAdmission.leaf(
                storedTokens: boundaryTokens,
                snapshot: boundarySnapshot,
                storage: .ramOnly,
                partitionKey: partitionKey,
                requestID: UUID()
            )
        )
        _ = manager.admit(boundaryAdmission)

        let request = try await ToyRequestKeying.keyedRequest(
            provider: provider,
            conversation: HTTPPrefixCacheConversation(
                systemPrompt: nil, messages: Array(stored.messages.dropLast())),
            prefillStepSize: 1024)
        #expect(request.facts.partitionKey == partitionKey)
        let seed = SpeculativeCanonicalPrefill.makeSeed(
            storedConversation: stored,
            request: request,
            canonicalLeafOffset: 0,
            diagnostics: diagnostics
        )

        let sessions = provider
        let pass = Task {
            await SpeculativeCanonicalPrefill.run(
                seed: seed, sessions: sessions, prefixCache: manager
            )
        }
        await gate.reached()
        pass.cancel()
        gate.open()
        await pass.value

        // The preempting request's Snapshot Resolution must surface the
        // settled partial leaf — boundary + the two completed chunks — not
        // the 1,024-token boundary it would otherwise re-prefill from.
        let resolved = await manager.resolve(
            tokens: admitPath,
            promptTokenCount: admitPath.count,
            partitionKey: partitionKey,
            modelFingerprint: nil,
            diagnostics: diagnostics
        )
        let snapshot = try #require(resolved.lookup.snapshot)
        #expect(snapshot.tokenOffset == 1024 + 2048)
        let extensionRecords = records.values.filter { $0.offset > 1024 }
        #expect(extensionRecords.count == 2)
        #expect(extensionRecords.allSatisfy { $0.capacity >= admitPath.count })
        #expect(extensionRecords.allSatisfy { $0.capacity < admitPath.count + 256 })
        #expect(Set(extensionRecords.map(\.capacity)).count == 1)
    }

    /// The failed-restore policy covers the speculative pass too (ADR-0094's
    /// declared change): a boundary whose restore throws is dropped through
    /// the manager, so no later request resolves to it and pays the same
    /// failure, and the pass logs `unrestorable-snapshot` before its own
    /// `restore-failed`. Nothing is prefilled or admitted.
    @Test func aBoundaryThatFailsToRestoreIsDropped() async throws {
        let tokenizer = ToySequencingTokenizer()
        let stored = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [
                HTTPPrefixCacheMessage(role: .user, content: String(repeating: "a", count: 2200)),
                .assistant(content: "Done"),
            ])
        let future = try #require(
            try LeafAdmissionBuilder.futureSharedPrefix(
                storedConversation: stored, keySpace: .identity(keyPath: []),
                render: ConversationRender.uncached(tokenizer: tokenizer))?.get())
        let admitPath = try #require(
            SpeculativeCanonicalPrefill.admitPath(
                futureSharedPrefix: future, canonicalLeafOffset: 0))
        let forwards = ForwardLog()
        let fault = ToyRestoreFault()
        let provider = ToyModelSessionProvider(
            model: ToyLanguageModel(script: [0], onForward: forwards.onForward),
            tokenizer: tokenizer, restoreFault: fault)
        let partitionKey = CachePartitionKey(
            modelID: "toy/unrestorable-\(UUID())", kvBits: nil, kvGroupSize: 64)
        let telemetry = TelemetryCapture(modelID: partitionKey.modelID)
        defer { telemetry.stop() }
        let manager = PrefixCacheManager(memoryBudgetBytes: 1 << 30)
        let diagnostics = PrefixCacheDiagnostics.Context(
            requestID: UUID(), modelID: partitionKey.modelID, kvBits: nil, kvGroupSize: 64)
        let boundaryTokens = Array(admitPath[0..<1024])
        let boundary = try #require(
            try await provider.withSession { session -> HybridCacheSnapshot? in
                let cache = try session.newCache(parameters: GenerateParameters(temperature: 0))
                _ = try session.prefill(
                    text: .init(tokens: MLXArray(boundaryTokens.map(Int32.init)), mask: nil),
                    cache: cache, checkpoints: [:], checkpointBaseOffset: 0,
                    prefillStepSize: 1024, consumeAll: true, initialState: nil,
                    evalPolicy: .pipelined, storedForm: nil)
                return session.captureSnapshot(
                    cache: cache, offset: boundaryTokens.count, type: .leaf, storedForm: nil)
            })
        _ = manager.admit(
            try #require(
                SnapshotAdmission.leaf(
                    storedTokens: boundaryTokens, snapshot: boundary, storage: .ramOnly,
                    partitionKey: partitionKey, requestID: UUID())))
        let request = try await ToyRequestKeying.keyedRequest(
            provider: provider,
            conversation: HTTPPrefixCacheConversation(
                systemPrompt: nil, messages: Array(stored.messages.dropLast())),
            modelID: partitionKey.modelID, prefillStepSize: 1024)
        let seed = SpeculativeCanonicalPrefill.makeSeed(
            storedConversation: stored, request: request, canonicalLeafOffset: 0,
            diagnostics: diagnostics)
        let forwardsBefore = forwards.offsets.count
        fault.arm()

        await SpeculativeCanonicalPrefill.run(seed: seed, sessions: provider, prefixCache: manager)

        #expect(fault.failedBodyIDs == [boundary.bodyID])
        let skips = telemetry.drain().filter { $0.eventName == "skip" }
        #expect(
            skips.map { "\($0.field("stage") ?? "")/\($0.field("reason") ?? "")" }
                == ["restore/unrestorable-snapshot", "speculativePrefill/restore-failed"])
        #expect(skips.first?.field("drop") == "body")
        #expect(skips.first?.intField("offset") == 1024)
        #expect(forwards.offsets.count == forwardsBefore)
        #expect(manager.lookup(tokens: admitPath, partitionKey: partitionKey).snapshot == nil)
    }

    // MARK: - The chunk loop (ADR-0094)

    /// A pass over a 1,024-token boundary of a long admit path, its chunks
    /// 1,024 tokens: what the chunk-loop cases share.
    @MainActor
    private struct ChunkedPass {
        let provider: ToyModelSessionProvider
        let manager: PrefixCacheManager
        let seed: SpeculativeCanonicalPrefill.Seed
        let admitPath: [Int]
        let partitionKey: CachePartitionKey
        let telemetry: TelemetryCapture

        init(
            onForward: (@Sendable (Int) -> Void)? = nil, prefillFault: ToyPrefillFault? = nil,
            sessionRequests: SessionRequestLog? = nil
        ) async throws {
            let tokenizer = ToySequencingTokenizer()
            let stored = HTTPPrefixCacheConversation(
                systemPrompt: nil,
                messages: [
                    HTTPPrefixCacheMessage(
                        role: .user, content: String(repeating: "a", count: 4000)),
                    .assistant(content: "Done"),
                ])
            let future = try #require(
                try LeafAdmissionBuilder.futureSharedPrefix(
                    storedConversation: stored, keySpace: .identity(keyPath: []),
                    render: ConversationRender.uncached(tokenizer: tokenizer))?.get())
            admitPath = try #require(
                SpeculativeCanonicalPrefill.admitPath(
                    futureSharedPrefix: future, canonicalLeafOffset: 0))
            provider = ToyModelSessionProvider(
                model: ToyLanguageModel(script: [0], onForward: onForward),
                tokenizer: tokenizer, prefillFault: prefillFault, sessionRequests: sessionRequests)
            partitionKey = CachePartitionKey(
                modelID: "toy/chunk-loop-\(UUID())", kvBits: nil, kvGroupSize: 64)
            telemetry = TelemetryCapture(modelID: partitionKey.modelID)
            manager = PrefixCacheManager(memoryBudgetBytes: 1 << 30)
            let diagnostics = PrefixCacheDiagnostics.Context(
                requestID: UUID(), modelID: partitionKey.modelID, kvBits: nil, kvGroupSize: 64)
            let boundaryTokens = Array(admitPath[0..<1024])
            let boundary = try #require(
                try await provider.withSession { session -> HybridCacheSnapshot? in
                    let cache = try session.newCache(parameters: GenerateParameters(temperature: 0))
                    _ = try session.prefill(
                        text: .init(tokens: MLXArray(boundaryTokens.map(Int32.init)), mask: nil),
                        cache: cache, checkpoints: [:], checkpointBaseOffset: 0,
                        prefillStepSize: 1024, consumeAll: true, initialState: nil,
                        evalPolicy: .pipelined, storedForm: nil)
                    return session.captureSnapshot(
                        cache: cache, offset: boundaryTokens.count, type: .leaf, storedForm: nil)
                })
            _ = manager.admit(
                try #require(
                    SnapshotAdmission.leaf(
                        storedTokens: boundaryTokens, snapshot: boundary, storage: .ramOnly,
                        partitionKey: partitionKey, requestID: UUID())))
            let request = try await ToyRequestKeying.keyedRequest(
                provider: provider,
                conversation: HTTPPrefixCacheConversation(
                    systemPrompt: nil, messages: Array(stored.messages.dropLast())),
                modelID: partitionKey.modelID, prefillStepSize: 1024)
            seed = SpeculativeCanonicalPrefill.makeSeed(
                storedConversation: stored, request: request, canonicalLeafOffset: 0,
                diagnostics: diagnostics)
        }

        func deepest() -> Int? {
            manager.lookup(tokens: admitPath, partitionKey: partitionKey).snapshot?.tokenOffset
        }
    }

    /// Each chunk is one `prefill` verb, so each ends with the executor's
    /// checked flush and its `Memory.clearCache()`: the pool is returned
    /// between chunks. And each evaluates checked, so an MLX failure inside a
    /// chunk throws instead of reaching MLX's process-fatal handler (ADR-0094's
    /// declared changes).
    @Test func eachChunkIsOneCheckedPrefill() async throws {
        let pass = try await ChunkedPass()
        defer { pass.telemetry.stop() }
        let before = pass.provider.recorder.prefillEvalPolicies.count

        await SpeculativeCanonicalPrefill.run(
            seed: pass.seed, sessions: pass.provider, prefixCache: pass.manager)

        let policies = Array(pass.provider.recorder.prefillEvalPolicies[before...])
        let residual = pass.admitPath.count - 1024
        #expect(policies.count == (residual + 1023) / 1024)
        #expect(policies.allSatisfy { $0 == .checkedSynchronous }, "\(policies)")
        #expect(pass.deepest() == pass.admitPath.count)
    }

    /// A chunk whose prefill throws (the checked MLX error's shape) ends the
    /// pass with `speculativePrefill/prefill-threw`, and nothing is admitted.
    @Test func aChunkThatThrowsEndsThePassWithNothingAdmitted() async throws {
        let fault = ToyPrefillFault()
        let pass = try await ChunkedPass(prefillFault: fault)
        defer { pass.telemetry.stop() }
        _ = pass.telemetry.drain()
        fault.arm()

        await SpeculativeCanonicalPrefill.run(
            seed: pass.seed, sessions: pass.provider, prefixCache: pass.manager)

        let skips = pass.telemetry.drain().filter { $0.eventName == "skip" }
        #expect(
            skips.map { "\($0.field("stage") ?? "")/\($0.field("reason") ?? "")" }
                == ["speculativePrefill/prefill-threw"])
        #expect(pass.provider.recorder.prefillEvalPolicies.last == .checkedSynchronous)
        #expect(pass.deepest() == 1024)
    }

    /// A preemption that lands while a chunk waits for its session stops the
    /// pass at the next check, not before that chunk: the chunk runs, as the
    /// pass's chunks always have (ADR-0009's one-chunk bound), and the settle
    /// admits it. The test holds the session while the pass's third chunk
    /// waits for it, cancels the pass, then lets the chunk in.
    @Test(.parksPoolThreads)
    func aPreemptionWhileAChunkWaitsForItsSessionStillRunsThatChunk() async throws {
        let gate = ForwardGate(threshold: 2048)
        let requests = SessionRequestLog()
        let pass = try await ChunkedPass(onForward: gate.onForward, sessionRequests: requests)
        defer { pass.telemetry.stop() }
        #expect(pass.admitPath.count > 3072)
        let provider = pass.provider
        let seed = pass.seed
        let manager = pass.manager
        let task = Task {
            await SpeculativeCanonicalPrefill.run(
                seed: seed, sessions: provider, prefixCache: manager)
        }
        // The pass is inside its second chunk's session.
        await gate.reached()
        let inside = requests.requests
        let release = DrainGate()
        let holder = Task.detached {
            await provider.withSession { _ in await release.wait() }
        }
        await requests.reached(inside + 1)
        gate.open()
        // The holder has the session; the pass has passed its check and
        // waits for the third chunk's.
        await requests.reached(inside + 2)
        task.cancel()
        await release.open()
        await holder.value
        await task.value

        #expect(pass.deepest() == min(4096, pass.admitPath.count))
    }
}
