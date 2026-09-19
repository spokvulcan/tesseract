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
                consumeAll: true, initialState: nil, evalPolicy: .pipelined)
            let system = try #require(
                session.captureSnapshot(cache: cache, offset: 2, type: .system))
            _ = try session.prefill(
                text: .init(tokens: MLXArray(prefix.suffix(2).map(Int32.init))),
                cache: cache, checkpoints: [:], checkpointBaseOffset: 2, prefillStepSize: 2,
                consumeAll: true, initialState: nil, evalPolicy: .pipelined)
            let view = try #require(
                session.captureSnapshot(cache: cache, offset: 4, type: .branchPoint))
            _ = try session.prefill(
                text: .init(tokens: MLXArray([Int32(42), 42, 42, 42])),
                cache: cache, checkpoints: [:], checkpointBaseOffset: 4, prefillStepSize: 4,
                consumeAll: true, initialState: nil, evalPolicy: .pipelined)
            return (
                system, view,
                try #require(session.captureSnapshot(cache: cache, offset: 8, type: .leaf))
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
        let seed = SpeculativeCanonicalPrefill.makeSeed(
            storedConversation: stored, render: render, keySpace: .identity(keyPath: prefix),
            partitionKey: key, prefillStepSize: 256, ssdEnabled: false,
            seedsPositionAnchor: false, canonicalLeafOffset: 0,
            transientBoundary: backingState == "planned" ? nil : view,
            ramOnlySpine: ramOnlySpine, diagnostics: diagnostics)
        await SpeculativeCanonicalPrefill.run(
            seed: seed, container: provider.container, prefixCache: manager)
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

    @Test func preemptedPassSettlesPartialLeafThatResolutionThenSurfaces() async throws {
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
        let provider = ToyModelSessionProvider(
            model: ToyLanguageModel(script: [0], onForward: gate.onForward),
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
                evalPolicy: .pipelined
            )
            return session.captureSnapshot(cache: cache, offset: boundaryTokens.count, type: .leaf)
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

        let seed = SpeculativeCanonicalPrefill.makeSeed(
            storedConversation: stored,
            render: ConversationRender.uncached(tokenizer: tokenizer),
            keySpace: .identity(keyPath: []),
            partitionKey: partitionKey,
            prefillStepSize: 1024,
            ssdEnabled: false,
            seedsPositionAnchor: false,
            canonicalLeafOffset: 0,
            diagnostics: diagnostics
        )

        let container = provider.container
        let pass = Task {
            await SpeculativeCanonicalPrefill.run(
                seed: seed, container: container, prefixCache: manager
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
    }
}
