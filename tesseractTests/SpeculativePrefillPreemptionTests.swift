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
            toolSpecs: nil,
            tokenizer: tokenizer,
            keySpace: .identity(keyPath: []),
            renderContext: .canonical
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
            let cache = session.newCache(parameters: GenerateParameters(temperature: 0))
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
            toolSpecs: nil,
            tokenizer: tokenizer,
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
