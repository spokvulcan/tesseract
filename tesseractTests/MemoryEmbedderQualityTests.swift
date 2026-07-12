//
//  MemoryEmbedderQualityTests.swift
//  tesseractTests
//
//  The embedder-sanity gate (#332).
//
//  Found on the owner's live store, 2026-07-12: every memory shorter than the
//  embedder's 16-token padding floor pooled the hidden state of an EOS pad —
//  the model ignores the attention mask it is handed, and pooling was never
//  handed one at all — so all short memories converged on one content-free
//  direction (the "EOS attractor", mean pairwise cosine 0.959 for sub-40-char
//  memories) and `recall` returned cats and terminal emulators for a query
//  about the Companion feature.
//
//  These tests are the gate that would have caught it: they assert properties
//  no honest sentence embedder can fail — unrelated texts are not identical,
//  a paraphrase beats an unrelated claim, a vector does not depend on its
//  batch-mates, and a query naming a memory ranks that memory first.
//
//  Gated on the embedder being downloaded, so CI (which has none) skips
//  rather than fails. Run it:
//
//      xcodebuild test -project tesseract.xcodeproj -scheme tesseract \
//        -destination 'platform=macOS' -parallel-testing-enabled NO \
//        -only-testing:tesseractTests/MemoryEmbedderQualityTests
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
@Suite("Embedder quality — the EOS-attractor gate", .serialized)
struct MemoryEmbedderQualityTests {

    /// One shared embedder for the suite: `load` is single-flight and
    /// idempotent, and the model costs 0.4 s to load.
    private static let embedder = MemoryEmbedder()

    private static func loaded() async throws -> MemoryEmbedder {
        let directory = try #require(MemoryEvalCorpus.embedderDirectory)
        try await embedder.load(from: directory)
        return embedder
    }

    /// Short, mutually unrelated first-person claims — the exact shape of the
    /// memories the attractor collapsed in the live store.
    private static let unrelatedShorts = [
        "I love cats.",
        "He is from Europe.",
        "I use the Ghostty terminal emulator.",
        "He exclusively uses Celsius for temperature measurements.",
    ]

    // MARK: - Vector geometry

    @Test(
        "Unrelated short texts do not embed to the same vector",
        .enabled(if: MemoryEvalCorpus.isEmbedderAvailable))
    func unrelatedShortsAreDistinct() async throws {
        let embedder = try await Self.loaded()
        let vectors = await embedder.embed(Self.unrelatedShorts)
        try #require(vectors.count == Self.unrelatedShorts.count)

        for i in vectors.indices {
            for j in vectors.indices where j > i {
                let similarity = MemoryStore.cosine(vectors[i], vectors[j])
                #expect(
                    similarity < 0.85,
                    """
                    "\(Self.unrelatedShorts[i])" vs "\(Self.unrelatedShorts[j])" \
                    cosine \(similarity) — attractor-grade similarity between \
                    unrelated claims
                    """)
            }
        }
    }

    @Test(
        "A paraphrase is closer than an unrelated claim",
        .enabled(if: MemoryEvalCorpus.isEmbedderAvailable))
    func paraphraseBeatsUnrelated() async throws {
        let embedder = try await Self.loaded()
        let vectors = await embedder.embed([
            "I love cats.",
            "He adores cats.",
            "He is from Europe.",
        ])
        try #require(vectors.count == 3)

        let paraphrase = MemoryStore.cosine(vectors[0], vectors[1])
        let unrelated = MemoryStore.cosine(vectors[0], vectors[2])
        // A wide margin on purpose: under the attractor both pairs sit ~0.96
        // and this becomes a coin flip. An honest embedder separates them by
        // far more than 0.15.
        #expect(
            paraphrase > unrelated + 0.15,
            "paraphrase \(paraphrase) vs unrelated \(unrelated) — content is not driving similarity"
        )
    }

    @Test(
        "A text's vector does not depend on its batch-mates",
        .enabled(if: MemoryEvalCorpus.isEmbedderAvailable))
    func batchInvariance() async throws {
        let embedder = try await Self.loaded()
        let short = "I love cats."
        let long =
            "The Companion feature is a core, Jarvis-inspired personal assistant for the "
            + "Tesseract application, currently in early development, aggregating context "
            + "across the owner's applications so it can act on his behalf."

        let alone = await embedder.embed([short])
        let accompanied = await embedder.embed([short, long])
        try #require(alone.count == 1 && accompanied.count == 2)

        let drift = MemoryStore.cosine(alone[0], accompanied[0])
        // Padding a batch out to its longest member must not move any other
        // member's vector: with last-token pooling at each sequence's OWN end,
        // trailing pads are invisible to a causal model.
        #expect(
            drift > 0.99,
            "same text, different batch → cosine \(drift); the pooled position depends on padding"
        )
    }

    // MARK: - Through the real read path

    @Test(
        "recall ranks the memory the query names above short distractors",
        .enabled(if: MemoryEvalCorpus.isEmbedderAvailable))
    func recallFindsTheNamedMemory() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("embedder-quality-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }

        let store = try MemoryStore(directory: root)
        let engine = MemoryEngine(
            store: store,
            embedder: Self.embedder,
            isEnabled: { true },
            isDictationCaptureEnabled: { false },
            embedderDirectory: { MemoryEvalCorpus.embedderDirectory }
        )
        await engine.prewarm()

        // The live failure, reconstructed: one long target every one of the
        // query's terms names, under a pile of short attractor residents.
        let target =
            "The Companion feature is a core, Jarvis-inspired personal assistant for the "
            + "Tesseract application, currently in early development."
        let distractors =
            Self.unrelatedShorts + [
                "I like rain, especially summer rain.",
                "He gave the assistant the nickname Pelican.",
                "He dislikes generic or templated responses.",
                "I prefer terse answers while debugging.",
                "He works late in the evening most days.",
                "I drink my coffee black.",
            ]
        for text in distractors { await engine.remember(text) }
        await engine.remember(target)

        let hits = await engine.search(query: "Companion feature Tesseract", limit: 5)
        try #require(!hits.isEmpty, "recall returned nothing for a query naming a stored memory")
        #expect(
            hits.first?.memory.text == target,
            """
            top hit was "\(hits.first?.memory.text ?? "nil")" — the named memory \
            lost to attractor residents
            """)
    }

    // MARK: - The versioned embedding space

    @Test(
        "A scheme bump regenerates every stored vector",
        .enabled(if: MemoryEvalCorpus.isEmbedderAvailable))
    func schemeReconcileRegeneratesVectors() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("embedding-scheme-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }

        // A store from the old era: a real memory under a garbage vector,
        // and no scheme stamp — which must read as scheme 1.
        let store = try MemoryStore(directory: root)
        let text = "The owner's cat is named Behemoth."
        let memory = MemoryRecord(
            text: text, kind: .belief, provenance: .stated, specificity: .general,
            tier: .hot, sourceEpisodeIDs: [], bornAt: Date())
        try await store.upsert(
            memory, embedding: [Float](repeating: 0, count: 1_024),
            journal: JournalEntry(
                at: Date(), mutation: .added, memoryID: memory.id,
                detail: "test fixture", after: text))
        #expect(try await store.embeddingScheme() == 1)

        let engine = MemoryEngine(
            store: store,
            embedder: Self.embedder,
            isEnabled: { true },
            isDictationCaptureEnabled: { false },
            embedderDirectory: { MemoryEvalCorpus.embedderDirectory }
        )
        await engine.prewarm()

        #expect(try await store.embeddingScheme() == MemoryEmbedder.scheme)
        let stored = try #require(try await store.embedding(for: memory.id))
        let fresh = try #require(await Self.embedder.embed(text))
        #expect(
            MemoryStore.cosine(stored, fresh) > 0.999,
            "the stored vector was not regenerated by the running scheme")
    }
}
