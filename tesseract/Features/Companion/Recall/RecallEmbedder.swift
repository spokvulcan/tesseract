//
//  RecallEmbedder.swift
//  tesseract
//
//  The recall embedder's holder: `recall` reranks by meaning through it. It
//  loads the embedding model on first use from the model's usable directory,
//  keeps a main-actor mirror of whether the model is in memory (the Models
//  section and page read it through **Model Residency**), and frees it for
//  Offload Model; the next recall loads it again (0.4 s).
//

import Foundation
import Observation

@Observable @MainActor
final class RecallEmbedder {
    /// Whether the embedding model is in memory, kept in step by this class's
    /// own load and unload paths, the only ones that touch the model.
    private(set) var isModelLoaded = false

    @ObservationIgnored private let embedder: MemoryEmbedder
    @ObservationIgnored private let modelDirectory: @MainActor () -> URL?

    init(embedder: MemoryEmbedder, modelDirectory: @escaping @MainActor () -> URL?) {
        self.embedder = embedder
        self.modelDirectory = modelDirectory
    }

    /// `recall`'s reranking, for the profile tools.
    var recallEmbedding: RecallEmbedding {
        { [weak self] texts in await self?.embed(texts) }
    }

    /// The texts' vectors, loading the model first; nil when it isn't
    /// downloaded or doesn't load, and recall stays keyword-only.
    func embed(_ texts: [String]) async -> [[Float]]? {
        guard let directory = modelDirectory() else { return nil }
        do { try await embedder.load(from: directory) } catch { return nil }
        isModelLoaded = await embedder.isLoaded
        let vectors = await embedder.embed(texts)
        return vectors.count == texts.count ? vectors : nil
    }

    func unload() async {
        await embedder.unload()
        isModelLoaded = await embedder.isLoaded
    }
}
