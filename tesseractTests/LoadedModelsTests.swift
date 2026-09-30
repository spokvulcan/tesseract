//
//  LoadedModelsTests.swift
//  tesseractTests
//
//  The Models page's in-memory mark (#577). A draft loads beside its target
//  only when the speculation setting allows it and the target pairs with it,
//  so the draft's row follows whether the draft loaded, not the target's id.
//

import Testing

@testable import Tesseract_Agent

@MainActor
struct LoadedModelsTests {
    private let draft = ModelDefinition.withID(DFlash2Support.draftModelID)!
    private let paroTarget = ModelDefinition.withID("qwen3.8-27b-paro")!
    private let mlxTarget = ModelDefinition.withID("qwen3.8-27b")!

    /// Qwen3.8-27B PARO with speculation on: the draft loaded beside it.
    @Test func theDraftShowsLoadedBesideThePAROTarget() {
        let loaded = LoadedModels(agentModelID: paroTarget.id, isDFlash2DraftLoaded: true)
        #expect(loaded.contains(paroTarget))
        #expect(loaded.contains(draft))
    }

    /// Qwen3.8-27B with speculation Off: the draft never loaded.
    @Test func theDraftShowsUnloadedWhenTheLoadLeftItOut() {
        let loaded = LoadedModels(agentModelID: mlxTarget.id, isDFlash2DraftLoaded: false)
        #expect(loaded.contains(mlxTarget))
        #expect(!loaded.contains(draft))
    }

    /// Both 27B entries list the draft as a dependency, and it is the only
    /// draft in the catalog, so one fact covers the page's draft rows.
    @Test func theDFlash2DraftIsTheCatalogsOnlyDraft() {
        let drafts = ModelDefinition.all.filter { $0.category == .draft }.map(\.id)
        #expect(drafts == [DFlash2Support.draftModelID])
        #expect(paroTarget.dependencies == [DFlash2Support.draftModelID])
        #expect(mlxTarget.dependencies == [DFlash2Support.draftModelID])
    }

    @Test func eachOtherRowFollowsItsOwnEngine() {
        let loaded = LoadedModels(
            speechToTextModelID: nil, isVoiceModelLoaded: true, agentModelID: nil,
            isProofreadModelLoaded: false, isEmbedderLoaded: true)
        let voice = ModelDefinition.withID(ModelDefinition.defaultTextToSpeechModelID)!
        let proofread = ModelDefinition.withID(ModelDefinition.defaultProofreadModelID)!
        let embedder = ModelDefinition.withID(ModelDefinition.defaultEmbeddingModelID)!
        #expect(loaded.contains(voice))
        #expect(!loaded.contains(proofread))
        #expect(loaded.contains(embedder))
        #expect(!loaded.contains(paroTarget))
        #expect(!loaded.contains(draft))
    }
}
