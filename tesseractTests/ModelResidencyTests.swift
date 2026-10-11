//
//  ModelResidencyTests.swift
//  tesseractTests
//
//  **Model Residency** through its interface, over test holders: what each
//  holder reports is what the menu's Models section shows and what the Models
//  page marks as in memory, and Offload Model frees every holder that lets go
//  of its model — at once, whatever one of them waits for.
//

import Foundation
import Observation
import Testing

@testable import Tesseract_Agent

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct ModelResidencyTests {

    /// A model's holder for the tests: its state is set by the test, and its
    /// unload is recorded (and can be held, the way the language model waits
    /// for the running generation).
    @MainActor @Observable
    final class TestHolder {
        let id: String
        let model: String
        var state: ModelActivityRow.State
        var holdsUnload = false
        private(set) var unloads = 0
        private var gate: CheckedContinuation<Void, Never>?
        var isWaitingToUnload: Bool { gate != nil }

        init(_ id: String, model: String, state: ModelActivityRow.State) {
            self.id = id
            self.model = model
            self.state = state
        }

        func unload() async {
            if holdsUnload { await withCheckedContinuation { gate = $0 } }
            unloads += 1
            state = .notLoaded
        }

        func release() {
            gate?.resume()
            gate = nil
        }

        func holder(
            role: String = "Role", unloads: Bool = true, listedOnlyWhileLoaded: Bool = false,
            hasSize: Bool = true
        ) -> ModelResidency.Holder {
            var holder = ModelResidency.Holder(
                id: id, role: role, model: { self.model }, state: { self.state },
                isListedOnlyWhileLoaded: listedOnlyWhileLoaded, hasSize: hasSize)
            if unloads { holder.unload = { await self.unload() } }
            return holder
        }
    }

    private let llmID = "qwen3.8-27b-paro"
    private let gigabytes: Int64 = 15_132_802_752

    // MARK: - The menu's Models section

    /// Each row is what its holder reports: the model's name from the
    /// catalog, its state, and its size on disk once it is in memory.
    @Test func eachRowIsWhatItsHolderReports() {
        let llm = TestHolder("llm", model: llmID, state: .working("Jarvis · Triage"))
        let voice = TestHolder(
            "tts", model: ModelDefinition.defaultTextToSpeechModelID, state: .notLoaded)
        let dictation = TestHolder(
            "stt", model: ModelDefinition.defaultSpeechToTextModelID, state: .working("Listening"))
        let residency = ModelResidency(
            holders: [
                llm.holder(role: "Language model"), voice.holder(role: "Voice"),
                dictation.holder(role: "Dictation", unloads: false, hasSize: false),
            ],
            sizeOnDisk: { $0 == llmID ? gigabytes : 1 },
            memory: { (21, 8) })

        let activity = residency.activity

        #expect(
            activity.rows == [
                ModelActivityRow(
                    id: "llm", role: "Language model",
                    name: ModelDefinition.withID(llmID)!.displayName,
                    state: .working("Jarvis · Triage"), bytes: gigabytes),
                ModelActivityRow(
                    id: "tts", role: "Voice",
                    name: ModelDefinition.withID(ModelDefinition.defaultTextToSpeechModelID)!
                        .displayName,
                    state: .notLoaded),
                ModelActivityRow(
                    id: "stt", role: "Dictation",
                    name: ModelDefinition.withID(ModelDefinition.defaultSpeechToTextModelID)!
                        .displayName,
                    state: .working("Listening")),
            ])
        #expect(activity.footprintBytes == 21)
        #expect(activity.swapBytes == 8)
    }

    /// The speed-up draft is listed only while it is loaded beside its target.
    @Test func theDraftIsListedOnlyWhileLoaded() {
        let draft = TestHolder("draft", model: DFlash2Support.draftModelID, state: .notLoaded)
        let residency = ModelResidency(
            holders: [draft.holder(unloads: false, listedOnlyWhileLoaded: true)],
            sizeOnDisk: { _ in nil }, memory: { (nil, nil) })

        #expect(residency.activity.rows.isEmpty)
        draft.state = .working("Drafting")
        #expect(residency.activity.rows.map(\.id) == ["draft"])
    }

    // MARK: - The Models page's mark

    /// A model is marked in memory exactly when the holder that holds it
    /// reports it loaded or working — one fact for the menu and the page.
    @Test func theModelsPageMarksWhatAHolderHolds() {
        let llm = TestHolder("llm", model: llmID, state: .loading)
        let draft = TestHolder("draft", model: DFlash2Support.draftModelID, state: .notLoaded)
        let embedder = TestHolder(
            "embedder", model: ModelDefinition.defaultEmbeddingModelID, state: .loaded)
        let residency = ModelResidency(
            holders: [llm.holder(), draft.holder(unloads: false), embedder.holder()],
            sizeOnDisk: { _ in nil }, memory: { (nil, nil) })

        #expect(!residency.isLoaded(llmID))
        #expect(!residency.isLoaded(DFlash2Support.draftModelID))
        #expect(residency.isLoaded(ModelDefinition.defaultEmbeddingModelID))
        #expect(!residency.isLoaded(ModelDefinition.defaultProofreadModelID))

        llm.state = .working("Generating")
        draft.state = .loaded
        #expect(residency.isLoaded(llmID))
        #expect(residency.isLoaded(DFlash2Support.draftModelID))
    }

    /// Both 27B entries list the draft as a dependency, and it is the only
    /// draft in the catalog, so one holder covers the page's draft rows.
    @Test func theDFlash2DraftIsTheCatalogsOnlyDraft() {
        let drafts = ModelDefinition.all.filter { $0.category == .draft }.map(\.id)
        #expect(drafts == [DFlash2Support.draftModelID])
        #expect(ModelDefinition.withID(llmID)?.dependencies == [DFlash2Support.draftModelID])
        #expect(
            ModelDefinition.withID("qwen3.8-27b")?.dependencies == [DFlash2Support.draftModelID])
    }

    // MARK: - Offload Model

    /// Offload frees every holder that lets go of its model — the voice, the
    /// proofreader and the recall embedder at once, the language model after
    /// the running generation — and leaves the one that doesn't (dictation:
    /// always armed, ADR-0025).
    @Test func offloadFreesEverythingThatReportsItselfUnloadable() async {
        let llm = TestHolder("llm", model: llmID, state: .working("Generating"))
        llm.holdsUnload = true
        let voice = TestHolder(
            "tts", model: ModelDefinition.defaultTextToSpeechModelID, state: .loaded)
        let dictation = TestHolder(
            "stt", model: ModelDefinition.defaultSpeechToTextModelID, state: .loaded)
        let proofreader = TestHolder(
            "proofread", model: ModelDefinition.defaultProofreadModelID, state: .loaded)
        let embedder = TestHolder(
            "embedder", model: ModelDefinition.defaultEmbeddingModelID, state: .loaded)
        let residency = ModelResidency(
            holders: [
                llm.holder(), voice.holder(), dictation.holder(unloads: false),
                proofreader.holder(), embedder.holder(),
            ],
            sizeOnDisk: { _ in nil }, memory: { (nil, nil) })
        #expect(residency.canOffload)

        let offload = Task { await residency.offload() }
        // The language model waits its turn; the rest are freed already.
        #expect(await observe(until: { llm.isWaitingToUnload && embedder.unloads == 1 }))
        #expect(voice.unloads == 1)
        #expect(proofreader.unloads == 1)
        #expect(llm.unloads == 0)

        llm.release()
        await offload.value
        #expect(llm.unloads == 1)
        #expect(dictation.unloads == 0)
        #expect(dictation.state == .loaded)
        #expect(!residency.canOffload)
    }

    /// Offload is offered while anything it would free is in memory — the
    /// proofreader or the embedder alone too — and not for a model it
    /// leaves loaded.
    @Test func offloadIsOfferedWhileAnythingItFreesIsLoaded() {
        let dictation = TestHolder(
            "stt", model: ModelDefinition.defaultSpeechToTextModelID, state: .loaded)
        let embedder = TestHolder(
            "embedder", model: ModelDefinition.defaultEmbeddingModelID, state: .notLoaded)
        let residency = ModelResidency(
            holders: [dictation.holder(unloads: false), embedder.holder()],
            sizeOnDisk: { _ in nil }, memory: { (nil, nil) })

        #expect(!residency.canOffload)
        embedder.state = .loaded
        #expect(residency.canOffload)
    }

    /// The app's own holders: the menu lists every model the app can hold,
    /// in order (the draft only beside its target), and with nothing loaded
    /// Offload has nothing to free.
    @Test func theAppsHoldersListEveryModel() {
        let container = DependencyContainer()

        let activity = container.modelResidency.activity

        #expect(
            activity.rows.map(\.role) == [
                "Language model", "Voice", "Dictation", "Proofreader", "Memory search",
            ])
        #expect(activity.rows.allSatisfy { $0.state == .notLoaded })
        #expect(!container.modelResidency.canOffload)
    }

    // MARK: - The usable directory

    /// One answer for every loader: a model's folder once its files are all
    /// there by its entry's completeness rule, nil until then.
    @Test func aModelsUsableDirectoryIsItsFolderOnceDownloaded() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("residency-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = ModelDefinition(
            id: "test-model", displayName: "Test", description: "", category: .proofread,
            source: .huggingFace(repo: "test/model", requiredExtension: "safetensors"),
            sizeDescription: "", dependencies: [])
        let downloads = ModelDownloadManager(
            fetching: InMemoryModelFetching(), storageRoot: root, definitions: [model])
        #expect(downloads.usableDirectory(for: model.id) == nil)

        let folder = try #require(downloads.modelPath(for: model.id))
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try Data(count: 64).write(to: folder.appendingPathComponent("model.safetensors"))
        downloads.refreshStatus(for: model.id)

        #expect(downloads.usableDirectory(for: model.id)?.path == folder.path)
        #expect(downloads.sizeOnDisk(of: model.id) != nil)
    }
}
