//
//  ModelResidency.swift
//  tesseract
//
//  **Model Residency**: the one module that knows which models are in memory,
//  what each is doing, and how to free them (ADR-0081 §4–5). Each model's
//  holder — the agent engine behind the LLM Gate, the speech engine's
//  presenter, the transcription engine, the Proofread Pass, the recall
//  embedder — reports to it as one row: the model it holds, whether that is
//  loaded and what it is doing, and how to free it. The menu's Models section,
//  the Models page's in-memory mark and Offload Model all read it, so they can't
//  disagree about a model.
//
//  Sizes are the weights on disk, which is what an MLX model holds in memory;
//  Whisper runs on the Neural Engine, so its row has no size.
//

import Foundation

@MainActor
final class ModelResidency {
    /// One model's holder, read live.
    struct Holder {
        /// The menu row's id.
        let id: String
        /// What the model is for: "Language model", "Voice", "Dictation"…
        let role: String
        /// The catalog id of the model it holds, or would load.
        let model: @MainActor () -> String
        /// Whether that model is loaded, and what it is doing.
        let state: @MainActor () -> ModelActivityRow.State
        /// Frees the model. Nil for one Offload leaves alone.
        var unload: (@MainActor () async -> Void)?
        /// Listed only while loaded: the speed-up draft loads only beside
        /// its target.
        var isListedOnlyWhileLoaded = false
        /// Whether its weights' size on disk is what it holds in memory.
        var hasSize = true
    }

    private let holders: [Holder]
    private let sizeOnDisk: @MainActor (String) -> Int64?
    private let memory: @MainActor () -> (footprint: Int?, swap: UInt64?)

    /// - Parameters:
    ///   - holders: one per model, in the menu's order.
    ///   - sizeOnDisk: a model's weights on disk, what an MLX model holds in
    ///     memory.
    ///   - memory: the app's footprint and the system's swap in use.
    init(
        holders: [Holder],
        sizeOnDisk: @escaping @MainActor (String) -> Int64?,
        memory: @escaping @MainActor () -> (footprint: Int?, swap: UInt64?) = {
            let sample = RequestMemoryTelemetry.Sample.current()
            return (sample.footprintBytes, sample.systemSwapUsedBytes)
        }
    ) {
        self.holders = holders
        self.sizeOnDisk = sizeOnDisk
        self.memory = memory
    }

    /// What the menu's Models section shows: every model, loaded or not, what
    /// it is doing, its size, and the app's memory with the system's swap.
    var activity: ModelActivity {
        let rows = holders.compactMap { holder -> ModelActivityRow? in
            let state = holder.state()
            let resident = Self.isResident(state)
            if holder.isListedOnlyWhileLoaded, !resident { return nil }
            let model = holder.model()
            return ModelActivityRow(
                id: holder.id, role: holder.role,
                name: ModelDefinition.withID(model)?.displayName ?? model, state: state,
                bytes: resident && holder.hasSize ? sizeOnDisk(model) : nil)
        }
        let memory = memory()
        return ModelActivity(rows: rows, footprintBytes: memory.footprint, swapBytes: memory.swap)
    }

    /// Whether a model is in memory: the Models page's mark.
    func isLoaded(_ modelID: String) -> Bool {
        holders.contains { $0.model() == modelID && Self.isResident($0.state()) }
    }

    /// Whether Offload Model would free anything.
    var canOffload: Bool {
        holders.contains { $0.unload != nil && Self.isResident($0.state()) }
    }

    /// Offload Model: frees every model a holder lets go of, all at once. The
    /// language model goes after the running generation (its holder waits its
    /// turn at the LLM Gate); the rest go at once. The next use loads each
    /// again.
    func offload() async {
        let unloads = holders.compactMap(\.unload).map { unload in Task { await unload() } }
        for unload in unloads { await unload.value }
    }

    private static func isResident(_ state: ModelActivityRow.State) -> Bool {
        switch state {
        case .loaded, .working: true
        case .notLoaded, .loading: false
        }
    }
}
