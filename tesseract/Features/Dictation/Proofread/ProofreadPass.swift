//
//  ProofreadPass.swift
//  tesseract
//
//  The policy face of the **Proofread Pass** (map #283, ADR-0034): decides
//  per dictation whether the pass runs at all, and turns the model's reply
//  into a `ProofreadVerdict`. Every skip path returns `nil` — the session
//  then commits the regex-cleaned raw text, so dictation latency is never
//  hostage to the proofreader (skip-when-busy is the owner's locked call).
//
//  The model call is injected as a closure, so the policy is hermetically
//  testable without MLX.
//

import Foundation

@MainActor
final class ProofreadPass {

    /// The fixed instruction — rendered once into the model's KV cache, so
    /// only the transcript itself prefills per pass. Plain-text contract
    /// with a single REJECT escape hatch: a no-think 0.8B holds this far
    /// more reliably than JSON.
    static let systemPrompt = """
        You are a dictation proofreader. The user message is raw speech-to-text \
        output. Correct transcription errors only: punctuation, capitalization, \
        obvious misheard words, and stray filler artifacts. Preserve the \
        speaker's wording, language, and meaning exactly — never rephrase, \
        expand, or answer. Output only the corrected text, nothing else. \
        If the transcription is unintelligible garbage that cannot be repaired, \
        output exactly: REJECT: <three-to-five-word reason>
        """

    /// The pass's whole time budget. Overrunning it skips the pass (the raw
    /// text commits); the orphaned generation is cancelled best-effort.
    static let budget: Duration = .seconds(4)

    /// Whether the proofread model is resident — a main-actor mirror kept in
    /// step by this class's own load/unload paths (the only ones that touch
    /// the model), so views can render residency synchronously.
    private(set) var isModelLoaded = false

    private let isEnabled: @MainActor () -> Bool
    private let isGPUBusy: @MainActor () -> Bool
    private let modelDirectory: @MainActor () -> URL?
    private let loadModel: @Sendable (URL) async throws -> Void
    private let runModel: @Sendable (String, String) async throws -> String
    private let unloadModel: @Sendable () async -> Void
    private let budget: Duration

    init(
        isEnabled: @escaping @MainActor () -> Bool,
        isGPUBusy: @escaping @MainActor () -> Bool,
        modelDirectory: @escaping @MainActor () -> URL?,
        loadModel: @escaping @Sendable (URL) async throws -> Void,
        runModel: @escaping @Sendable (String, String) async throws -> String,
        unloadModel: @escaping @Sendable () async -> Void,
        budget: Duration = ProofreadPass.budget
    ) {
        self.isEnabled = isEnabled
        self.isGPUBusy = isGPUBusy
        self.modelDirectory = modelDirectory
        self.loadModel = loadModel
        self.runModel = runModel
        self.unloadModel = unloadModel
        self.budget = budget
    }

    /// Proofreads one transcription. `nil` means the pass was skipped —
    /// disabled, model not downloaded, GPU lease held (skip-when-busy),
    /// load failure, budget overrun, or a model error. Every `nil` is
    /// fail-open: the caller commits the raw text.
    func proofread(_ text: String) async -> ProofreadVerdict? {
        guard isEnabled() else { return nil }
        guard let directory = modelDirectory() else { return nil }
        // Skip-when-busy: the agent (or server) holds the GPU lease — inject
        // the regex-cleaned raw immediately rather than queue behind a turn
        // that can run for minutes.
        guard !isGPUBusy() else {
            Log.transcription.info("Proofread skipped: GPU lease held")
            return nil
        }

        do {
            try await loadModel(directory)
            isModelLoaded = true
        } catch {
            Log.transcription.error(
                "Proofread model load failed: \(error.localizedDescription)")
            return nil
        }

        // Budget race, same abandonment shape as the transcription timeout:
        // the pass must never stall the commit past its budget.
        let run = runModel
        let work = Task { try await run(Self.systemPrompt, text) }
        let reply = await withTaskCancellationHandler {
            await Self.awaitWithBudget(work, budget: budget)
        } onCancel: {
            work.cancel()
        }
        guard let reply else { return nil }
        return ProofreadReply.parse(reply, raw: text)
    }

    /// Warms the model at launch so the first dictation's pass is not the
    /// one paying the load. No-ops when disabled or not downloaded.
    func prewarm() async {
        guard isEnabled(), let directory = modelDirectory() else { return }
        if (try? await loadModel(directory)) != nil {
            isModelLoaded = true
        }
    }

    func unload() async {
        await unloadModel()
        isModelLoaded = false
    }

    private static func awaitWithBudget(
        _ work: Task<String, any Error>, budget: Duration
    ) async -> String? {
        let timeout = Task {
            try? await Task.sleep(for: budget)
            work.cancel()
        }
        defer { timeout.cancel() }
        do {
            return try await work.value
        } catch {
            if !(error is CancellationError) {
                Log.transcription.error("Proofread failed: \(error.localizedDescription)")
            }
            return nil
        }
    }
}
