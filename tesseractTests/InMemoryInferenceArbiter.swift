//
//  InMemoryInferenceArbiter.swift
//  tesseractTests
//
//  The in-memory **Inference Arbitrating** peer — the second adapter that makes
//  the consumer seam real (ADR-0001's "two adapters" rule). It records LLM
//  turns and runs the body without loading a model, so a coordinator's gate
//  contract is assertable hermetically: no engines, no downloads, no GPU.
//

@testable import Tesseract_Agent

@MainActor
final class InMemoryInferenceArbiter: InferenceArbitrating {

    nonisolated struct GateCall: Equatable {
        let modelIDOverride: String?
        let vision: LLMVisionRequirement

        init(modelIDOverride: String? = nil, vision: LLMVisionRequirement = .fromSettings) {
            self.modelIDOverride = modelIDOverride
            self.vision = vision
        }
    }

    /// Every LLM turn taken, in order.
    private(set) var gateCalls: [GateCall] = []

    /// When set, the turn throws before the body runs — the in-memory analogue
    /// of `ensureLLMLoaded` failing (e.g. `modelNotDownloaded`).
    var ensureLoadedError: (any Error)?

    /// When set, the turn suspends this long before running the body — the
    /// in-memory analogue of a run sitting *queued* at the gate (e.g. a
    /// cold-start model load). A cancellation during the wait surfaces as
    /// `CancellationError`, exactly like a real queued turn.
    var gateDelay: Duration?

    func withLLM<T: Sendable>(
        modelIDOverride: String?,
        vision: LLMVisionRequirement,
        body: () async throws -> T
    ) async throws -> T {
        gateCalls.append(GateCall(modelIDOverride: modelIDOverride, vision: vision))
        if let ensureLoadedError { throw ensureLoadedError }
        if let gateDelay { try await Task.sleep(for: gateDelay) }
        return try await body()
    }
}
