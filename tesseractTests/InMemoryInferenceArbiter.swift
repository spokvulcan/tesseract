//
//  InMemoryInferenceArbiter.swift
//  tesseractTests
//
//  The in-memory **Inference Arbitrating** peer — the second adapter that makes
//  the consumer seam real (ADR-0001's "two adapters" rule). It records lease
//  calls and runs the body without loading a model, so a coordinator's lease
//  contract is assertable hermetically: no engines, no downloads, no GPU.
//

@testable import Tesseract_Agent

@MainActor
final class InMemoryInferenceArbiter: InferenceArbitrating {

    nonisolated struct LeaseCall: Equatable {
        let slot: ModelSlot
        let llmModelIDOverride: String?
        let llmVision: LLMVisionRequirement

        init(
            slot: ModelSlot,
            llmModelIDOverride: String? = nil,
            llmVision: LLMVisionRequirement = .fromSettings
        ) {
            self.slot = slot
            self.llmModelIDOverride = llmModelIDOverride
            self.llmVision = llmVision
        }
    }

    /// Every lease acquisition, in order.
    private(set) var leaseCalls: [LeaseCall] = []

    /// When set, the lease throws before the body runs — the in-memory analogue
    /// of `ensureLoaded` failing (e.g. `modelNotDownloaded`).
    var ensureLoadedError: (any Error)?

    func withExclusiveGPU<T: Sendable>(
        _ slot: ModelSlot,
        llmModelIDOverride: String?,
        llmVision: LLMVisionRequirement,
        body: () async throws -> T
    ) async throws -> T {
        leaseCalls.append(
            LeaseCall(
                slot: slot,
                llmModelIDOverride: llmModelIDOverride,
                llmVision: llmVision
            ))
        if let ensureLoadedError { throw ensureLoadedError }
        return try await body()
    }
}
