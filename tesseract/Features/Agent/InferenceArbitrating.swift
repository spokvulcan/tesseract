//
//  InferenceArbitrating.swift
//  tesseract
//
//  The narrow seam the lease-acquiring consumers (`AgentRunController`,
//  `SpeechCoordinator`, `AgentCoordinator`) depend on — one member, the scoped
//  lease. Two adapters make it real (ADR-0001): the production
//  `InferenceArbiter`, and the in-memory peer in `tesseractTests` that records
//  lease calls and runs the body without loading a model.
//
//  Deliberately single-member: `reloadLLMIfNeeded` and the read-only model
//  state stay on the concrete facade — their only consumers already hold it.
//  Both adapters are `@MainActor final class`es (not actors), so this is a
//  plain `@MainActor` protocol — no `nonisolated` escape hatch is needed
//  (contrast the actor-backed speech model ports of ADR-0003).
//

/// How a lease chooses the `.llm` slot's vision mode (ADR-0008).
nonisolated enum LLMVisionRequirement: Sendable, Equatable {
    /// Chat UI and background agents: vision follows the user's toggle.
    case fromSettings
    /// HTTP server path: vision whenever the target model is capable, so a
    /// generated client config that advertises image input is always honored —
    /// the chat toggle cannot silently break a configured client.
    case visionIfCapable
}

/// Scoped exclusive GPU access with the required model loaded: waits FIFO-fair
/// for the lease, ensures `slot`'s model is resident, runs `body`, releases on
/// exit — including on throw.
@MainActor
protocol InferenceArbitrating {
    func withExclusiveGPU<T: Sendable>(
        _ slot: ModelSlot,
        llmModelIDOverride: String?,
        llmVision: LLMVisionRequirement,
        body: () async throws -> T
    ) async throws -> T
}

extension InferenceArbitrating {
    /// Protocol requirements cannot carry default arguments; this restores the
    /// common two-argument call shape (`withExclusiveGPU(.llm) { … }`).
    func withExclusiveGPU<T: Sendable>(
        _ slot: ModelSlot,
        body: () async throws -> T
    ) async throws -> T {
        try await withExclusiveGPU(
            slot,
            llmModelIDOverride: nil,
            llmVision: .fromSettings,
            body: body
        )
    }
}
