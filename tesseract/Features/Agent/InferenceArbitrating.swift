//
//  InferenceArbitrating.swift
//  tesseract
//
//  The narrow seam LLM consumers (`AgentRunController`, the Day Thread, the
//  completion handler) depend on — one member, the scoped LLM turn. Two
//  adapters make it real (ADR-0001): the production `InferenceArbiter`, and
//  the in-memory peer in `tesseractTests` that records turns and runs the
//  body without loading a model.
//
//  Deliberately single-member: `reloadLLMIfNeeded` and the read-only model
//  state stay on the concrete facade — their only consumers already hold it.
//  Both adapters are `@MainActor final class`es (not actors), so this is a
//  plain `@MainActor` protocol — no `nonisolated` escape hatch is needed
//  (contrast the actor-backed speech model ports of ADR-0003).
//

/// How an LLM turn chooses the model's vision mode (ADR-0008).
nonisolated enum LLMVisionRequirement: Sendable, Equatable {
    /// Chat UI and background agents: load vision when the user's global
    /// "Use vision models when available" opt-out is on *and* the model is
    /// capable. Opting out forces the text-only container.
    case fromSettings
    /// HTTP server path: vision whenever the target model is capable, so a
    /// generated client config that advertises image input is always honored —
    /// the global opt-out cannot silently break a configured client.
    case visionIfCapable

    /// Resolve to a concrete "load the vision container?" decision. Pure so the
    /// policy is unit-tested without the arbiter: `.fromSettings` honors the
    /// global opt-out *and* capability; `.visionIfCapable` ignores the opt-out
    /// (ADR-0008) and follows capability alone.
    nonisolated func wantsVision(useVisionWhenAvailable: Bool, isVisionCapable: Bool) -> Bool {
        switch self {
        case .fromSettings: useVisionWhenAvailable && isVisionCapable
        case .visionIfCapable: isVisionCapable
        }
    }
}

/// One LLM turn with the selected model loaded: waits its FIFO turn at the
/// **LLM Gate**, makes sure the model is resident, runs `body`, releases on
/// exit — including on throw. Only LLM work takes the gate (ADR-0081).
@MainActor
protocol InferenceArbitrating {
    func withLLM<T: Sendable>(
        modelIDOverride: String?,
        vision: LLMVisionRequirement,
        body: () async throws -> T
    ) async throws -> T
}

extension InferenceArbitrating {
    /// Protocol requirements cannot carry default arguments; this restores the
    /// common one-argument call shape (`withLLM { … }`).
    func withLLM<T: Sendable>(body: () async throws -> T) async throws -> T {
        try await withLLM(modelIDOverride: nil, vision: .fromSettings, body: body)
    }
}
