import Foundation
import Testing

@testable import Tesseract_Agent

/// Owns the bare `LLMActor` + directly-constructed **Server Completion**
/// module the toy-backed sequencing suites drive. `@unchecked Sendable`:
/// every module use goes through the actor's executor via the `isolated`
/// parameter — the same confinement production has (ADR-0015).
///
/// Passing a `fingerprint` (optionally with an `ssdConfig`) installs the
/// per-load snapshot state exactly as the actor's `loadModel` would, so
/// SSD-tier scenarios (write-through, warm start, hydration) run through
/// the module's real config-resolution chain.
nonisolated final class ServerCompletionFixture: @unchecked Sendable {
    let actor = LLMActor()
    let module = ServerCompletion()
    let provider: ToyModelSessionProvider

    init(
        provider: ToyModelSessionProvider,
        fingerprint: String? = nil,
        ssdConfig: SSDPrefixCacheConfig? = nil,
        identity: ModelIdentity? = nil
    ) {
        self.provider = provider
        if fingerprint != nil || identity != nil {
            module.installLoadTimeState(
                modelIdentity: identity ?? ModelIdentity(configJSON: nil, chatTemplate: nil),
                fingerprint: fingerprint ?? "toy-fingerprint",
                ssdConfig: ssdConfig
            )
        }
        module.installLoadedModelFacts(
            promptStartsThinking: false,
            modelWeightBytes: 0,
            prefixCacheBudgetBytes: 1 << 30
        )
    }

    func start(
        conversation: HTTPPrefixCacheConversation,
        parameters: AgentGenerateParameters
    ) async throws -> HTTPServerGenerationStart {
        try await module.start(
            on: actor,
            sessions: provider,
            modelID: "toy/model",
            conversation: conversation,
            toolSpecs: nil,
            parameters: parameters
        )
    }

    func drain() async {
        await module.drainActiveCompletion(on: actor)
    }

    /// Block until pending SSD-tier writes have drained and the manifest is
    /// durably persisted — the pre-teardown flush a warm-start scenario needs.
    func flush() async {
        await module.flushPrefixCache(on: actor)
    }
}

/// Drains a generation handle's stream into its text and final info payload.
nonisolated func collectServerText(
    _ handle: HTTPServerGenerationStart
) async throws -> (text: String, info: AgentGeneration.Info?) {
    var text = ""
    var info: AgentGeneration.Info?
    for try await event in handle.stream {
        switch event {
        case .text(let chunk): text += chunk
        case .info(let value): info = value
        default: break
        }
    }
    await handle.waitForCompletion()
    return (text, info)
}
