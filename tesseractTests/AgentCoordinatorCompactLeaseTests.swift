import Foundation
import Testing
@testable import Tesseract_Agent

/// Epic 3 Task 4 lease coverage — `/compact` must go through the same
/// `InferenceArbiter.withExclusiveGPU(.llm)` contract as a regular agent turn.
/// Without the lease, manual compaction would bypass `ensureLoaded` and could
/// race a foreground turn with a stale model state.
///
/// We prove the fix by making the arbiter's model load deterministically fail:
/// `selectedAgentModelID` is pointed at a model the `ModelDownloadManager`
/// knows nothing about, so `ensureLoaded(.llm)` throws
/// `AgentEngineError.modelNotDownloaded` *before* the lease body runs.
/// The injected summarize closure records any invocation; if `/compact` was
/// wired to the arbiter path, it must never fire.
@MainActor
struct AgentCoordinatorCompactLeaseTests {

    private actor SummarizeRecorder {
        private(set) var callCount = 0
        func record() { callCount += 1 }
    }

    private static let oldUser = UserMessage(content: String(repeating: "A", count: 2_000))
    private static let oldAssistant = AssistantMessage(content: String(repeating: "B", count: 2_000))
    private static let recentUser = UserMessage(content: String(repeating: "C", count: 4_000))
    private static let recentAssistant = AssistantMessage(content: String(repeating: "D", count: 4_000))

    private static let compactCommand = SlashCommand(
        name: "compact",
        description: "",
        source: .builtIn,
        argumentHint: nil
    )

    private func makeAgent() -> Agent {
        let config = AgentLoopConfig(
            model: AgentModelRef(id: "nonexistent-test-model"),
            convertToLlm: { _ in [] },
            contextTransform: nil,
            getSteeringMessages: nil,
            getFollowUpMessages: nil
        )
        return Agent(
            config: config,
            systemPrompt: "test",
            tools: [],
            generate: { _, _, _, _ in
                AsyncThrowingStream { $0.finish() }
            }
        )
    }

    /// Waits until `isGenerating` flips back to `false`, polling the MainActor
    /// state every 20ms. Bounded so a stuck coordinator fails loudly rather
    /// than hanging CI.
    private func waitForIdle(
        _ coordinator: AgentCoordinator,
        timeout: Duration = .seconds(3)
    ) async throws {
        let deadline = ContinuousClock.now + timeout
        while coordinator.isGenerating {
            try await Task.sleep(for: .milliseconds(20))
            if ContinuousClock.now >= deadline {
                Issue.record("Coordinator did not become idle within timeout")
                return
            }
        }
    }

    @Test func compactCommandRunsUnderArbiterLeaseAndSkipsSummarizeWhenModelUnavailable() async throws {
        // `SettingsManager` persists via UserDefaults — clean up our override so
        // sibling tests see the production default on subsequent runs.
        UserDefaults.standard.removeObject(forKey: "selectedAgentModelID")
        defer { UserDefaults.standard.removeObject(forKey: "selectedAgentModelID") }

        let settings = SettingsManager()
        // Force the arbiter's `ensureLoaded` to throw deterministically —
        // an ID the download manager has never seen.
        settings.selectedAgentModelID = "tesseract-compact-lease-test-model"

        let arbiter = InferenceArbiter(
            agentEngine: AgentEngine(),
            speechEngine: SpeechEngine(),
            imageGenEngine: ImageGenEngine(),
            zimageGenEngine: ZImageGenEngine(),
            settingsManager: settings,
            modelDownloadManager: ModelDownloadManager()
        )

        let agent = makeAgent()
        let recorder = SummarizeRecorder()
        let summarize: @Sendable (String) async throws -> String = { _ in
            await recorder.record()
            return "summary"
        }

        let coordinator = AgentCoordinator(
            agent: agent,
            conversationStore: AgentConversationStore(),
            settings: settings,
            arbiter: arbiter,
            contextManager: ContextManager(settings: .small),
            contextWindow: 5_000,
            summarize: summarize
        )

        // Coordinator init may rehydrate an older conversation from disk — force
        // the transcript we want to see under compaction regardless.
        agent.loadMessages([
            Self.oldUser,
            Self.oldAssistant,
            Self.recentUser,
            Self.recentAssistant,
        ])

        coordinator.executeCommand(Self.compactCommand, arguments: "")

        // Must flip synchronously so the UI shows progress right away.
        #expect(coordinator.isGenerating == true)

        try await waitForIdle(coordinator)

        // The arbiter threw before the lease body ran, so `forceCompact` never
        // got a chance to drive `summarize`. A non-arbiter path would have
        // invoked summarize at least once (messages are sized to exceed the
        // small-compaction budget).
        #expect(await recorder.callCount == 0)
        #expect(coordinator.error != nil)
    }

    /// Without an arbiter the coordinator falls back to the pre-existing
    /// direct path — summarize still runs. Locks in the back-compat branch so
    /// a future refactor cannot silently require an arbiter to be wired.
    @Test func compactCommandFallsBackToDirectPathWhenArbiterMissing() async throws {
        let agent = makeAgent()
        let recorder = SummarizeRecorder()
        let summarize: @Sendable (String) async throws -> String = { _ in
            await recorder.record()
            return "## Goal\ncompacted"
        }

        let coordinator = AgentCoordinator(
            agent: agent,
            conversationStore: AgentConversationStore(),
            settings: nil,
            arbiter: nil,
            contextManager: ContextManager(settings: .small),
            contextWindow: 5_000,
            summarize: summarize
        )

        agent.loadMessages([
            Self.oldUser,
            Self.oldAssistant,
            Self.recentUser,
            Self.recentAssistant,
        ])

        coordinator.executeCommand(Self.compactCommand, arguments: "")

        // The direct path runs `forceCompact` synchronously up to the summarize
        // call's suspension — poll for the recorder to capture the invocation.
        let deadline = ContinuousClock.now + .seconds(3)
        while await recorder.callCount == 0 {
            try await Task.sleep(for: .milliseconds(20))
            if ContinuousClock.now >= deadline {
                Issue.record("Fallback path did not invoke summarize within timeout")
                break
            }
        }

        #expect(await recorder.callCount >= 1)
    }
}
