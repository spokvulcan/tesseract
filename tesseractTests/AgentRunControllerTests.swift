//
//  AgentRunControllerTests.swift
//  tesseractTests
//
//  Tests the **Agent Run** module at its own seam — no AgentCoordinator, no
//  conversation store, no transcript. Covers the single-writer `isGenerating`
//  with its eager "queued-behind-the-lease" semantics, and BOTH lease paths:
//  the arbiter lease contract and the arbiter-less direct fallback. Absorbs the
//  former `AgentCoordinatorCompactLeaseTests`, whose `/compact` lease coverage
//  now lives here via the shared `runUnderLease` entry.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct AgentRunControllerTests {

    // MARK: - Doubles

    private actor SummarizeRecorder {
        private(set) var callCount = 0
        func record() { callCount += 1 }
    }

    @MainActor
    private final class ErrorRecorder {
        private(set) var messages: [String] = []
        func report(_ message: String) { messages.append(message) }
    }

    private static let oldUser = UserMessage(content: String(repeating: "A", count: 2_000))
    private static let oldAssistant = AssistantMessage(content: String(repeating: "B", count: 2_000))
    private static let recentUser = UserMessage(content: String(repeating: "C", count: 4_000))
    private static let recentAssistant = AssistantMessage(content: String(repeating: "D", count: 4_000))

    private func makeAgent() -> Agent {
        makeNoOpAgent(modelID: "agent-run-controller-test-model")
    }

    /// An arbiter whose `ensureLoaded(.llm)` throws deterministically — the model
    /// id points at something the download manager has never seen, so the lease
    /// throws `modelNotDownloaded` *before* the body runs.
    private func makeFailingArbiter() -> InferenceArbiter {
        let settings = SettingsManager(store: InMemorySettingsStore())
        settings.selectedAgentModelID = "agent-run-controller-nonexistent-model"
        return InferenceArbiter(
            agentEngine: AgentEngine(),
            speechEngine: SpeechEngine(),
            settingsManager: settings,
            modelDownloadManager: ModelDownloadManager()
        )
    }

    private func waitUntilIdle(
        _ run: AgentRunController,
        timeout: Duration = .seconds(3)
    ) async throws {
        let deadline = ContinuousClock.now + timeout
        while run.isGenerating {
            try await Task.sleep(for: .milliseconds(20))
            if ContinuousClock.now >= deadline {
                Issue.record("AgentRunController did not become idle within timeout")
                return
            }
        }
    }

    // MARK: - Eager flag

    /// `isGenerating` flips to `true` *synchronously* in `runUnderLease`, before
    /// the lease body runs — while the agent is still `.idle` (the run is queued
    /// behind the lease). This is the "queued **or** active" semantics that only
    /// the run module can know.
    @Test func runUnderLeaseSetsIsGeneratingEagerlyWhileAgentStillIdle() async throws {
        let agent = makeAgent()
        let errors = ErrorRecorder()
        let run = AgentRunController(
            agent: agent,
            arbiter: makeFailingArbiter(),
            reportError: { errors.report($0) }
        )

        run.runUnderLease { /* never reached — lease throws first */ }

        // Eager: flag is up while the run is still queued and the agent idle.
        #expect(run.isGenerating == true)
        #expect(agent.state.phase == .idle)

        // The lease throws in ensureLoaded → the catch resets the flag.
        try await waitUntilIdle(run)
        #expect(run.isGenerating == false)
    }

    // MARK: - Lease path

    /// With the arbiter present and its model load failing, the lease throws
    /// before the body — so a `/compact`-shaped body never reaches `summarize`,
    /// `isGenerating` resets, and the failure surfaces through `reportError`.
    /// (Absorbs `compactCommandRunsUnderArbiterLeaseAndSkipsSummarizeWhenModelUnavailable`.)
    @Test func runUnderLeaseSkipsBodyAndReportsErrorWhenModelUnavailable() async throws {
        let agent = makeAgent()
        agent.loadMessages([Self.oldUser, Self.oldAssistant, Self.recentUser, Self.recentAssistant])

        let recorder = SummarizeRecorder()
        let summarize: @Sendable (String) async throws -> String = { _ in
            await recorder.record()
            return "summary"
        }
        let contextManager = ContextManager(settings: .small)
        let errors = ErrorRecorder()

        let run = AgentRunController(
            agent: agent,
            arbiter: makeFailingArbiter(),
            reportError: { errors.report($0) }
        )

        run.runUnderLease { [agent] in
            agent.forceCompact(contextManager: contextManager, contextWindow: 5_000, summarize: summarize)
            await agent.waitForIdle()
        }

        #expect(run.isGenerating == true)   // eager, synchronous
        try await waitUntilIdle(run)

        #expect(await recorder.callCount == 0)
        #expect(errors.messages.isEmpty == false)
    }

    // MARK: - Arbiter-less fallback

    /// Without an arbiter, `runUnderLease` drives the body directly (no lease) —
    /// so `summarize` still runs. Locks the back-compat branch so a future
    /// refactor cannot silently require an arbiter.
    /// (Absorbs `compactCommandFallsBackToDirectPathWhenArbiterMissing`.)
    @Test func runUnderLeaseFallsBackToDirectBodyWhenArbiterMissing() async throws {
        let agent = makeAgent()
        agent.loadMessages([Self.oldUser, Self.oldAssistant, Self.recentUser, Self.recentAssistant])

        let recorder = SummarizeRecorder()
        let summarize: @Sendable (String) async throws -> String = { _ in
            await recorder.record()
            return "## Goal\ncompacted"
        }
        let contextManager = ContextManager(settings: .small)

        let run = AgentRunController(
            agent: agent,
            arbiter: nil,
            reportError: { _ in }
        )

        run.runUnderLease { [agent] in
            agent.forceCompact(contextManager: contextManager, contextWindow: 5_000, summarize: summarize)
            await agent.waitForIdle()
        }

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

    // MARK: - send

    /// Arbiter-less `send` issues the prompt to the agent and raises the eager
    /// flag. The no-op agent finishes the stream; with no dispatcher wired, the
    /// flag stays up (only the event spine flips it down).
    @Test func sendIssuesPromptAndRaisesEagerFlag() async throws {
        let agent = makeAgent()
        let run = AgentRunController(agent: agent, arbiter: nil, reportError: { _ in })

        run.send(CoreMessage.user(UserMessage(content: "Hello")))
        #expect(run.isGenerating == true)

        // The direct body runs on the next main-actor tick — let it reach the agent.
        for _ in 0..<200 where agent.state.messages.isEmpty { await Task.yield() }
        #expect(agent.state.messages.contains { $0.asUser?.content == "Hello" })
    }

    // MARK: - cancel

    /// `cancelAndWait` drains the run and drives `isGenerating` to `false`.
    @Test func cancelAndWaitResetsIsGenerating() async throws {
        let agent = makeAgent()
        let run = AgentRunController(agent: agent, arbiter: nil, reportError: { _ in })

        run.send(CoreMessage.user(UserMessage(content: "Hi")))
        #expect(run.isGenerating == true)

        await run.cancelAndWait()
        #expect(run.isGenerating == false)
    }
}
