//
//  AgentRunControllerTests.swift
//  tesseractTests
//
//  Tests the **Agent Run** module at its own seam — no AgentCoordinator, no
//  conversation store, no transcript. Covers the single-writer `isGenerating`
//  with its eager "queued-behind-the-lease" semantics, and the lease contract
//  through the **Inference Arbitrating** peer: the happy path (acquire → body →
//  release) and the lease-throws path. Absorbs the former
//  `AgentCoordinatorCompactLeaseTests`, whose `/compact` lease coverage now
//  lives here via the shared `runUnderLease` entry.
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
    private static let oldAssistant = AssistantMessage(
        content: String(repeating: "B", count: 2_000))
    private static let recentUser = UserMessage(content: String(repeating: "C", count: 4_000))
    private static let recentAssistant = AssistantMessage(
        content: String(repeating: "D", count: 4_000))

    private func makeAgent() -> Agent {
        makeNoOpAgent(modelID: "agent-run-controller-test-model")
    }

    /// A peer whose lease throws deterministically before the body runs — the
    /// in-memory analogue of `ensureLoaded` failing with `modelNotDownloaded`.
    private func makeFailingArbiter() -> InMemoryInferenceArbiter {
        let peer = InMemoryInferenceArbiter()
        peer.ensureLoadedError = AgentEngineError.modelNotDownloaded(
            modelID: "agent-run-controller-nonexistent-model"
        )
        return peer
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

        run.runUnderLease { /* never reached — lease throws first */  }

        // Eager: flag is up while the run is still queued and the agent idle.
        #expect(run.isGenerating == true)
        #expect(agent.state.phase == .idle)

        // The lease throws in ensureLoaded → the catch resets the flag.
        try await waitUntilIdle(run)
        #expect(run.isGenerating == false)
    }

    // MARK: - Lease path

    /// The happy lease path — previously unreachable in tests: `runUnderLease`
    /// wraps its body in the `.llm` lease (the peer records the acquisition and
    /// runs the body), and `isGenerating` clears when the body completes.
    @Test func runUnderLeaseRunsBodyInsideLLMLease() async throws {
        let agent = makeAgent()
        let peer = InMemoryInferenceArbiter()
        let log = ErrorRecorder()
        let run = AgentRunController(
            agent: agent,
            arbiter: peer,
            reportError: { log.report($0) }
        )

        let bodyMark = ErrorRecorder()
        run.runUnderLease { bodyMark.report("body") }

        #expect(run.isGenerating == true)  // eager, while queued
        try await waitUntilIdle(run)

        #expect(bodyMark.messages == ["body"])
        #expect(peer.leaseCalls == [.init(slot: .llm, llmModelIDOverride: nil)])
        #expect(log.messages.isEmpty)
        #expect(run.isGenerating == false)
    }

    // MARK: - Vision requirement on the send path (ADR-0013)

    /// With "Use vision models when available" on (the default), the chat send
    /// path demands the vision container for any vision-capable model —
    /// `.visionIfCapable` — so attaching an image just works with no toggle and
    /// no re-prefill. Asserted at the **Inference Arbitrating** seam.
    @Test func runUnderLeaseRequestsVisionIfCapableWhenSettingOn() async throws {
        let agent = makeAgent()
        let peer = InMemoryInferenceArbiter()
        let settings = SettingsManager(store: InMemorySettingsStore())
        #expect(settings.useVisionWhenAvailable == true)  // default on
        let run = AgentRunController(
            agent: agent, arbiter: peer, settings: settings, reportError: { _ in }
        )

        run.runUnderLease {}
        try await waitUntilIdle(run)

        #expect(peer.leaseCalls == [.init(slot: .llm, llmVision: .visionIfCapable)])
    }

    /// With the vision opt-out off, the send path falls back to `.fromSettings`
    /// (→ the text-only container), so opting out truly forces text-only for chat.
    @Test func runUnderLeaseFallsBackToFromSettingsWhenVisionOptedOut() async throws {
        let agent = makeAgent()
        let peer = InMemoryInferenceArbiter()
        let settings = SettingsManager(store: InMemorySettingsStore())
        settings.useVisionWhenAvailable = false
        let run = AgentRunController(
            agent: agent, arbiter: peer, settings: settings, reportError: { _ in }
        )

        run.runUnderLease {}
        try await waitUntilIdle(run)

        #expect(peer.leaseCalls == [.init(slot: .llm, llmVision: .fromSettings)])
    }

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
            await agent.forceCompact(
                contextManager: contextManager, contextWindow: 5_000, summarize: summarize)
        }

        #expect(run.isGenerating == true)  // eager, synchronous
        try await waitUntilIdle(run)

        #expect(await recorder.callCount == 0)
        #expect(errors.messages.isEmpty == false)
    }

    // MARK: - /compact under the lease

    /// A `/compact`-shaped body runs to completion under the lease — `summarize`
    /// fires — and re-pins the standalone `/compact` busy-flag clearing to
    /// **body completion** (no event gate): once the awaited `forceCompact`
    /// finishes, `isGenerating` drops to `false`.
    @Test func compactBodyRunsUnderLeaseAndClearsFlagOnCompletion() async throws {
        let agent = makeAgent()
        agent.loadMessages([Self.oldUser, Self.oldAssistant, Self.recentUser, Self.recentAssistant])

        let recorder = SummarizeRecorder()
        let summarize: @Sendable (String) async throws -> String = { _ in
            await recorder.record()
            return "## Goal\ncompacted"
        }
        let contextManager = ContextManager(settings: .small)
        let peer = InMemoryInferenceArbiter()

        let run = AgentRunController(
            agent: agent,
            arbiter: peer,
            reportError: { _ in }
        )

        run.runUnderLease { [agent] in
            await agent.forceCompact(
                contextManager: contextManager, contextWindow: 5_000, summarize: summarize)
        }

        let deadline = ContinuousClock.now + .seconds(3)
        while await recorder.callCount == 0 {
            try await Task.sleep(for: .milliseconds(20))
            if ContinuousClock.now >= deadline {
                Issue.record("Lease body did not invoke summarize within timeout")
                break
            }
        }
        #expect(await recorder.callCount >= 1)
        #expect(peer.leaseCalls == [.init(slot: .llm)])

        // The completion-based clear: `/compact` owns its busy-flag lifecycle by
        // finishing under the lease, not via a `phase == .idle` event gate.
        try await waitUntilIdle(run)
        #expect(run.isGenerating == false)
    }

    // MARK: - send

    /// `send` issues the prompt to the agent and raises the eager flag. The
    /// no-op agent finishes the stream; with no dispatcher wired, the flag stays
    /// up (only the event spine flips it down).
    @Test func sendIssuesPromptAndRaisesEagerFlag() async throws {
        let agent = makeAgent()
        let run = AgentRunController(
            agent: agent, arbiter: InMemoryInferenceArbiter(), reportError: { _ in })

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
        let run = AgentRunController(
            agent: agent, arbiter: InMemoryInferenceArbiter(), reportError: { _ in })

        run.send(CoreMessage.user(UserMessage(content: "Hi")))
        #expect(run.isGenerating == true)

        await run.cancelAndWait()
        #expect(run.isGenerating == false)
    }

    // MARK: - Web-access gating of browser MCP tools (PRD #190, US #16)

    private func makeRegistry(extraToolNames: [String]) -> ToolRegistry {
        let host = ExtensionHost()
        host.register(StubToolsExtension(names: extraToolNames))
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mcp-gating-\(UUID().uuidString)", isDirectory: true)
        return ToolRegistry(sandbox: PathSandbox(root: root), extensionHost: host)
    }

    /// With web access off, the switch gates the built-in Browser server's MCP
    /// tools — the sole web surface now that search and fetch live under
    /// `browser.*` (ADR-0028) — while a non-browser MCP tool is untouched, so one
    /// switch keeps meaning what it says (#190, US #16).
    @Test func webAccessOffGatesBrowserMCPToolsButNotOtherServers() async throws {
        let agent = makeAgent()
        let settings = SettingsManager(store: InMemorySettingsStore())
        settings.webAccessEnabled = false
        let run = AgentRunController(
            agent: agent, arbiter: InMemoryInferenceArbiter(),
            toolRegistry: makeRegistry(extraToolNames: [
                "browser.navigate", "browser.read_page", "files.list",
            ]),
            settings: settings, reportError: { _ in })

        run.send(CoreMessage.user(UserMessage(content: "hi")))
        let names = Set(agent.state.tools.map(\.name))

        #expect(!names.contains("browser.navigate"))
        #expect(!names.contains("browser.read_page"))
        #expect(names.contains("files.list"))  // a non-browser MCP tool survives

        await run.cancelAndWait()
    }

    /// With web access on, browser MCP tools are present.
    @Test func webAccessOnKeepsBrowserMCPTools() async throws {
        let agent = makeAgent()
        let settings = SettingsManager(store: InMemorySettingsStore())
        settings.webAccessEnabled = true
        let run = AgentRunController(
            agent: agent, arbiter: InMemoryInferenceArbiter(),
            toolRegistry: makeRegistry(extraToolNames: ["browser.navigate", "browser.read_page"]),
            settings: settings, reportError: { _ in })

        run.send(CoreMessage.user(UserMessage(content: "hi")))
        let names = Set(agent.state.tools.map(\.name))

        #expect(names.contains("browser.navigate"))
        #expect(names.contains("browser.read_page"))

        await run.cancelAndWait()
    }
}

/// A minimal extension that exposes tools with the given names — enough to test
/// the run controller's web-access gating without standing up real tools.
private final class StubToolsExtension: AgentExtension, @unchecked Sendable {
    let path = "stub-tools"
    let commands: [String: RegisteredCommand] = [:]
    let handlers: [ExtensionEventType: [ExtensionEventHandler]] = [:]
    let tools: [String: AgentToolDefinition]

    init(names: [String]) {
        var built: [String: AgentToolDefinition] = [:]
        for name in names {
            built[name] = AgentToolDefinition(
                name: name, label: name, description: "",
                parameterSchema: JSONSchema(type: "object", properties: [:], required: []),
                execute: { _, _, _, _ in .text("") })
        }
        tools = built
    }
}
