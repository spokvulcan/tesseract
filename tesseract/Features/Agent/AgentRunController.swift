//
//  AgentRunController.swift
//  tesseract
//
//  The **Agent Run** module: the foreground-run envelope carved out of
//  `AgentCoordinator`. It owns the lifecycle of one foreground LLM invocation —
//  a `send` turn or a `/compact` — serialized behind the `InferenceArbiter`'s
//  exclusive `.llm` lease, and is the **single writer** of `isGenerating`.
//
//  `isGenerating` is set *eagerly* in `send`/`runUnderLease` — before
//  `agent.prompt` runs — because the run may sit **queued** behind the lease
//  while `agent.state.phase` is still `.idle`. The flag therefore means "a
//  foreground run is queued **or** active," a fact only this module knows.
//
//  Publisher-agnostic (ARCHITECTURE.md's panel-controller rule): the
//  coordinator's event dispatcher *feeds* the event-side transitions
//  (`markStarted`/`finish`); this module never subscribes to `AgentEvent`s
//  itself. A standalone `/compact` carries no `.agentEnd`, so it clears the flag
//  by completing its body under `runUnderLease`, not via an event gate.
//

import Foundation
import Observation

@Observable @MainActor
final class AgentRunController {

    // MARK: - Observable State

    /// "A foreground run is queued **or** active." Single-writer here; the
    /// coordinator re-exposes it as a computed passthrough so existing view
    /// reads (and Observation tracking) are unchanged.
    private(set) var isGenerating: Bool = false

    // MARK: - Dependencies

    private let agent: Agent
    private let arbiter: any InferenceArbitrating
    private let toolRegistry: ToolRegistry?
    private let settings: SettingsManager?
    /// Failures inside the lease task surface through the coordinator's shared
    /// `error` banner via this injected closure. Settable post-construction so the
    /// coordinator can wire it once `self` is fully initialized.
    @ObservationIgnored private var reportError: @MainActor (String) -> Void

    /// The task that holds the arbiter lease for the current run. Cancelled by
    /// `cancel()` to abort both queued waits and active runs.
    @ObservationIgnored private var sendTask: Task<Void, Never>?

    // MARK: - Init

    init(
        agent: Agent,
        arbiter: any InferenceArbitrating,
        toolRegistry: ToolRegistry? = nil,
        settings: SettingsManager? = nil,
        reportError: @escaping @MainActor (String) -> Void = { _ in }
    ) {
        self.agent = agent
        self.arbiter = arbiter
        self.toolRegistry = toolRegistry
        self.settings = settings
        self.reportError = reportError
    }

    /// Wire the report-error sink after construction (the coordinator injects its
    /// shared `error` banner once `self` is available).
    func setReportError(_ handler: @escaping @MainActor (String) -> Void) {
        reportError = handler
    }

    // MARK: - Command Side

    /// Begin a foreground turn for `message`. Syncs the active tool set for the
    /// current web-access setting, then drives `agent.prompt` under the lease.
    func send(_ message: any AgentMessageProtocol & Sendable) {
        syncToolsForWebAccess()
        runUnderLease { [agent] in
            agent.prompt(message)
            await agent.waitForIdle()
        }
    }

    /// Run `body` under the exclusive `.llm` lease, holding it until the body
    /// (and the in-agent work it awaits) finishes. Shared by `send` and
    /// `/compact` so the lease/flag/cancel contract is written once.
    ///
    /// `isGenerating` is set eagerly and cleared when the body completes (or on
    /// cancel/error), so a body that awaits its work to completion — a `send`
    /// turn *or* a `/compact` — owns the whole busy-flag lifecycle through one
    /// rule. (A `send` body also sees `.agentEnd` clear the flag via the event
    /// spine; the completion clear is idempotent with it.)
    func runUnderLease(_ body: @escaping @MainActor () async -> Void) {
        isGenerating = true

        // ADR-0013: when "Use vision models when available" is on (the default),
        // the chat send path demands the vision container for any vision-capable
        // model so attaching an image just works with no toggle and no re-prefill.
        // Off → `.fromSettings`, which the arbiter resolves against the same
        // opt-out (→ text-only). Absent settings (some tests) keep the
        // `.fromSettings` default so the lease contract is unchanged there.
        let visionReq: LLMVisionRequirement =
            (settings?.useVisionWhenAvailable ?? false) ? .visionIfCapable : .fromSettings

        sendTask = Task {
            do {
                try await arbiter.withExclusiveGPU(.llm, llmModelIDOverride: nil, llmVision: visionReq) {
                    await body()
                }
                // Body ran to completion under the lease — clear the busy flag.
                self.isGenerating = false
            } catch is CancellationError {
                // Cancelled while queued or during run — clean up.
                self.isGenerating = false
            } catch {
                self.reportError(error.localizedDescription)
                self.isGenerating = false
            }
            self.sendTask = nil
        }
    }

    /// Cancel the current run — aborts both a queued lease wait and an active run.
    func cancel() {
        sendTask?.cancel()
        sendTask = nil
        agent.abort()
    }

    /// Cancel and await full settle — used by app termination so generation is
    /// fully drained before shutdown.
    func cancelAndWait() async {
        let task = sendTask
        sendTask = nil
        task?.cancel()
        agent.abort()
        await task?.value
        await agent.waitForIdle()
        isGenerating = false
    }

    // MARK: - Event Side (fed by the coordinator's dispatcher)

    /// `.agentStart`: a run became active.
    func markStarted() {
        isGenerating = true
    }

    /// `.agentEnd`: the run finished. Also used by conversation-reset to clear
    /// the flag synchronously.
    func finish() {
        isGenerating = false
    }

    // MARK: - Private

    /// Filter active tools based on the `webAccessEnabled` setting. Called before
    /// each prompt so the LLM sees the current tool set.
    private func syncToolsForWebAccess() {
        guard let toolRegistry else { return }
        let allTools = toolRegistry.allTools
        if settings?.webAccessEnabled == true {
            agent.updateTools(allTools)
        } else {
            agent.updateTools(allTools.filter { $0.name != "web_search" && $0.name != "web_fetch" })
        }
    }
}
