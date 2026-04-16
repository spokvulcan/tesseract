import Foundation
import Testing
@testable import Tesseract_Agent

/// Epic 3 Task 4 seam coverage — compaction is the last internal runtime
/// entry point that used to call `agentEngine.generate(...)` directly. After
/// the cutover, `ContextManager.compact` and `makeCompactionTransform` are the
/// only places that touch summarization, and they reach the LLM exclusively
/// through the injected `summarize` closure that the factories wire up via
/// `makeSummarizeClosure(...)`. These tests lock in that seam so a future
/// regression cannot reintroduce a direct engine dependency here.
@MainActor
struct ContextManagerCompactionTests {

    // MARK: - Fixtures

    /// Actor-isolated recorder so the summarize closure can capture its input
    /// across suspension points without violating Sendable.
    private actor SummarizeRecorder {
        private(set) var prompts: [String] = []
        func record(_ prompt: String) { prompts.append(prompt) }
    }

    /// Actor-isolated counter so repeated summarize invocations stay
    /// `@Sendable` without mutable closure captures.
    private actor CallCounter {
        private var count = 0
        func increment() -> Int { count += 1; return count }
    }

    /// Small synthetic context used for the direct-compact tests. Messages are
    /// sized so that the first two fit into the "compact" half and the last
    /// two fit into the "keep recent" half under `CompactionSettings.small`
    /// and a 5_000-token context window (reserve 2_048, keepRecent 1_024):
    /// older = 1_000 tokens, newer = 2_000 tokens, total ≈ 3_000 > 2_952.
    private static let oldUser = UserMessage(content: String(repeating: "A", count: 2_000))
    private static let oldAssistant = AssistantMessage(content: String(repeating: "B", count: 2_000))
    private static let recentUser = UserMessage(content: String(repeating: "C", count: 4_000))
    private static let recentAssistant = AssistantMessage(content: String(repeating: "D", count: 4_000))

    private static func overBudgetMessages() -> [any AgentMessageProtocol] {
        [oldUser, oldAssistant, recentUser, recentAssistant]
    }

    private static let contextWindow = 5_000

    // MARK: - Direct compact() seam

    /// Locks in that `ContextManager.compact` reaches the LLM strictly via the
    /// injected `summarize` closure. Post-cutover the factories always hand in
    /// `makeSummarizeClosure(...)` which targets `ServerInferenceService`, so
    /// proving this seam exists is equivalent to proving compaction runs on
    /// the server core — no extra wiring check is needed above this layer.
    @Test func compactInvokesInjectedSummarizeAndReturnsSummaryMessage() async throws {
        let manager = ContextManager(settings: .small)
        let recorder = SummarizeRecorder()
        let summary = "## Goal\ncompacted output"
        let summarize: @Sendable (String) async throws -> String = { prompt in
            await recorder.record(prompt)
            return summary
        }

        let compacted = try await manager.compact(
            messages: Self.overBudgetMessages(),
            contextWindow: Self.contextWindow,
            summarize: summarize
        )

        let prompts = await recorder.prompts
        #expect(prompts.count == 1)

        // The summarization prompt must carry the old messages that were cut,
        // not the recent ones. Anything else means the cut point migrated.
        let prompt = try #require(prompts.first)
        #expect(prompt.contains("Summarize the following conversation history"))
        #expect(prompt.contains(Self.oldUser.content))
        #expect(prompt.contains(Self.oldAssistant.content))
        #expect(!prompt.contains(Self.recentUser.content))

        // The first message must be the summary produced by the injected
        // closure — this is the only way LLM output reaches compacted state.
        let head = try #require(compacted.first as? CompactionSummaryMessage)
        #expect(head.summary == summary)
        #expect(head.tokensBefore > 0)

        // Recent messages survive verbatim in original order.
        let tail = compacted.dropFirst()
        #expect(tail.count == 2)
        #expect((tail.first as? UserMessage)?.content == Self.recentUser.content)
        #expect((tail.last as? AssistantMessage)?.content == Self.recentAssistant.content)
    }

    /// The update-prompt branch runs once a prior summary exists. It must
    /// still go through the same injected seam — no other path is allowed.
    @Test func compactUsesUpdatePromptOnceSummaryExistsStillViaInjectedSummarize() async throws {
        let manager = ContextManager(settings: .small)
        let recorder = SummarizeRecorder()
        let counter = CallCounter()
        let summarize: @Sendable (String) async throws -> String = { prompt in
            let n = await counter.increment()
            await recorder.record(prompt)
            return n == 1 ? "initial" : "updated"
        }

        _ = try await manager.compact(
            messages: Self.overBudgetMessages(),
            contextWindow: Self.contextWindow,
            summarize: summarize
        )
        _ = try await manager.compact(
            messages: Self.overBudgetMessages(),
            contextWindow: Self.contextWindow,
            summarize: summarize
        )

        let prompts = await recorder.prompts
        #expect(prompts.count == 2)
        #expect(prompts[0].contains("Summarize the following conversation history"))
        // The second call must be the update variant carrying the first summary.
        #expect(prompts[1].contains("Update the following summary"))
        #expect(prompts[1].contains("initial"))
    }

    /// A summarize failure must not be silently swallowed by compact. The
    /// transform layer is responsible for graceful fallback (covered below);
    /// the manager itself must let the caller observe the error.
    @Test func compactPropagatesSummarizeErrors() async throws {
        struct SummarizeFailure: Error, Equatable {}
        let manager = ContextManager(settings: .small)
        let summarize: @Sendable (String) async throws -> String = { _ in throw SummarizeFailure() }

        await #expect(throws: SummarizeFailure.self) {
            _ = try await manager.compact(
                messages: Self.overBudgetMessages(),
                contextWindow: Self.contextWindow,
                summarize: summarize
            )
        }
    }

    // MARK: - makeCompactionTransform seam

    /// When the context exceeds budget, the transform must route through the
    /// injected `summarize` closure and mark the result as mutated. This is
    /// the exact wiring `AgentFactory` and `BackgroundAgentFactory` use for
    /// the live chat and background agent paths.
    @Test func compactionTransformRoutesThroughInjectedSummarizeWhenOverBudget() async throws {
        let manager = ContextManager(settings: .small)
        let recorder = SummarizeRecorder()
        let summarize: @Sendable (String) async throws -> String = { prompt in
            await recorder.record(prompt)
            return "## Goal\ncompacted"
        }

        let transform = makeCompactionTransform(
            contextManager: manager,
            contextWindow: Self.contextWindow,
            summarize: summarize
        )

        #expect(transform.reason == .compaction)

        let result = await transform.transform(Self.overBudgetMessages(), nil)

        let prompts = await recorder.prompts
        #expect(prompts.count == 1)
        #expect(result.didMutate)
        #expect(result.reason == .compaction)
        let head = try #require(result.messages.first as? CompactionSummaryMessage)
        #expect(head.summary == "## Goal\ncompacted")
    }

    /// Below the budget, compaction must not fire. Guarantees we do not call
    /// the LLM on every turn — the server-core cutover did not widen this
    /// gate.
    @Test func compactionTransformSkipsSummarizeWhenUnderBudget() async throws {
        let manager = ContextManager(settings: .small)
        let recorder = SummarizeRecorder()
        let summarize: @Sendable (String) async throws -> String = { prompt in
            await recorder.record(prompt)
            return "should not run"
        }

        let transform = makeCompactionTransform(
            contextManager: manager,
            contextWindow: Self.contextWindow,
            summarize: summarize
        )

        let small: [any AgentMessageProtocol] = [
            UserMessage(content: "hi"),
            AssistantMessage(content: "hello"),
        ]
        let result = await transform.transform(small, nil)

        let prompts = await recorder.prompts
        #expect(prompts.isEmpty)
        #expect(!result.didMutate)
        #expect(result.reason == .compaction)
        #expect(result.messages.count == small.count)
    }

    /// When the injected summarize throws, the transform must preserve the
    /// original messages and report `didMutate: false`. This is the explicit
    /// graceful-fallback contract — without it an LLM error would corrupt the
    /// transcript on the agent chat path.
    @Test func compactionTransformFallsBackSilentlyWhenSummarizeThrows() async throws {
        struct SummarizeFailure: Error {}
        let manager = ContextManager(settings: .small)
        let summarize: @Sendable (String) async throws -> String = { _ in
            throw SummarizeFailure()
        }

        let transform = makeCompactionTransform(
            contextManager: manager,
            contextWindow: Self.contextWindow,
            summarize: summarize
        )

        let messages = Self.overBudgetMessages()
        let result = await transform.transform(messages, nil)

        #expect(!result.didMutate)
        #expect(result.reason == .compaction)
        #expect(result.messages.count == messages.count)
        #expect(!(result.messages.first is CompactionSummaryMessage))
    }

    /// Disabled settings short-circuit before touching the closure. Locks in
    /// that the kill-switch path exists and honors its promise regardless of
    /// which inference backend sits behind the closure.
    @Test func compactionTransformRespectsDisabledKillSwitch() async throws {
        let disabled = CompactionSettings(
            enabled: false,
            reserveTokens: CompactionSettings.small.reserveTokens,
            keepRecentTokens: CompactionSettings.small.keepRecentTokens
        )
        let manager = ContextManager(settings: disabled)
        let recorder = SummarizeRecorder()
        let summarize: @Sendable (String) async throws -> String = { prompt in
            await recorder.record(prompt)
            return "unreachable"
        }

        let transform = makeCompactionTransform(
            contextManager: manager,
            contextWindow: Self.contextWindow,
            summarize: summarize
        )

        let result = await transform.transform(Self.overBudgetMessages(), nil)

        let prompts = await recorder.prompts
        #expect(prompts.isEmpty)
        #expect(!result.didMutate)
    }
}
