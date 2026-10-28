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
    private static let oldAssistant = AssistantMessage(
        content: String(repeating: "B", count: 2_000))
    private static let recentUser = UserMessage(content: String(repeating: "C", count: 4_000))
    private static let recentAssistant = AssistantMessage(
        content: String(repeating: "D", count: 4_000))

    private static func overBudgetMessages() -> [any AgentMessageProtocol] {
        [oldUser, oldAssistant, recentUser, recentAssistant]
    }

    private static let window = CompactionWindow(contextLength: 5_000)

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
            window: Self.window,
            summarize: summarize
        )

        let prompts = await recorder.prompts
        #expect(prompts.count == 1)

        // The summarization prompt must carry the old messages that were cut,
        // not the recent ones. Anything else means the cut point migrated.
        let prompt = try #require(prompts.first)
        #expect(prompt.contains("Summarize the following conversation history"))
        #expect(prompt.contains(Self.oldUser.content))
        #expect(prompt.contains(Self.oldAssistant.text))
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

    /// The update-prompt branch runs when the conversation compacted before:
    /// it starts with its own summary. It must still go through the same
    /// injected seam — no other path is allowed — and carry that summary
    /// once, even when another conversation compacted in between.
    @Test func compactingAConversationAgainUpdatesItsOwnSummary() async throws {
        let manager = ContextManager(settings: .small)
        let recorder = SummarizeRecorder()
        let counter = CallCounter()
        let summarize: @Sendable (String) async throws -> String = { prompt in
            let n = await counter.increment()
            await recorder.record(prompt)
            return n == 2 ? "summary of B" : "summary of A"
        }

        let conversationA = try await manager.compact(
            messages: Self.overBudgetMessages(),
            window: Self.window,
            summarize: summarize
        )
        // Another conversation compacts on the same manager in between.
        _ = try await manager.compact(
            messages: Self.turns(4, tokens: 1_000),
            window: Self.window,
            summarize: summarize
        )
        // A grows by two turns, then compacts again.
        _ = try await manager.compact(
            messages: conversationA + Self.turns(2, tokens: 1_000),
            window: Self.window,
            summarize: summarize
        )

        let prompts = await recorder.prompts
        try #require(prompts.count == 3)
        #expect(prompts[0].contains("Summarize the following conversation history"))
        // A's second compaction is the update variant carrying A's summary
        // once, as the summary to update, and never B's.
        let update = prompts[2]
        #expect(update.contains("Update the following summary"))
        #expect(update.contains("Previous summary:\nsummary of A\n"))
        #expect(update.components(separatedBy: "summary of A").count - 1 == 1)
        #expect(!update.contains("summary of B"))
        #expect(update.contains(Self.recentUser.content))
    }

    /// A conversation that never compacted gets the first-time prompt after
    /// another one compacted: one manager compacts every chat conversation
    /// and the Day Thread, and none of them updates another's summary.
    @Test func aFreshConversationGetsTheFirstTimePrompt() async throws {
        let manager = ContextManager(settings: .small)
        let recorder = SummarizeRecorder()
        let counter = CallCounter()
        let summarize: @Sendable (String) async throws -> String = { prompt in
            let n = await counter.increment()
            await recorder.record(prompt)
            return n == 1 ? "summary of A" : "summary of B"
        }

        _ = try await manager.compact(
            messages: Self.overBudgetMessages(),
            window: Self.window,
            summarize: summarize
        )
        _ = try await manager.compact(
            messages: Self.turns(4, tokens: 1_000),
            window: Self.window,
            summarize: summarize
        )

        let prompts = await recorder.prompts
        try #require(prompts.count == 2)
        #expect(prompts[1].contains("Summarize the following conversation history"))
        #expect(!prompts[1].contains("Update the following summary"))
        #expect(!prompts[1].contains("summary of A"))
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
                window: Self.window,
                summarize: summarize
            )
        }
    }

    // MARK: - makeCompactionTransform seam

    /// When the context exceeds budget, the transform must route through the
    /// injected `summarize` closure and mark the result as mutated. This is
    /// the exact wiring `AgentFactory` uses for the live chat path.
    @Test func compactionTransformRoutesThroughInjectedSummarizeWhenOverBudget() async throws {
        let manager = ContextManager(settings: .small)
        let recorder = SummarizeRecorder()
        let summarize: @Sendable (String) async throws -> String = { prompt in
            await recorder.record(prompt)
            return "## Goal\ncompacted"
        }

        let transform = makeCompactionTransform(
            contextManager: manager,
            window: { [window = Self.window] in window },
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
            window: { [window = Self.window] in window },
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
            window: { [window = Self.window] in window },
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
            window: { [window = Self.window] in window },
            summarize: summarize
        )

        let result = await transform.transform(Self.overBudgetMessages(), nil)

        let prompts = await recorder.prompts
        #expect(prompts.isEmpty)
        #expect(!result.didMutate)
    }

    // MARK: - What a user turn counts for

    private static let image = ImageAttachment(
        data: ImageTestFixtures.tinyPNGData, mimeType: "image/png")

    /// A user turn's images count as a tool result's do (~1,200 tokens each),
    /// and the chat's own user turn counts at all: `ChatSession` sends it as
    /// `CoreMessage.user`, and the agent loop keeps the wrapper in context.
    @Test func userImagesCountAndTheChatsWrappedTurnCounts() {
        let words = String(repeating: "W", count: 400)
        #expect(TokenEstimator.estimate(UserMessage(content: words)) == 100)
        let shown = UserMessage(content: words, images: [Self.image, Self.image])
        #expect(TokenEstimator.estimate(shown) == 100 + 2 * 1_200)
        #expect(TokenEstimator.estimate(CoreMessage.user(shown)) == 100 + 2 * 1_200)
        #expect(
            TokenEstimator.estimateTotal([CoreMessage.user(shown), Self.recentAssistant])
                == 2_500 + 1_000)
    }

    /// The summarizer reads the chat's wrapped user turns too, each image
    /// marked where it was, and each assistant turn as its text.
    @Test func summaryPromptCarriesTheChatsWrappedUserTurns() async throws {
        let manager = ContextManager(settings: .small)
        let recorder = SummarizeRecorder()
        let summarize: @Sendable (String) async throws -> String = { prompt in
            await recorder.record(prompt)
            return "## Goal\ncompacted"
        }
        let asked = UserMessage(content: Self.oldUser.content, images: [Self.image])

        _ = try await manager.compact(
            messages: [
                CoreMessage.user(asked), Self.oldAssistant, Self.recentUser,
                Self.recentAssistant,
            ],
            window: Self.window,
            summarize: summarize
        )

        let prompt = try #require(await recorder.prompts.first)
        #expect(prompt.contains("User: [1 image attached] \(asked.content)"))
        // An assistant turn is its text, not its parts' debug description.
        #expect(prompt.hasSuffix("Assistant: \(Self.oldAssistant.text)"))
        #expect(!prompt.contains("ContentPart"))
        #expect(!prompt.contains(Self.recentUser.content))
    }

    // MARK: - The selected model's window

    /// `count` turns, user and assistant in turn, of `tokens` estimated tokens
    /// each, every one its own letter ("a", "b", …).
    private static func turns(_ count: Int, tokens: Int) -> [any AgentMessageProtocol] {
        (0..<count).map { index in
            let text = String(
                repeating: Character(UnicodeScalar(UInt8(97 + index))), count: tokens * 4)
            let turn: any AgentMessageProtocol =
                index.isMultiple(of: 2)
                ? UserMessage(content: text) : AssistantMessage(content: text)
            return turn
        }
    }

    /// Qwen's 262,144-token window, and LFM3-VL's 128,000, keep the standard
    /// thresholds as set: compaction past the window less 16,384, with 20,000
    /// recent tokens kept.
    @Test func largeWindowsKeepTheStandardThresholds() {
        let standard = CompactionSettings.standard
        let qwen = CompactionWindow(contextLength: 262_144)
        #expect(standard.compactionPoint(in: qwen) == 245_760)
        #expect(standard.keepRecentTokens(in: qwen) == 20_000)
        let vision = CompactionWindow(contextLength: 128_000)
        #expect(standard.compactionPoint(in: vision) == 111_616)
        #expect(standard.keepRecentTokens(in: vision) == 20_000)
        #expect(standard.reserveTokens(in: standard.smallestWindow) == standard.reserveTokens)
    }

    /// LFM3-Audio's 32,768-token window fits the standard thresholds in
    /// proportion. As set, it would compact past 16,384 tokens and keep
    /// 20,000: no cut point until the conversation passed 20,000, then a
    /// compaction every turn.
    @Test func aSmallWindowFitsTheStandardThresholdsInProportion() {
        let standard = CompactionSettings.standard
        let audio = CompactionWindow(contextLength: 32_768)
        #expect(standard.reserveTokens(in: audio.contextLength) == 9_521)
        #expect(standard.compactionPoint(in: audio) == 23_247)
        #expect(standard.keepRecentTokens(in: audio) == 11_623)
    }

    /// Past a 32K window's compaction point the transform compacts once and
    /// keeps the newest turns verbatim, and the compacted conversation has
    /// room to grow: the next turn's check leaves it alone.
    @Test func aSmallWindowCompactsOnceAndKeepsARecentTail() async throws {
        let manager = ContextManager(settings: .standard)
        let recorder = SummarizeRecorder()
        let summarize: @Sendable (String) async throws -> String = { prompt in
            await recorder.record(prompt)
            return "## Goal\ncompacted"
        }
        let transform = makeCompactionTransform(
            contextManager: manager,
            window: { CompactionWindow(contextLength: 32_768) },
            summarize: summarize
        )
        // Twelve 2,000-token turns: 24,000 tokens, past the 23,247 point.
        let turns = Self.turns(12, tokens: 2_000)

        let compacted = await transform.transform(turns, nil)

        #expect(compacted.didMutate)
        #expect(compacted.messages.first is CompactionSummaryMessage)
        // The six newest turns, 12,000 tokens, cover the 11,623 kept.
        let tail = Array(compacted.messages.dropFirst())
        #expect(tail.count == 6)
        #expect((tail.first as? UserMessage)?.content == (turns[6] as? UserMessage)?.content)
        #expect(TokenEstimator.estimateTotal(tail) == 12_000)

        let next = await transform.transform(
            compacted.messages + Self.turns(1, tokens: 2_000), nil)
        #expect(!next.didMutate)
        #expect(await recorder.prompts.count == 1)
    }

    /// The transform reads its window at every check: a conversation that
    /// fits Qwen's window compacts once the owner picks a 32K model.
    @Test func theTransformReadsTheWindowAtEveryCheck() async throws {
        let manager = ContextManager(settings: .standard)
        let recorder = SummarizeRecorder()
        let summarize: @Sendable (String) async throws -> String = { prompt in
            await recorder.record(prompt)
            return "## Goal\ncompacted"
        }
        let selected = Locked(CompactionWindow(contextLength: 262_144))
        let transform = makeCompactionTransform(
            contextManager: manager,
            window: { selected.value },
            summarize: summarize
        )
        let turns = Self.turns(12, tokens: 2_000)

        let underQwen = await transform.transform(turns, nil)
        #expect(!underQwen.didMutate)

        selected.value = CompactionWindow(contextLength: 32_768)
        let underAudio = await transform.transform(turns, nil)
        #expect(underAudio.didMutate)
        #expect(await recorder.prompts.count == 1)
    }

    /// A ceiling (the Day Thread's) compacts before the window's own point
    /// but never after it, and the tail stays at most half the point: a
    /// 20,000 ceiling keeps 10,000, so it does not compact again every turn.
    @Test func aCeilingComesFirstButNeverPastTheWindow() {
        let standard = CompactionSettings.standard
        let thread = CompactionWindow(contextLength: 262_144, ceiling: 64_000)
        #expect(standard.compactionPoint(in: thread) == 64_000)
        #expect(standard.keepRecentTokens(in: thread) == 20_000)
        let small = CompactionWindow(contextLength: 32_768, ceiling: 64_000)
        #expect(standard.compactionPoint(in: small) == 23_247)
        #expect(standard.keepRecentTokens(in: small) == 11_623)
        let low = CompactionWindow(contextLength: 262_144, ceiling: 20_000)
        #expect(standard.compactionPoint(in: low) == 20_000)
        #expect(standard.keepRecentTokens(in: low) == 10_000)
    }
}
