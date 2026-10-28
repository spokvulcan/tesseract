import Foundation

// MARK: - CompactionSettings

/// Thresholds that control when and how context compaction runs.
nonisolated struct CompactionSettings: Sendable {
    /// Kill switch — set to false to disable compaction entirely.
    var enabled: Bool
    /// Token budget reserved for the next assistant reply + tool overhead.
    var reserveTokens: Int
    /// Minimum tokens to keep verbatim at the tail of the conversation.
    var keepRecentTokens: Int

    /// Default for Qwen3.5-family models (262,144 native context).
    /// Compaction triggers at ~245K tokens (262K - 16K reserve), keeping 20K
    /// recent verbatim. Reserve is sized for the common interactive output
    /// length; raise it toward 81,920 if long-form generations from the agent
    /// chat (not the HTTP server) start getting truncated by compaction.
    /// A window under 56,384 tokens fits it in proportion: LFM3-Audio's 32,768
    /// compacts past 23,247 tokens and keeps 11,623.
    static let standard = CompactionSettings(
        enabled: true,
        reserveTokens: 16_384,
        keepRecentTokens: 20_000
    )

    /// Conservative for smaller models (4–8K context).
    /// Compaction triggers at ~2K before the window fills, keeps 1K recent.
    static let small = CompactionSettings(
        enabled: true,
        reserveTokens: 2_048,
        keepRecentTokens: 1_024
    )

    /// The smallest window these thresholds fit as set: the reserve plus twice
    /// the recent tail, so a conversation that compacted grows by a tail's
    /// worth before it compacts again. A smaller window scales the reserve
    /// and the tail down in proportion.
    var smallestWindow: Int { reserveTokens + 2 * keepRecentTokens }

    /// The reserve in a window of `contextLength` tokens: as set, or scaled
    /// down in proportion in a window under `smallestWindow`.
    func reserveTokens(in contextLength: Int) -> Int {
        guard contextLength < smallestWindow else { return reserveTokens }
        return reserveTokens * contextLength / smallestWindow
    }

    /// The estimated tokens past which a conversation compacts in `window`:
    /// the window less its reserve, or the ceiling when that comes first.
    func compactionPoint(in window: CompactionWindow) -> Int {
        let point = window.contextLength - reserveTokens(in: window.contextLength)
        return min(point, window.ceiling ?? point)
    }

    /// The newest tokens a compaction in `window` keeps verbatim: as set, but
    /// never more than half the compaction point, so the next compaction is
    /// at least a tail's growth away.
    func keepRecentTokens(in window: CompactionWindow) -> Int {
        min(keepRecentTokens, compactionPoint(in: window) / 2)
    }
}

// MARK: - CompactionWindow

/// The room one conversation compacts in: the selected model's context window
/// and, for the Day Thread, its ceiling. Callers read it at every check, so a
/// model picked between turns compacts in its own window.
nonisolated struct CompactionWindow: Sendable, Equatable {
    /// The model's context window in tokens: its catalog entry's
    /// `contextLength`.
    var contextLength: Int
    /// The estimated tokens past which the conversation compacts even with
    /// room left in the window (the Day Thread's ceiling), or nil.
    var ceiling: Int?
}

// MARK: - ContextManager

/// Manages context compaction for the agent loop.
///
/// Tracks whether compaction should run and executes the summarize-and-trim
/// algorithm when the context window is nearly full. Designed to be used via
/// `makeTransformContext` which produces a closure compatible with `ContextTransformConfig`.
actor ContextManager {
    let settings: CompactionSettings
    private var lastSummary: String?

    init(settings: CompactionSettings) {
        self.settings = settings
    }

    // MARK: - Should Compact

    /// Returns true when estimated context tokens pass the compaction point
    /// in `window`.
    func shouldCompact(contextTokens: Int, window: CompactionWindow) -> Bool {
        guard settings.enabled else { return false }
        return contextTokens > settings.compactionPoint(in: window)
    }

    // MARK: - Compact

    /// Run compaction: find a cut point, summarize old messages, return the
    /// compacted array with a `CompactionSummaryMessage` prepended.
    ///
    /// - Parameters:
    ///   - messages: Current conversation messages.
    ///   - window: The room the conversation compacts in, which sets how
    ///     much of its tail stays verbatim.
    ///   - summarize: Async closure that sends a prompt to the LLM and returns the summary text.
    /// - Returns: Compacted message array.
    func compact(
        messages: [any AgentMessageProtocol],
        window: CompactionWindow,
        summarize: (String) async throws -> String
    ) async throws -> [any AgentMessageProtocol] {
        guard messages.count > 1 else { return messages }

        // 1. Find the cut point
        let cutIndex = findCutPoint(
            messages: messages, keepRecentTokens: settings.keepRecentTokens(in: window))
        guard cutIndex > 0 else { return messages }

        let oldMessages = Array(messages[..<cutIndex])
        let recentMessages = Array(messages[cutIndex...])

        // 2. Build summarization prompt
        let prompt = buildSummarizationPrompt(oldMessages)

        // 3. Call LLM for summary
        let summary = try await summarize(prompt)

        // 4. Store for incremental updates
        lastSummary = summary

        // 5. Build result
        let tokensBefore = TokenEstimator.estimateTotal(oldMessages)
        let summaryMessage = CompactionSummaryMessage(
            summary: summary,
            tokensBefore: tokensBefore
        )

        var result: [any AgentMessageProtocol] = [summaryMessage]
        result.append(contentsOf: recentMessages)
        return result
    }

    // MARK: - Cut Point Detection

    /// Walk backward from the newest message, accumulating token estimates.
    /// Stop when we've accumulated at least `keepRecentTokens`, then walk
    /// backward from that position to find the nearest valid cut point.
    ///
    /// Valid cut points: user, assistant, compaction_summary, custom messages.
    /// Never cut at a tool result (it must follow its tool call).
    ///
    /// Walking backward ensures we keep *at least* `keepRecentTokens` worth of
    /// recent messages — the cut may include more than the minimum if needed to
    /// avoid splitting a tool-call/tool-result pair.
    private func findCutPoint(messages: [any AgentMessageProtocol], keepRecentTokens: Int) -> Int {
        var accumulated = 0
        var candidateIndex = messages.count

        // Walk backward, accumulating tokens
        for i in stride(from: messages.count - 1, through: 0, by: -1) {
            accumulated += TokenEstimator.estimate(messages[i])
            if accumulated >= keepRecentTokens {
                candidateIndex = i
                break
            }
        }

        // If we walked all the way to the start without reaching the threshold,
        // there's nothing worth compacting.
        if candidateIndex == messages.count || candidateIndex == 0 {
            return 0
        }

        // Walk backward from candidateIndex to find a valid cut point.
        // This keeps at least keepRecentTokens (may keep more to avoid
        // splitting tool-call/tool-result pairs).
        for i in stride(from: candidateIndex, through: 1, by: -1) where !isToolResult(messages[i]) {
            return i
        }

        // No valid cut point found — don't compact
        return 0
    }

    /// Check if a message is a tool result, bare or wrapped.
    private func isToolResult(_ message: any AgentMessageProtocol) -> Bool {
        message.asToolResult != nil
    }

    // MARK: - Summarization Prompts

    /// Build the summarization prompt from messages being compacted.
    /// Uses the update prompt when a previous summary exists.
    ///
    /// When using the update prompt, `CompactionSummaryMessage` instances are
    /// filtered out of the formatted messages because the previous summary is
    /// already injected separately via `lastSummary`. Without this filter, the
    /// summary would appear twice, causing drift and bloat over successive
    /// compactions.
    private func buildSummarizationPrompt(
        _ messages: [any AgentMessageProtocol]
    ) -> String {
        if let previous = lastSummary {
            // Filter out prior compaction summaries — they're already represented
            // by `previous`. Only format the actual conversation messages.
            let nonSummary = messages.filter { !($0 is CompactionSummaryMessage) }
            let formatted = formatMessagesForSummary(nonSummary)

            return """
                Update the following summary with new information from the conversation.
                Keep the same structure. Merge, don't duplicate.

                Previous summary:
                \(previous)

                New conversation:
                \(formatted)
                """
        }

        let formatted = formatMessagesForSummary(messages)
        return """
            Summarize the following conversation history into a structured checkpoint.

            Format:
            ## Goal
            ## Constraints & Preferences
            ## Progress
            ### Done / In Progress / Blocked
            ## Key Decisions
            ## Next Steps
            ## Critical Context

            Focus on preserving actionable information. Be concise.

            ---

            \(formatted)
            """
    }

    /// Format messages as readable text for the summarization prompt.
    private func formatMessagesForSummary(
        _ messages: [any AgentMessageProtocol]
    ) -> String {
        var lines: [String] = []
        // The chat's own user turns arrive as `CoreMessage.user`: the unwrapping
        // accessors read them as they read a bare message.
        for message in messages {
            if let msg = message.asUser {
                // The summarizer reads text only: say an image was there, so
                // the summary keeps that it was shown.
                let images =
                    msg.images.isEmpty
                    ? ""
                    : "[\(msg.images.count) image\(msg.images.count == 1 ? "" : "s") attached] "
                lines.append("User: \(images)\(msg.content)")
            } else if let msg = message.asAssistant {
                if msg.text.isEmpty {
                    if !msg.toolCalls.isEmpty {
                        let names = msg.toolCalls.map(\.name).joined(separator: ", ")
                        lines.append("Assistant: [called tools: \(names)]")
                    }
                } else {
                    lines.append("Assistant: \(msg.text)")
                }
            } else if let msg = message.asToolResult {
                let prefix = msg.isError ? "Tool Error" : "Tool Result"
                lines.append("\(prefix) (\(msg.toolName)): \(msg.content.textContent)")
            } else if let msg = message as? CompactionSummaryMessage {
                lines.append("Previous Summary: \(msg.summary)")
            }
        }
        return lines.joined(separator: "\n")
    }
}

// MARK: - Transform Context Factory

/// Creates a `ContextTransformConfig` that plugs the compaction system into the
/// agent loop. The returned closure checks whether compaction should run and
/// executes it when needed. It reads `window` at every check, so a model
/// picked between turns compacts in its own window.
///
/// The `summarize` closure is injected so callers can route compaction through
/// the shared inference path without coupling compaction to a concrete engine.
nonisolated func makeCompactionTransform(
    contextManager: ContextManager,
    window: @escaping @Sendable () async -> CompactionWindow,
    summarize: @escaping @Sendable (String) async throws -> String
) -> ContextTransformConfig {
    ContextTransformConfig(
        reason: .compaction,
        transform: { messages, _ in
            let tokens = TokenEstimator.estimateTotal(messages)
            let current = await window()
            guard
                await contextManager.shouldCompact(
                    contextTokens: tokens,
                    window: current
                )
            else {
                return ContextTransformResult(
                    messages: messages, didMutate: false, reason: .compaction
                )
            }
            do {
                let compacted = try await contextManager.compact(
                    messages: messages,
                    window: current,
                    summarize: summarize
                )
                return ContextTransformResult(
                    messages: compacted, didMutate: true, reason: .compaction
                )
            } catch {
                Log.agent.error("Compaction failed: \(error.localizedDescription)")
                return ContextTransformResult(
                    messages: messages, didMutate: false, reason: .compaction
                )
            }
        }
    )
}

// MARK: - TokenEstimator

/// Coarse token estimation for compaction decisions.
///
/// Heuristic: ~4 characters per token (ceil division). Images — a user's
/// attachments and a tool's image blocks alike — are estimated at 4,800
/// characters (~1,200 tokens). This is intentionally conservative — actual
/// token counts from the LLM's `usage.totalTokens` can refine the estimate
/// during integration.
nonisolated enum TokenEstimator: Sendable {

    /// Characters per estimated image token payload (~1,200 tokens × 4 chars).
    private static let imageCharEstimate = 4_800

    /// Estimate token count for a string.
    static func estimate(_ text: String) -> Int {
        (text.utf8.count + 3) / 4
    }

    /// Estimate token count for a single message.
    static func estimate(_ message: any AgentMessageProtocol) -> Int {
        let chars = charCount(message)
        return (chars + 3) / 4
    }

    /// Estimate total tokens for a message array.
    static func estimateTotal(_ messages: [any AgentMessageProtocol]) -> Int {
        var total = 0
        for message in messages {
            total += estimate(message)
        }
        return total
    }

    // MARK: - Private

    /// Sum all text content characters in a message, and its images'
    /// estimate. The chat sends its user turns as `CoreMessage.user` and the
    /// agent loop keeps the wrapper, so the unwrapping accessors read it.
    private static func charCount(_ message: any AgentMessageProtocol) -> Int {
        if let msg = message.asUser {
            return msg.content.utf8.count + msg.images.count * imageCharEstimate
        }

        if let msg = message.asAssistant {
            var count = msg.text.utf8.count
            if let thinking = msg.thinking {
                count += thinking.utf8.count
            }
            for call in msg.toolCalls {
                count += call.name.utf8.count
                count += call.argumentsJSON.utf8.count
            }
            return count
        }

        if let msg = message.asToolResult {
            var count = 0
            for block in msg.content {
                switch block {
                case .text(let text):
                    count += text.utf8.count
                case .image:
                    count += imageCharEstimate
                }
            }
            return count
        }

        if let msg = message as? CompactionSummaryMessage {
            return msg.summary.utf8.count
        }
        return 0
    }
}
