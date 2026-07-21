//
//  CompanionDigest.swift
//  tesseract
//
//  Mission Control's fold-down (ADR-0046 #373): the standing conversation
//  rides an 80k-token ceiling. Nightly — after ADR-0035 memory consolidation,
//  in the same sleep pass — the entity authors a Digest of its older history,
//  and a pure splice makes it the conversation's new head with the last K
//  turns kept verbatim. Hitting the ceiling intraday runs the same fold
//  early, on the record. One prefix-cache invalidation per day, at night, by
//  design; the conversation stays append-only between folds.
//
//  `CompanionDigestSplice` is the math (pure, directly tested);
//  `CompanionDigest` is the practice (the generation, the gates, the record).
//

import Foundation

// MARK: - The pure splice

nonisolated enum CompanionDigestSplice {

    /// How many most-recent turns survive a fold verbatim — the working set
    /// the digest never has to re-explain.
    static let verbatimTailTurns = 6

    /// Chars-per-token estimate. Coarse by design: the ceiling is a budget,
    /// not an invoice, and 4 chars/token is conservative for English prose.
    static let charsPerToken = 4

    /// What a fold needs, planned at read time: the history to digest, the
    /// identity of the tail's first message (the splice re-anchors on it
    /// against a fresh read, so turns that land mid-generation stay verbatim),
    /// and the size going in.
    struct FoldPlan {
        let older: [any AgentMessageProtocol & Sendable]
        let cutID: UUID
        let estimatedTokensBefore: Int
    }

    /// Turn openings are user messages — in Mission Control every user
    /// message is a fold-turn opening (the composer is refused there).
    static func turnStarts(_ messages: [any AgentMessageProtocol & Sendable]) -> [Int] {
        messages.indices.filter { messages[$0].asUser != nil }
    }

    /// Nil when there is nothing to fold: fewer than `keepLastTurns + 1`
    /// turns means the whole conversation IS the tail.
    static func plan(
        _ messages: [any AgentMessageProtocol & Sendable],
        keepLastTurns: Int = verbatimTailTurns
    ) -> FoldPlan? {
        let starts = turnStarts(messages)
        guard starts.count > keepLastTurns else { return nil }
        let cut = starts[starts.count - keepLastTurns]
        return FoldPlan(
            older: Array(messages[..<cut]),
            cutID: messages[cut].messageUUID,
            estimatedTokensBefore: estimatedTokens(messages))
    }

    /// The splice: digest head + everything from the cut message on,
    /// verbatim — including turns that appended after the plan was made.
    /// Nil when the cut message vanished (a competing fold already ran).
    static func splice(
        _ fresh: [any AgentMessageProtocol & Sendable],
        digest: String,
        cutID: UUID,
        tokensBefore: Int,
        at now: Date = Date()
    ) -> [any AgentMessageProtocol & Sendable]? {
        guard let cut = fresh.firstIndex(where: { $0.messageUUID == cutID }) else {
            return nil
        }
        let head = CompactionSummaryMessage(
            summary: digest, tokensBefore: tokensBefore, timestamp: now)
        return [head as any AgentMessageProtocol & Sendable] + fresh[cut...]
    }

    /// The ceiling's measuring stick — chars/4 over everything a message
    /// contributes to the prompt, plus a small per-message envelope. UTF-8
    /// counts, deliberately: O(1) per native string where grapheme counting
    /// walks the text — this runs on every loop tick as the ceiling's signal.
    static func estimatedTokens(_ messages: [any AgentMessageProtocol & Sendable]) -> Int {
        var chars = 0
        for message in messages {
            if let user = message.asUser {
                chars += user.content.utf8.count + (user.injectedContext?.utf8.count ?? 0)
            } else if let assistant = message.asAssistant {
                chars += assistant.text.utf8.count
                chars += assistant.thinking?.utf8.count ?? 0
            } else if let compaction = message as? CompactionSummaryMessage {
                chars += compaction.summary.utf8.count
            } else if let toolResult = message as? ToolResultMessage {
                for block in toolResult.content {
                    if case .text(let text) = block { chars += text.utf8.count }
                }
            }
            chars += 64  // role/framing envelope
        }
        return chars / charsPerToken
    }

    // MARK: - Rendering the older history for the digest prompt

    /// Per-message caps: the digest is about the whole, not the transcript.
    static let openingRenderCap = 1_600
    static let replyRenderCap = 2_400
    static let totalRenderCap = 60_000

    /// The older history as the digest author reads it. A previous digest
    /// renders whole and first; the rest renders chronologically with
    /// per-message caps, and the oldest excess is dropped under a notice —
    /// the previous digest already covers it.
    static func renderForDigest(_ older: [any AgentMessageProtocol & Sendable]) -> String {
        var previousDigest: String?
        var lines: [String] = []

        for message in older {
            if let compaction = message as? CompactionSummaryMessage {
                previousDigest = compaction.summary
            } else if let user = message.asUser {
                let body = capped(
                    strippingInstructionsBlock(user.content), at: openingRenderCap)
                guard !body.isEmpty else { continue }
                let origin = user.turnOrigin.map { "[\($0.rawValue)] " } ?? ""
                lines.append("\(origin)OPENING:\n\(body)")
            } else if let assistant = message.asAssistant {
                let body = capped(assistant.text, at: replyRenderCap)
                guard !body.isEmpty else { continue }
                lines.append("YOU:\n\(body)")
            } else if let toolResult = message as? ToolResultMessage {
                let text = toolResult.content.compactMap { block -> String? in
                    if case .text(let t) = block { return t }
                    return nil
                }.joined(separator: " ")
                guard !text.isEmpty else { continue }
                lines.append("[\(toolResult.toolName) → \(capped(text, at: 240))]")
            }
        }

        var body = lines.joined(separator: "\n\n")
        if body.count > totalRenderCap {
            body =
                "(…earlier history omitted — your previous digest covers it)\n\n"
                + String(body.suffix(totalRenderCap))
        }
        if let previousDigest {
            body = "YOUR PREVIOUS DIGEST:\n\(previousDigest)\n\n\(body)"
        }
        return body
    }

    /// The standing-instructions block rides every opening — folding it into
    /// the digest prompt N times would drown the signal.
    static func strippingInstructionsBlock(_ text: String) -> String {
        guard let start = text.range(of: "<companion-instructions"),
            let end = text.range(of: "</companion-instructions>"),
            start.lowerBound < end.upperBound
        else { return text }
        var out = text
        out.removeSubrange(start.lowerBound..<end.upperBound)
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func capped(_ text: String, at cap: Int) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count > cap else { return trimmed }
        return String(trimmed.prefix(cap)) + " (…)"
    }
}

// MARK: - The practice

@MainActor
final class CompanionDigest {

    /// A digest that comes back longer than this is noise, not memory — the
    /// fold is skipped and retried at the next pass.
    static let digestMaxLength = 10_000
    /// The nightly window, local hours [21, 04): the one planned
    /// prefix-cache invalidation lands at night, never mid-day.
    static func isNightHour(_ hour: Int) -> Bool { hour >= 21 || hour < 4 }
    /// A 02:00 fold belongs to the evening's night, not the new day's —
    /// shifting by the day-start floor keys one fold per night window.
    static func nightKey(for now: Date) -> String {
        TrackingDay.key(for: now.addingTimeInterval(-4 * 3600))
    }

    enum FoldOutcome { case folded, nothingToFold, failed }

    /// Concrete store, deliberately: `save(_:)` is the fold's one write door
    /// and is not on the chat seam (ADR-0046).
    private let conversationStore: AgentConversationStore
    private let store: MemoryStore
    private let recorder: CompanionFlightRecorder
    private let arbiter: any InferenceArbitrating
    private let complete: @Sendable (String) async throws -> String
    private let isEnabled: () -> Bool
    private var isFolding = false

    init(
        conversationStore: AgentConversationStore,
        store: MemoryStore,
        recorder: CompanionFlightRecorder,
        arbiter: any InferenceArbitrating,
        complete: @escaping @Sendable (String) async throws -> String,
        isEnabled: @escaping () -> Bool
    ) {
        self.conversationStore = conversationStore
        self.store = store
        self.recorder = recorder
        self.arbiter = arbiter
        self.complete = complete
        self.isEnabled = isEnabled
    }

    /// The sleep pass's tail (after consolidation and the instructions
    /// review): at most one fold per night window. A failed generation leaves
    /// the night unstamped so the next idle pass retries.
    func nightlyFold(now: Date = Date()) async {
        guard isEnabled() else { return }
        guard Self.isNightHour(Calendar.current.component(.hour, from: now)) else { return }
        let key = Self.nightKey(for: now)
        guard let day = try? await store.loopDayState(key), day.digestFoldAt == nil
        else { return }

        switch await fold(reason: "nightly", now: now) {
        case .folded, .nothingToFold:
            var updated = (try? await store.loopDayState(key)) ?? CompanionLoopDayState()
            updated.digestFoldAt = now
            try? await store.setLoopDayState(key, updated)
        case .failed:
            break
        }
    }

    /// The ceiling fold (the evaluator's `.compactFold` grant): same fold,
    /// intraday, on the record — no day gate, the ceiling is the gate.
    func earlyFold(now: Date = Date()) async {
        guard isEnabled() else { return }
        _ = await fold(reason: "ceiling", now: now)
    }

    /// One fold: plan → the entity authors the digest → splice → save. The
    /// whole sequence holds the model lease, and the splice re-reads and
    /// re-anchors on the plan's cut id — a turn that was queued behind this
    /// lease reads the folded conversation, never a stale one, and a turn
    /// that landed before the lease stays verbatim in the tail.
    private func fold(reason: String, now: Date) async -> FoldOutcome {
        guard !isFolding else { return .failed }
        isFolding = true
        defer { isFolding = false }

        do {
            return try await arbiter.withExclusiveGPU(.llm) {
                let missionControl = self.conversationStore.missionControl()
                guard let plan = CompanionDigestSplice.plan(missionControl.messages) else {
                    return .nothingToFold
                }

                let prompt = Self.digestPrompt(
                    older: CompanionDigestSplice.renderForDigest(plan.older))
                let digest = try await self.complete(prompt)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                guard !digest.isEmpty, digest.count <= Self.digestMaxLength else {
                    self.recorder.record(
                        .digestRejected,
                        snapshot: ["reason": reason, "length": String(digest.count)])
                    return .failed
                }

                let fresh = self.conversationStore.missionControl()
                guard
                    let spliced = CompanionDigestSplice.splice(
                        fresh.messages, digest: digest, cutID: plan.cutID,
                        tokensBefore: plan.estimatedTokensBefore, at: now)
                else { return .failed }

                var updated = fresh
                updated.messages = spliced
                self.conversationStore.save(updated)
                self.recorder.record(
                    .digestFolded,
                    conversationID: fresh.id,
                    snapshot: [
                        "reason": reason,
                        "tokensBefore": String(plan.estimatedTokensBefore),
                        "tokensAfter": String(
                            CompanionDigestSplice.estimatedTokens(spliced)),
                        "messages": String(spliced.count),
                    ])
                return .folded
            }
        } catch {
            Log.companion.error("Digest fold failed: \(error.localizedDescription)")
            self.recorder.record(.digestFailed, snapshot: ["reason": reason])
            return .failed
        }
    }

    static func digestPrompt(older: String) -> String {
        """
        You are Jarvis, folding your standing conversation — Mission Control, \
        your one cognitive record — down under its context ceiling. Below is \
        the OLDER history about to be folded away; your most recent turns stay \
        beside the digest verbatim, so do not re-tell them.

        <older-history>
        \(older)
        </older-history>

        Author the Digest that replaces this history. It becomes the head of \
        your conversation from now on — everything future-you must still know: \
        contracts and promises in flight (with their dates), wakes you have \
        reasoned about, his patterns and current state, unresolved threads, \
        standing decisions, anything you said you would remember. If a \
        previous digest appears above, merge it in and supersede it. Terse \
        prose and lists, in your own voice. Answer with the digest text only — \
        no preamble.
        """
    }
}
