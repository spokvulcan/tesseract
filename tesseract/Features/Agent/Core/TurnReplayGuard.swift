//
//  TurnReplayGuard.swift
//  tesseract
//
//  The agent loop's replay breaker. Local models can fall into a turn-level
//  attractor: a turn whose text already answered the user plus terminal
//  bookkeeping tool calls (track, report_back) leaves nothing to say once the
//  results return, and the highest-probability continuation is the model's own
//  previous turn — verbatim. Each replay re-executes the tools and reinforces
//  the pattern, so without a breaker the run loops until the owner hits stop
//  (observed live: 10 identical track+report_back turns on
//  qwen3.6-35b-a3b-paro, 2026-07-20).
//
//  The guard is deliberately narrow: only an *identical* consecutive turn —
//  same trimmed text, same tool calls with canonically-equal arguments — is a
//  replay. Paraphrases and same-tool-different-args turns pass untouched, so
//  legitimate retries ("fetch the same URL again") never trip it. Any owner
//  intervention (steering, follow-up) resets the chain: "do it again" is a
//  fresh request, not a replay.
//

import Foundation

/// Detects identical consecutive assistant turns within one agent run.
/// Pure value state — the loop owns one instance per run.
nonisolated struct TurnReplayGuard {

    /// Replays tolerated before the run is stopped: the first replay gets its
    /// tools refused with a corrective result (one chance to recover), the
    /// second proves the model is stuck.
    static let maxReplays = 2

    /// The corrective tool result a replayed turn's calls receive instead of
    /// re-execution. Written for the model: it must read as new information,
    /// or it would feed the very attractor it is meant to break.
    static let replayRefusal = """
        Repeated turn detected: this reply and its tool calls are identical to \
        your previous turn, which already executed — the results are above. Do \
        not send them again. Say anything that remains unsaid, or end the turn.
        """

    /// The user-facing notice when the run is stopped.
    static let terminationNotice = """
        Run stopped: the model repeated the identical turn \
        \(maxReplays + 1) times in a row.
        """

    enum Verdict: Equatable, Sendable {
        case fresh
        /// `consecutive` counts replays of the standing turn: 1 on the first
        /// identical repeat, 2 on the second.
        case replay(consecutive: Int)
    }

    private var previousSignature: String?
    private var consecutiveReplays = 0

    /// Classify the next assistant turn. Turns without tool calls reset the
    /// chain — the loop never asks about them (they end its inner loop on
    /// their own), but the guard's contract stays coherent for any caller.
    mutating func observe(_ message: AssistantMessage) -> Verdict {
        guard let signature = Self.signature(of: message) else {
            reset()
            return .fresh
        }
        if signature == previousSignature {
            consecutiveReplays += 1
            return .replay(consecutive: consecutiveReplays)
        }
        previousSignature = signature
        consecutiveReplays = 0
        return .fresh
    }

    /// A user intervened (steering or follow-up): the standing turn is no
    /// longer a replay candidate — repeating an answer on request is legitimate.
    mutating func reset() {
        previousSignature = nil
        consecutiveReplays = 0
    }

    /// The turn's identity for replay comparison: trimmed text plus each tool
    /// call's name and *canonicalized* arguments. Canonicalization matters —
    /// a replaying model re-emits argument JSON with shuffled key order (seen
    /// in the live transcript), and shuffled keys are still the same call.
    /// The canonicalizer is deliberately the prefix cache's: one definition
    /// of "the same arguments" across the app, not a second dialect.
    /// Thinking is deliberately excluded: it varies freely between otherwise
    /// identical turns, and the user-facing loop is text + calls.
    /// `nil` for turns without tool calls — they never continue the inner loop.
    static func signature(of message: AssistantMessage) -> String? {
        let calls = message.toolCalls
        guard !calls.isEmpty else { return nil }
        let text = (message.text ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let callSignatures = calls.map { call in
            "\(call.name)|\(canonicalizeHTTPPrefixCacheToolArgumentsJSON(call.argumentsJSON))"
        }
        return ([text] + callSignatures).joined(separator: "\u{1F}")
    }
}
