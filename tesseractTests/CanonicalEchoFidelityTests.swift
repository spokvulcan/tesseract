import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// Contract tests for the **Canonical-Echo Fidelity** harness (PRD #94):
/// every boundary-leaf token path the server derives at a completed turn must
/// be a token-identical prefix of the next request's real render, across the
/// client-variation axes observed in the wild — reasoning as field vs
/// embedded `<think>`, trailing-whitespace trims, tool-call argument
/// ordering. Plus the harness's own teeth: a genuinely mutated echo must be
/// reported as a mismatch, not absorbed.
@MainActor
struct CanonicalEchoFidelityTests {
    private let tokenizer = FakeParoThinkingTokenizer()

    private func conversation(
        _ messages: [HTTPPrefixCacheMessage]
    ) -> HTTPPrefixCacheConversation {
        HTTPPrefixCacheConversation(systemPrompt: "You are a test.", messages: messages)
    }

    private let baseMessages: [HTTPPrefixCacheMessage] = [
        HTTPPrefixCacheMessage(role: .user, content: "What is the answer?")
    ]

    // MARK: - Stop answers (canonical-user boundary)

    @Test func stopAnswerLeafAndSpeculationAreFaithful() {
        let previous = conversation(baseMessages)
        let echo = HTTPPrefixCacheMessage.assistant(
            content: "The answer is 42.",
            reasoning: "Deep thought says 42."
        )
        let next = conversation(baseMessages + [
            echo,
            HTTPPrefixCacheMessage(role: .user, content: "And the question?"),
        ])

        let report = CanonicalEchoFidelity.check(
            previous: previous,
            echo: echo,
            next: next,
            probeToolSpecs: nil,
            nextToolSpecs: nil,
            requestIndex: 0,
            tokenizer: tokenizer
        )

        #expect(report.boundary.kind == .canonicalUser)
        guard case .faithful(let leafLength) = report.leaf else {
            Issue.record("leaf verdict: \(report.leaf)")
            return
        }
        guard case .faithful(let speculationLength) = report.speculation else {
            Issue.record("speculation verdict: \(String(describing: report.speculation))")
            return
        }
        // The speculation path runs strictly deeper than the canonical leaf:
        // through the stripped answer and the next user turn's header.
        #expect(speculationLength > leafLength)
    }

    @Test func embeddedThinkEchoRendersIdenticallyToFieldForm() {
        let previous = conversation(baseMessages)
        let fieldForm = HTTPPrefixCacheMessage.assistant(
            content: "The answer is 42.",
            reasoning: "Deep thought says 42."
        )
        let embeddedForm = HTTPPrefixCacheMessage.assistant(
            content: "<think>\nDeep thought says 42.\n</think>\n\nThe answer is 42."
        )
        let follow = HTTPPrefixCacheMessage(role: .user, content: "And the question?")

        // The *next request* echoes the embedded form while the stored turn
        // used the field form — the render must agree anyway.
        let next = conversation(baseMessages + [embeddedForm, follow])
        let report = CanonicalEchoFidelity.check(
            previous: previous,
            echo: fieldForm,
            next: next,
            probeToolSpecs: nil,
            nextToolSpecs: nil,
            requestIndex: 0,
            tokenizer: tokenizer
        )
        #expect(!report.hasMismatch, "field-form leaf must match embedded-form echo render")
    }

    @Test func trailingWhitespaceEchoStaysFaithful() {
        let previous = conversation(baseMessages)
        let stored = HTTPPrefixCacheMessage.assistant(
            content: "The answer is 42.\n\n\n",
            reasoning: "r"
        )
        let trimmedEcho = HTTPPrefixCacheMessage.assistant(
            content: "The answer is 42.",
            reasoning: "r"
        )
        let next = conversation(baseMessages + [
            trimmedEcho,
            HTTPPrefixCacheMessage(role: .user, content: "ok"),
        ])
        let report = CanonicalEchoFidelity.check(
            previous: previous,
            echo: stored,
            next: next,
            probeToolSpecs: nil,
            nextToolSpecs: nil,
            requestIndex: 0,
            tokenizer: tokenizer
        )
        #expect(!report.hasMismatch, "assistant-content trim normalization must hold")
    }

    @Test func mutatedEchoIsReportedAsMismatch() {
        let previous = conversation(baseMessages)
        let stored = HTTPPrefixCacheMessage.assistant(
            content: "The answer is 42, computed over 7.5 million years.",
            reasoning: "r"
        )
        // The client rewrites the answer tail — the admitted leaf would sit
        // past the fork, exactly the incident shape the gate must detect.
        let mutated = HTTPPrefixCacheMessage.assistant(
            content: "The answer is 42, computed instantly.",
            reasoning: "r"
        )
        let next = conversation(baseMessages + [
            mutated,
            HTTPPrefixCacheMessage(role: .user, content: "ok"),
        ])
        let report = CanonicalEchoFidelity.check(
            previous: previous,
            echo: stored,
            next: next,
            probeToolSpecs: nil,
            nextToolSpecs: nil,
            requestIndex: 0,
            tokenizer: tokenizer
        )
        guard case .mismatch(let length, let matched, let derivedTail, let nextTail) =
            report.speculation
        else {
            Issue.record("speculation verdict: \(String(describing: report.speculation))")
            return
        }
        #expect(matched < length)
        #expect(!derivedTail.isEmpty && !nextTail.isEmpty)
    }

    // MARK: - Tool-call turns (tool-continuation boundary)

    @Test func toolCallLeafIsFaithfulPrefixOfContinuation() {
        let previous = conversation(baseMessages)
        let echo = HTTPPrefixCacheMessage.assistant(
            content: "",
            reasoning: "Need to look this up.",
            toolCalls: [HTTPPrefixCacheToolCall(
                name: "read",
                argumentsJSON: #"{"filePath":"/tmp/answer.txt"}"#
            )]
        )
        let next = conversation(baseMessages + [
            echo,
            HTTPPrefixCacheMessage(role: .tool, content: "42"),
        ])
        let report = CanonicalEchoFidelity.check(
            previous: previous,
            echo: echo,
            next: next,
            probeToolSpecs: nil,
            nextToolSpecs: nil,
            requestIndex: 0,
            tokenizer: tokenizer
        )
        #expect(report.boundary.kind == .toolContinuation)
        #expect(report.speculation == nil)
        guard case .faithful = report.leaf else {
            Issue.record("leaf verdict: \(report.leaf)")
            return
        }
    }

    @Test func toolCallArgumentOrderDoesNotChangeTheRender() {
        let previous = conversation(baseMessages)
        let orderA = HTTPPrefixCacheMessage.assistant(
            content: "",
            reasoning: "edit",
            toolCalls: [HTTPPrefixCacheToolCall(
                name: "edit",
                argumentsJSON: #"{"filePath":"/a.txt","oldString":"x","newString":"y"}"#
            )]
        )
        let orderB = HTTPPrefixCacheMessage.assistant(
            content: "",
            reasoning: "edit",
            toolCalls: [HTTPPrefixCacheToolCall(
                name: "edit",
                argumentsJSON: #"{"newString":"y","oldString":"x","filePath":"/a.txt"}"#
            )]
        )
        // Canonical argument JSON makes the two messages equal — and the
        // render path must agree no matter which form the client echoed.
        #expect(orderA == orderB)
        let next = conversation(baseMessages + [
            orderB,
            HTTPPrefixCacheMessage(role: .tool, content: "ok"),
        ])
        let report = CanonicalEchoFidelity.check(
            previous: previous,
            echo: orderA,
            next: next,
            probeToolSpecs: nil,
            nextToolSpecs: nil,
            requestIndex: 0,
            tokenizer: tokenizer
        )
        #expect(!report.hasMismatch)
    }

    private let stretchHead = HTTPPrefixCacheMessage.assistant(
        content: "",
        reasoning: "Let me look at the file first.",
        toolCalls: [HTTPPrefixCacheToolCall(
            name: "read",
            argumentsJSON: #"{"filePath":"/tmp/answer.txt"}"#
        )]
    )

    @Test func interruptedStretchChecksRewindFloorAndSpeculationSpine() {
        // The immediate-interrupt shape: a tool-call turn whose next
        // request carries the steering user message directly — no tool
        // result landed. The whole stretch renders think-stripped; the
        // gate must check the rewind floor + speculation spine, not the
        // tool leaf, and both must be faithful.
        let previous = conversation(baseMessages)
        let next = conversation(baseMessages + [
            stretchHead,
            HTTPPrefixCacheMessage(role: .user, content: "Stop — try the other file instead."),
        ])
        let report = CanonicalEchoFidelity.check(
            previous: previous,
            echo: stretchHead,
            next: next,
            probeToolSpecs: nil,
            nextToolSpecs: nil,
            requestIndex: 0,
            tokenizer: tokenizer
        )

        #expect(report.boundary.kind == .interruptRewind)
        guard case .faithful(let floorLength) = report.leaf else {
            Issue.record("rewind floor verdict: \(report.leaf)")
            return
        }
        guard case .faithful(let spineLength) = report.speculation else {
            Issue.record("speculation verdict: \(String(describing: report.speculation))")
            return
        }
        // The speculation spine covers the think-stripped stretch and the
        // next user header — strictly deeper than the strip-floor path.
        #expect(spineLength > floorLength)
    }

    @Test func interruptTailWithToolResultsBindsOnlyTheFloor() {
        // The incident shape: the interrupt tail keeps the already-run
        // tool results *before* the steering message. The user-turn spine
        // has no exact counterpart (its trailing header tokens fork at
        // the BPE seam), so only the rewind floor binds.
        let previous = conversation(baseMessages)
        let next = conversation(baseMessages + [
            stretchHead,
            HTTPPrefixCacheMessage(role: .tool, content: "42"),
            HTTPPrefixCacheMessage(role: .user, content: "Stop — try the other file instead."),
        ])
        let report = CanonicalEchoFidelity.check(
            previous: previous,
            echo: stretchHead,
            next: next,
            probeToolSpecs: nil,
            nextToolSpecs: nil,
            requestIndex: 0,
            tokenizer: tokenizer
        )

        #expect(report.boundary.kind == .interruptRewind)
        #expect(report.speculation == nil)
        guard case .faithful = report.leaf else {
            Issue.record("rewind floor verdict: \(report.leaf)")
            return
        }
    }

    // MARK: - Session walk

    @Test func walkSessionReducesPairsAndRepairsDroppedReasoning() async {
        func openAI(_ role: OpenAI.ChatRole, _ content: String,
                    reasoning: String? = nil) -> OpenAI.ChatMessage {
            OpenAI.ChatMessage(role: role, content: .text(content), reasoning_content: reasoning)
        }
        let turn1 = "The answer is 42."
        let requests = [
            CanonicalEchoFidelity.RecordedRequest(
                messages: [openAI(.system, "You are a test."), openAI(.user, "q1")],
                tools: nil
            ),
            CanonicalEchoFidelity.RecordedRequest(
                messages: [
                    openAI(.system, "You are a test."), openAI(.user, "q1"),
                    openAI(.assistant, turn1, reasoning: "thinking hard"),
                    openAI(.user, "q2"),
                ],
                tools: nil
            ),
            // The client drops turn1's reasoning here — the walk's repair
            // store must recover it so the pair still reduces cleanly.
            CanonicalEchoFidelity.RecordedRequest(
                messages: [
                    openAI(.system, "You are a test."), openAI(.user, "q1"),
                    openAI(.assistant, turn1),
                    openAI(.user, "q2"),
                    openAI(.assistant, "Indeed.", reasoning: "confirming"),
                    openAI(.user, "q3"),
                ],
                tools: nil
            ),
        ]

        let report = await CanonicalEchoFidelity.walkSession(
            requests: requests,
            sessionAffinity: "ses_test",
            modelID: "fake-paro",
            tokenizer: tokenizer
        )

        #expect(report.boundaries.count == 2)
        #expect(report.skipped.isEmpty)
        #expect(report.mismatchCount == 0)
    }
}
