import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// E11 memo tests: a memo hit must not re-probe, a stale or colliding entry
/// must degrade to a fresh two-probe detect (never a wrong boundary), and the
/// 256-entry eviction must keep the map bounded without breaking detection.
///
/// The memo is process-global static state; this suite is `@MainActor` like
/// the other detector suites, so all three serialize on the main actor and
/// `resetMemo()` in each suite's `init` cannot race another suite's detect.
@MainActor
struct StablePrefixDetectorMemoTests {

    // MARK: - Counting Mock Tokenizer

    /// ChatML-ish mock tokenizer (1 char = 1 token) that counts template
    /// calls, so tests can observe whether the memo spared the two probes.
    private final class CountingTokenizer: Tokenizer, @unchecked Sendable {
        let bosToken: String? = nil
        let eosToken: String? = "<|im_end|>"
        let unknownToken: String? = nil

        var templateCalls = 0

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            Array(text.utf8).map { Int($0) }
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            String(tokenIds.compactMap { UnicodeScalar($0) }.map { Character($0) })
        }

        func convertTokenToId(_ token: String) -> Int? {
            token.count == 1 ? Int(token.utf8.first!) : nil
        }

        func convertIdToToken(_ id: Int) -> String? {
            UnicodeScalar(id).map { String($0) }
        }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            templateCalls += 1
            var rendered = ""
            for msg in messages {
                let role = msg["role"] as? String ?? "unknown"
                let content = msg["content"] as? String ?? ""
                rendered += "<|im_start|>\(role)\n\(content)<|im_end|>\n"
            }
            return encode(text: rendered, addSpecialTokens: false)
        }
    }

    // MARK: - Helpers

    private let tokenizer = CountingTokenizer()

    init() {
        // The detector's memo is process-global — start every test clean.
        StablePrefixDetector.resetMemo()
    }

    /// Render a full conversation (counts as one template call).
    private func fullTokens(system: String, user: String) throws -> [Int] {
        try tokenizer.applyChatTemplate(
            messages: [
                ["role": "system", "content": system],
                ["role": "user", "content": user],
            ], tools: nil, additionalContext: nil)
    }

    // MARK: - Tests

    @Test func memoHitSkipsProbing() throws {
        let full = try fullTokens(system: "You are helpful.", user: "hi")

        let first = try StablePrefixDetector.detect(
            systemPrompt: "You are helpful.", toolSpecs: nil,
            fullTokens: full, tokenizer: tokenizer)
        #expect(first != nil)
        // 1 (full render above) + 2 (probes) — the memo was cold.
        #expect(tokenizer.templateCalls == 3)

        let second = try StablePrefixDetector.detect(
            systemPrompt: "You are helpful.", toolSpecs: nil,
            fullTokens: full, tokenizer: tokenizer)
        #expect(second == first)
        // Memo hit — no additional template renders.
        #expect(tokenizer.templateCalls == 3)
    }

    @Test func memoHitToleratesDifferentUserContent() throws {
        // Same stable prefix, different user turn: the stored prefix hash
        // still matches, so the second request must be served from the memo.
        let fullA = try fullTokens(system: "sys", user: "first question")
        let first = try StablePrefixDetector.detect(
            systemPrompt: "sys", toolSpecs: nil, fullTokens: fullA, tokenizer: tokenizer)
        #expect(first != nil)
        let callsAfterFirst = tokenizer.templateCalls

        let fullB = try fullTokens(system: "sys", user: "something else entirely")
        let second = try StablePrefixDetector.detect(
            systemPrompt: "sys", toolSpecs: nil, fullTokens: fullB, tokenizer: tokenizer)
        #expect(second == first)
        // Only fullB's render was added — no probes.
        #expect(tokenizer.templateCalls == callsAfterFirst + 1)
    }

    @Test func collidingEntryDegradesToFreshDetect() throws {
        let system = "shared system prompt"
        let full = try fullTokens(system: system, user: "hello")
        let detected = try StablePrefixDetector.detect(
            systemPrompt: system, toolSpecs: nil, fullTokens: full, tokenizer: tokenizer)
        let boundary = try #require(detected)
        #expect(boundary > 0)
        let callsAfterMemo = tokenizer.templateCalls

        // Same memo key (same system prompt) but fullTokens whose prefix does
        // NOT match the memoized one: the hit verification must reject the
        // entry, fall through to a fresh two-probe detect, and that detect
        // must fail its own verification — a wrong boundary is never returned.
        var tampered = full
        tampered[2] ^= 0xFF  // inside the stable prefix (im_start markup)
        let result = try StablePrefixDetector.detect(
            systemPrompt: system, toolSpecs: nil, fullTokens: tampered, tokenizer: tokenizer)
        #expect(result == nil)
        // The fall-through ran both probes.
        #expect(tokenizer.templateCalls == callsAfterMemo + 2)
    }

    @Test func evictionKeepsMemoBoundedAndCorrect() throws {
        // 260 distinct system prompts overflow the 256-entry memo; detection
        // must keep working throughout.
        for i in 0..<260 {
            let system = "system prompt variant \(i)"
            let full = try fullTokens(system: system, user: "q")
            let boundary = try StablePrefixDetector.detect(
                systemPrompt: system, toolSpecs: nil, fullTokens: full, tokenizer: tokenizer)
            #expect(boundary != nil)
        }
        let calls = tokenizer.templateCalls

        // The first variant was evicted → re-probes, still correct.
        let full = try fullTokens(system: "system prompt variant 0", user: "q2")
        let detected = try StablePrefixDetector.detect(
            systemPrompt: "system prompt variant 0", toolSpecs: nil,
            fullTokens: full, tokenizer: tokenizer)
        let boundary = try #require(detected)
        #expect(boundary > 0)
        // 1 (full render) + 2 (probes after eviction).
        #expect(tokenizer.templateCalls == calls + 3)
    }
}
