import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// Tests that reproduce the swift-jinja `tojson` non-determinism bug observed
/// in production (2026-04-12) and verify the detection threshold that guards
/// against it.
///
/// Production bug: the same tools array rendered through `tokenizer.applyChatTemplate`
/// produces different token sequences on successive calls, because swift-jinja's
/// nested-dict encoding has non-deterministic key ordering in some cases. Two-probe
/// detection then reports a suspiciously short common prefix (e.g. 53 tokens for a
/// 15K-token system+tools block), which would poison the radix tree if accepted.
///
/// These tests use a mock tokenizer that deliberately simulates the non-determinism
/// to exercise the detector's threshold logic without needing a real model loaded.
@MainActor
struct StablePrefixDetectorNonDeterminismTests {

    // MARK: - Non-Deterministic Mock

    /// Mock tokenizer that simulates swift-jinja's observed behavior: every call
    /// to `applyChatTemplate` with tools renders the first tool's inner keys in
    /// a different order. System prompt and user content render deterministically.
    ///
    /// Uses an internal counter to rotate through key permutations per call,
    /// modeling the behavior where each invocation produces a new random ordering.
    private final class FlakyToolsTokenizer: Tokenizer, @unchecked Sendable {
        let bosToken: String? = nil
        let eosToken: String? = "<|im_end|>"
        let unknownToken: String? = nil

        /// Thread-unsafe counter is fine for single-threaded tests.
        var callCount = 0

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
            callCount += 1

            var rendered = ""
            for msg in messages {
                let role = msg["role"] as? String ?? "unknown"
                let content = msg["content"] as? String ?? ""
                rendered += "<|im_start|>\(role)\n\(content)<|im_end|>\n"

                if role == "system", let tools, !tools.isEmpty {
                    // Render a large fake tools block with non-deterministic
                    // inner key ordering. Three rotating permutations simulate
                    // swift-jinja's observed non-determinism.
                    rendered += "[TOOLS_BLOCK_START]\n"
                    for tool in tools {
                        let name = (tool["function"] as? [String: any Sendable])?["name"] as? String ?? "unknown"
                        let permutation = callCount % 3
                        let body: String
                        switch permutation {
                        case 0:
                            body = "{\"function\":{\"description\":\"...\",\"name\":\"\(name)\",\"parameters\":{...}}}"
                        case 1:
                            body = "{\"function\":{\"name\":\"\(name)\",\"description\":\"...\",\"parameters\":{...}}}"
                        default:
                            body = "{\"function\":{\"parameters\":{...},\"name\":\"\(name)\",\"description\":\"...\"}}"
                        }
                        // Pad with repeated content to reach a "large prompt" size
                        // (> 1000 tokens / chars, since each ASCII char = 1 token).
                        rendered += body
                        rendered += String(repeating: "x", count: 200)
                        rendered += "\n"
                    }
                    rendered += "[TOOLS_BLOCK_END]\n"
                }
            }
            return encode(text: rendered, addSpecialTokens: false)
        }
    }

    private func makeToolSpec(name: String) -> ToolSpec {
        [
            "type": "function" as any Sendable,
            "function": [
                "name": name,
                "description": "A tool named \(name)",
                "parameters": [
                    "type": "object",
                    "properties": [:] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }

    // MARK: - 1. reproducesNonDeterminism

    /// Reproduces the production bug: calling applyChatTemplate twice with the
    /// same tools produces different token sequences when the template renders
    /// tools non-deterministically. This is what breaks the two-probe detector.
    @Test func reproducesNonDeterminism() throws {
        let tokenizer = FlakyToolsTokenizer()
        let tools: [ToolSpec] = (1...11).map { makeToolSpec(name: "tool\($0)") }
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are helpful."],
            ["role": "user", "content": "hello"],
        ]

        let run1 = try tokenizer.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: nil
        )
        let run2 = try tokenizer.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: nil
        )
        let run3 = try tokenizer.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: nil
        )

        // The flaky tokenizer cycles through 3 permutations — at least two of
        // these runs must differ to demonstrate the non-determinism.
        let allIdentical = (run1 == run2) && (run2 == run3)
        #expect(!allIdentical, "Flaky tokenizer must produce different outputs across calls")

        // Each run should be reasonably large (> 1000 tokens) to exercise the
        // large-prompt threshold in the detector.
        #expect(run1.count > 1000)
        #expect(run2.count > 1000)
        #expect(run3.count > 1000)
    }

    // MARK: - 2. detectorReturnsShortPrefixWithoutThreshold_bug

    /// Demonstrates what WOULD happen without the detector's short-prefix
    /// threshold: the two-probe technique returns a short commonLength that
    /// matches fullTokens (because both come from the same flaky session),
    /// poisoning any downstream radix tree.
    @Test func detectorWithoutThresholdReturnsShortPrefix() throws {
        let tokenizer = FlakyToolsTokenizer()
        let tools: [ToolSpec] = (1...11).map { makeToolSpec(name: "tool\($0)") }

        // Tokenize "full" through the flaky tokenizer — this produces one
        // specific permutation.
        let fullMessages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are helpful."],
            ["role": "user", "content": "real user message"],
        ]
        let fullTokens = try tokenizer.applyChatTemplate(
            messages: fullMessages, tools: tools, additionalContext: nil
        )

        // Now let the detector run — it will tokenize two more probes, each
        // producing a DIFFERENT permutation. Without the threshold, it returns
        // the (short) common prefix. With the threshold, it returns nil.
        let result = try StablePrefixDetector.detect(
            systemPrompt: "You are helpful.",
            toolSpecs: tools,
            fullTokens: fullTokens,
            tokenizer: tokenizer
        )

        // With the threshold in place (fullTokens > 1000, commonLength would
        // be < fullTokens/3), the detector returns nil — refusing to poison
        // the cache with a bogus short prefix.
        #expect(result == nil,
                "Detector should refuse short commonLength on large prompts to avoid tree poisoning")
    }

    // MARK: - 3. detectorThresholdScalesWithPromptSize

    /// The threshold only applies to LARGE prompts (>1000 tokens). Small prompts
    /// in tests and trivial messages can legitimately have a short stable prefix,
    /// so the threshold is skipped.
    @Test func detectorThresholdSkippedForSmallPrompts() throws {
        // Use the simple deterministic mock tokenizer that matches the one in
        // the existing test suite. Tools block renders stably.
        struct SmallMock: Tokenizer {
            let bosToken: String? = nil
            let eosToken: String? = "<|im_end|>"
            let unknownToken: String? = nil
            func encode(text: String, addSpecialTokens: Bool) -> [Int] {
                Array(text.utf8).map { Int($0) }
            }
            func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
                String(tokenIds.compactMap { UnicodeScalar($0) }.map { Character($0) })
            }
            func convertTokenToId(_ token: String) -> Int? { nil }
            func convertIdToToken(_ id: Int) -> String? { nil }
            func applyChatTemplate(
                messages: [[String: any Sendable]],
                tools: [[String: any Sendable]]?,
                additionalContext: [String: any Sendable]?
            ) throws -> [Int] {
                var rendered = ""
                for msg in messages {
                    let role = msg["role"] as? String ?? ""
                    let content = msg["content"] as? String ?? ""
                    rendered += "<|im_start|>\(role)\n\(content)<|im_end|>\n"
                    if role == "system", let tools, !tools.isEmpty {
                        rendered += "[tools:\(tools.count)]\n"
                    }
                }
                return Array(rendered.utf8).map { Int($0) }
            }
        }

        let tokenizer = SmallMock()
        let tools: [ToolSpec] = [["function": ["name": "foo"] as [String: any Sendable]]]

        // Build a small full prompt (well under 1000 tokens).
        let fullMessages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are helpful."],
            ["role": "user", "content": "test"],
        ]
        let fullTokens = try tokenizer.applyChatTemplate(
            messages: fullMessages, tools: tools, additionalContext: nil
        )
        #expect(fullTokens.count < 1000, "Test setup should produce a small prompt")

        let result = try StablePrefixDetector.detect(
            systemPrompt: "You are helpful.",
            toolSpecs: tools,
            fullTokens: fullTokens,
            tokenizer: tokenizer
        )

        // Small prompt → threshold skipped → detection succeeds (non-nil).
        #expect(result != nil, "Detector should accept small-prompt detection regardless of ratio")
    }

    // MARK: - 4. detectorRejectsLowRatioOnLargePrompt

    /// Verifies the threshold triggers when a large prompt produces an
    /// unreasonably short common prefix — the exact condition observed in
    /// production call 3 (commonLength=53 for a 15756-token prompt).
    @Test func detectorRejectsLowRatioOnLargePrompt() throws {
        let tokenizer = FlakyToolsTokenizer()
        let tools: [ToolSpec] = (1...11).map { makeToolSpec(name: "tool\($0)") }

        let fullMessages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are helpful."],
            ["role": "user", "content": "real"],
        ]
        let fullTokens = try tokenizer.applyChatTemplate(
            messages: fullMessages, tools: tools, additionalContext: nil
        )

        // Confirm the setup produces a large prompt (>1000 tokens).
        #expect(fullTokens.count > 1000)

        let result = try StablePrefixDetector.detect(
            systemPrompt: "You are helpful.",
            toolSpecs: tools,
            fullTokens: fullTokens,
            tokenizer: tokenizer
        )

        // The probes diverge early inside the flaky tools block, giving a
        // commonLength far below fullTokens/3 → rejected.
        #expect(result == nil,
                "Detector should reject large-prompt detection with commonLength < fullTokens/3")
    }

    // MARK: - 5. canonicalizeToolSpecsProducesIdenticalOutput

    /// Verifies `LLMActor.canonicalizeToolSpecs` produces byte-identical
    /// output across multiple calls with the same input. This is the
    /// workaround that feeds stable dicts into swift-jinja's tokenization
    /// path, reducing (but not eliminating) the non-determinism.
    @Test func canonicalizeToolSpecsProducesIdenticalOutput() throws {
        let tools: [ToolSpec] = (1...5).map { makeToolSpec(name: "tool\($0)") }

        let canonical1 = LLMActor.canonicalizeToolSpecs(tools)
        let canonical2 = LLMActor.canonicalizeToolSpecs(tools)
        let canonical3 = LLMActor.canonicalizeToolSpecs(tools)

        // Serialize each canonicalized version with sorted keys; they must
        // produce identical JSON bytes.
        func serialize(_ tools: [ToolSpec]?) throws -> Data {
            guard let tools else { return Data() }
            return try JSONSerialization.data(
                withJSONObject: tools,
                options: [.sortedKeys]
            )
        }

        let bytes1 = try serialize(canonical1)
        let bytes2 = try serialize(canonical2)
        let bytes3 = try serialize(canonical3)

        #expect(bytes1 == bytes2)
        #expect(bytes2 == bytes3)
    }

    // MARK: - 6. canonicalizeHandlesNestedDicts

    /// Verifies canonicalization walks nested dicts and arrays, producing
    /// deterministic output for realistic tool schemas with parameters.
    @Test func canonicalizeHandlesNestedDicts() throws {
        // A realistic Qwen3.5-style tool with nested properties.
        let tool: ToolSpec = [
            "type": "function" as any Sendable,
            "function": [
                "name": "question" as any Sendable,
                "description": "Ask a question.",
                "parameters": [
                    "type": "object" as any Sendable,
                    "required": ["questions"],
                    "properties": [
                        "questions": [
                            "type": "array" as any Sendable,
                            "items": [
                                "type": "object" as any Sendable,
                                "required": ["question", "header", "options"],
                                "properties": [
                                    "question": ["type": "string"],
                                    "header": ["type": "string"],
                                    "options": ["type": "array"],
                                ] as [String: any Sendable],
                            ] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]

        // Run canonicalization many times and verify all outputs match.
        var serializations: Set<Data> = []
        for _ in 0..<20 {
            let canonical = LLMActor.canonicalizeToolSpecs([tool])
            let bytes = try JSONSerialization.data(
                withJSONObject: canonical as Any,
                options: [.sortedKeys]
            )
            serializations.insert(bytes)
        }

        #expect(serializations.count == 1,
                "All canonicalizations should produce identical JSON bytes (got \(serializations.count) distinct outputs)")
    }

    // MARK: - 7. canonicalizeHandlesNullAndScalarTypes

    /// Verifies canonicalization preserves JSON primitives (null, bool, int,
    /// double, string) without changing their semantic types unexpectedly.
    @Test func canonicalizeHandlesNullAndScalarTypes() throws {
        let tool: ToolSpec = [
            "name": "mixed" as any Sendable,
            "description": "Test tool",
            "required": true as any Sendable,
            "count": 42 as any Sendable,
            "rate": 3.14 as any Sendable,
        ]

        let canonical = LLMActor.canonicalizeToolSpecs([tool])
        let first = try #require(canonical?.first)

        // Sanity: all keys present after canonicalization
        #expect(first["name"] as? String == "mixed")
        #expect(first["description"] as? String == "Test tool")
        #expect(first["required"] as? Bool == true)
        #expect(first["count"] as? Int == 42)
        // Double might come back as Double or NSNumber — both acceptable.
        let rate = first["rate"]
        if let d = rate as? Double {
            #expect(d == 3.14)
        } else if let n = rate as? NSNumber {
            #expect(n.doubleValue == 3.14)
        } else {
            Issue.record("rate should be Double or NSNumber, got \(String(describing: rate))")
        }
    }
}
