import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// Task 1.3 tests: stable-prefix boundary detection via two-probe technique.
@MainActor
struct StablePrefixDetectorTests {

    // MARK: - Mock Tokenizer

    /// Deterministic mock tokenizer implementing a ChatML-like template.
    /// Each ASCII character maps to its codepoint as a token ID.
    /// Template format:
    ///   <|im_start|>system\n{content}<|im_end|>\n
    ///   [tools:{json}]\n        (if tools present)
    ///   <|im_start|>user\n{content}<|im_end|>\n
    private struct MockTokenizer: Tokenizer {
        let bosToken: String? = nil
        let eosToken: String? = "<|im_end|>"
        let unknownToken: String? = nil

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
            var rendered = ""
            for msg in messages {
                let role = msg["role"] as? String ?? "unknown"
                let content = msg["content"] as? String ?? ""
                rendered += "<|im_start|>\(role)\n\(content)<|im_end|>\n"

                // Insert tools + additionalContext after system message
                if role == "system" {
                    if let tools, !tools.isEmpty {
                        let toolNames = tools.compactMap { tool -> String? in
                            guard let fn = tool["function"] as? [String: any Sendable] else {
                                return nil
                            }
                            return fn["name"] as? String
                        }
                        rendered += "[tools:\(toolNames.joined(separator: ","))]\n"
                    }
                    if let ctx = additionalContext,
                       let extra = ctx["prefix_text"] as? String
                    {
                        rendered += "[ctx:\(extra)]\n"
                    }
                }
            }
            return encode(text: rendered, addSpecialTokens: false)
        }
    }

    // MARK: - Helpers

    private let tokenizer = MockTokenizer()

    /// Tokenize a full conversation through the mock template for verification.
    private func tokenizeFull(
        systemPrompt: String,
        userContent: String,
        toolSpecs: [ToolSpec]? = nil
    ) throws -> [Int] {
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": systemPrompt],
            ["role": "user", "content": userContent],
        ]
        return try tokenizer.applyChatTemplate(messages: messages, tools: toolSpecs)
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

    // MARK: - 1. detectsSystemPlusToolsBoundary

    @Test func detectsSystemPlusToolsBoundary() throws {
        let tools: [ToolSpec] = (1...5).map { makeToolSpec(name: "tool\($0)") }
        let fullTokens = try tokenizeFull(
            systemPrompt: "You are an assistant.",
            userContent: "Hello there!",
            toolSpecs: tools
        )

        let boundary = try StablePrefixDetector.detect(
            systemPrompt: "You are an assistant.",
            toolSpecs: tools,
            fullTokens: fullTokens,
            tokenizer: tokenizer
        )

        let offset = try #require(boundary)
        // Boundary must be past the system message AND tool definitions
        // but before the user content
        let prefix = tokenizer.decode(tokenIds: Array(fullTokens[0..<offset]))
        #expect(prefix.contains("You are an assistant."))
        #expect(prefix.contains("[tools:"))
        #expect(!prefix.contains("Hello there!"))
    }

    // MARK: - 2. noSystemMessageReturnsNil

    @Test func noSystemMessageReturnsNil() throws {
        let result = try StablePrefixDetector.detect(
            systemPrompt: nil,
            toolSpecs: nil,
            fullTokens: [1, 2, 3],
            tokenizer: tokenizer
        )
        #expect(result == nil)
    }

    @Test func emptySystemMessageReturnsNil() throws {
        let result = try StablePrefixDetector.detect(
            systemPrompt: "",
            toolSpecs: nil,
            fullTokens: [1, 2, 3],
            tokenizer: tokenizer
        )
        #expect(result == nil)
    }

    // MARK: - 3. stablePrefixIsPrefixOfFullSequence

    @Test func stablePrefixIsPrefixOfFullSequence() throws {
        let tools: [ToolSpec] = [makeToolSpec(name: "read"), makeToolSpec(name: "write")]
        let fullTokens = try tokenizeFull(
            systemPrompt: "System prompt here.",
            userContent: "User request here.",
            toolSpecs: tools
        )

        let detected = try StablePrefixDetector.detect(
            systemPrompt: "System prompt here.",
            toolSpecs: tools,
            fullTokens: fullTokens,
            tokenizer: tokenizer
        )
        let boundary = try #require(detected)

        // Verify: probeA[0..<boundary] == fullTokens[0..<boundary]
        // (This is the verification step inside detect() — confirm it passed)
        let probeA = try tokenizeFull(
            systemPrompt: "System prompt here.",
            userContent: "A_prefix_probe",
            toolSpecs: tools
        )
        #expect(Array(probeA[0..<boundary]) == Array(fullTokens[0..<boundary]))
    }

    // MARK: - 4. noToolsDetectsSystemOnlyBoundary

    @Test func noToolsDetectsSystemOnlyBoundary() throws {
        let fullTokens = try tokenizeFull(
            systemPrompt: "You are helpful.",
            userContent: "What is 2+2?"
        )

        let detected = try StablePrefixDetector.detect(
            systemPrompt: "You are helpful.",
            toolSpecs: nil,
            fullTokens: fullTokens,
            tokenizer: tokenizer
        )
        let boundary = try #require(detected)

        let prefix = tokenizer.decode(tokenIds: Array(fullTokens[0..<boundary]))
        #expect(prefix.contains("You are helpful."))
        #expect(!prefix.contains("What is 2+2?"))
    }

    // MARK: - 5. longSystemPrompt8KTokens

    @Test func longSystemPrompt8KTokens() throws {
        // ~8K characters ≈ 8K tokens in our 1-char-per-token mock
        let longSystem = String(repeating: "x", count: 8000)
        let tools: [ToolSpec] = [makeToolSpec(name: "search")]
        let fullTokens = try tokenizeFull(
            systemPrompt: longSystem,
            userContent: "query",
            toolSpecs: tools
        )

        let detected = try StablePrefixDetector.detect(
            systemPrompt: longSystem,
            toolSpecs: tools,
            fullTokens: fullTokens,
            tokenizer: tokenizer
        )
        let boundary = try #require(detected)

        // Boundary must be past the 8K system prompt + tool section
        #expect(boundary > 8000)
        #expect(boundary < fullTokens.count)
    }

    // MARK: - 6. differentToolsProduceDifferentBoundaries

    @Test func differentToolsProduceDifferentBoundaries() throws {
        let system = "Agent system prompt."

        let toolsSmall: [ToolSpec] = [makeToolSpec(name: "ls")]
        let toolsLarge: [ToolSpec] = (1...10).map { makeToolSpec(name: "tool_\($0)") }

        let fullSmall = try tokenizeFull(
            systemPrompt: system, userContent: "hi", toolSpecs: toolsSmall)
        let fullLarge = try tokenizeFull(
            systemPrompt: system, userContent: "hi", toolSpecs: toolsLarge)

        let detectedSmall = try StablePrefixDetector.detect(
            systemPrompt: system, toolSpecs: toolsSmall,
            fullTokens: fullSmall, tokenizer: tokenizer
        )
        let detectedLarge = try StablePrefixDetector.detect(
            systemPrompt: system, toolSpecs: toolsLarge,
            fullTokens: fullLarge, tokenizer: tokenizer
        )
        let boundarySmall = try #require(detectedSmall)
        let boundaryLarge = try #require(detectedLarge)

        // More tools → longer tool section → larger boundary
        #expect(boundaryLarge > boundarySmall)
    }

    // MARK: - 7. twoProbesDivergeAtUserContent

    @Test func twoProbesDivergeAtUserContent() throws {
        let system = "Be concise."
        let tools: [ToolSpec] = [makeToolSpec(name: "read")]

        let probeA = try tokenizeFull(
            systemPrompt: system, userContent: "A_prefix_probe", toolSpecs: tools)
        let probeB = try tokenizeFull(
            systemPrompt: system, userContent: "Z_prefix_probe", toolSpecs: tools)

        // Common prefix: exact match up to divergence point
        let commonLength = zip(probeA, probeB).prefix(while: ==).count
        #expect(commonLength > 0)

        // After divergence, tokens differ (user content)
        #expect(probeA[commonLength] != probeB[commonLength])

        // The common prefix includes system + tools + user turn markup
        // but NOT user content
        let prefix = tokenizer.decode(tokenIds: Array(probeA[0..<commonLength]))
        #expect(prefix.contains("Be concise."))
        #expect(prefix.contains("[tools:"))
        #expect(!prefix.contains("A_prefix"))
        #expect(!prefix.contains("Z_prefix"))
    }

    // MARK: - 8. probeRobustToSpecialCharsInSystem

    @Test func probeRobustToSpecialCharsInSystem() throws {
        // System prompt with quotes, newlines, template-like markers
        let system = """
            You are "an assistant".
            Rules:
            - Don't use <|im_start|> in responses
            - Handle {curly braces} and [brackets]
            """
        let fullTokens = try tokenizeFull(
            systemPrompt: system, userContent: "test")

        let detected = try StablePrefixDetector.detect(
            systemPrompt: system, toolSpecs: nil,
            fullTokens: fullTokens, tokenizer: tokenizer
        )
        let boundary = try #require(detected)

        let prefix = tokenizer.decode(tokenIds: Array(fullTokens[0..<boundary]))
        #expect(prefix.contains("an assistant"))
        #expect(!prefix.contains("test"))
    }

    // MARK: - Edge cases

    @Test func verificationFailsOnMismatchedFullTokens() throws {
        // fullTokens from a DIFFERENT system prompt → verification fails → nil
        let fullTokens = try tokenizeFull(
            systemPrompt: "Different system.", userContent: "hello")

        let result = try StablePrefixDetector.detect(
            systemPrompt: "Original system.",
            toolSpecs: nil,
            fullTokens: fullTokens,
            tokenizer: tokenizer
        )
        #expect(result == nil)
    }

    @Test func fullTokensShorterThanBoundaryReturnsNil() throws {
        // Provide truncated fullTokens shorter than what the boundary would be
        let result = try StablePrefixDetector.detect(
            systemPrompt: "A system prompt that produces many tokens.",
            toolSpecs: nil,
            fullTokens: [1, 2, 3], // way too short
            tokenizer: tokenizer
        )
        #expect(result == nil)
    }

    @Test func emptyToolArrayTreatedAsNoTools() throws {
        let system = "Hello."
        let withNil = try tokenizeFull(systemPrompt: system, userContent: "q", toolSpecs: nil)
        let withEmpty = try tokenizeFull(systemPrompt: system, userContent: "q", toolSpecs: [])

        let boundaryNil = try StablePrefixDetector.detect(
            systemPrompt: system, toolSpecs: nil,
            fullTokens: withNil, tokenizer: tokenizer
        )
        let boundaryEmpty = try StablePrefixDetector.detect(
            systemPrompt: system, toolSpecs: [],
            fullTokens: withEmpty, tokenizer: tokenizer
        )

        // Both should produce the same boundary (no tools section rendered)
        #expect(boundaryNil == boundaryEmpty)
    }

    // MARK: - additionalContext forwarding

    @Test func additionalContextIncludedInStablePrefix() throws {
        let system = "You are helpful."
        let ctx: [String: any Sendable] = ["prefix_text": "extra_context_data"]

        // Full tokenization with additionalContext
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": system],
            ["role": "user", "content": "hello"],
        ]
        let fullTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: ctx
        )

        // Detection with matching additionalContext → succeeds
        let detected = try StablePrefixDetector.detect(
            systemPrompt: system, toolSpecs: nil, additionalContext: ctx,
            fullTokens: fullTokens, tokenizer: tokenizer
        )
        let boundary = try #require(detected)

        let prefix = tokenizer.decode(tokenIds: Array(fullTokens[0..<boundary]))
        #expect(prefix.contains("extra_context_data"))
        #expect(!prefix.contains("hello"))
    }

    @Test func missingAdditionalContextCausesVerificationFailure() throws {
        let system = "You are helpful."
        let ctx: [String: any Sendable] = ["prefix_text": "injected_section"]

        // fullTokens rendered WITH additionalContext
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": system],
            ["role": "user", "content": "hello"],
        ]
        let fullTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: ctx
        )

        // Detection WITHOUT additionalContext → probes don't include [ctx:...],
        // verification against fullTokens fails → nil
        let result = try StablePrefixDetector.detect(
            systemPrompt: system, toolSpecs: nil, additionalContext: nil,
            fullTokens: fullTokens, tokenizer: tokenizer
        )
        #expect(result == nil)
    }

    @Test func additionalContextWithToolsExtendsBoundary() throws {
        let system = "Agent."
        let tools: [ToolSpec] = [makeToolSpec(name: "read")]
        let ctx: [String: any Sendable] = ["prefix_text": "session_info"]

        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": system],
            ["role": "user", "content": "go"],
        ]
        let fullWithCtx = try tokenizer.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: ctx
        )
        let fullWithoutCtx = try tokenizer.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: nil
        )

        let detectedWithCtx = try StablePrefixDetector.detect(
            systemPrompt: system, toolSpecs: tools, additionalContext: ctx,
            fullTokens: fullWithCtx, tokenizer: tokenizer
        )
        let detectedWithoutCtx = try StablePrefixDetector.detect(
            systemPrompt: system, toolSpecs: tools, additionalContext: nil,
            fullTokens: fullWithoutCtx, tokenizer: tokenizer
        )
        let boundaryWithCtx = try #require(detectedWithCtx)
        let boundaryWithoutCtx = try #require(detectedWithoutCtx)

        // Context adds tokens → larger boundary
        #expect(boundaryWithCtx > boundaryWithoutCtx)
    }
}
