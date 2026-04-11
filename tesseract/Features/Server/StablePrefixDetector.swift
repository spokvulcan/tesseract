import Foundation
import MLXLMCommon

/// Detects the stable-prefix boundary: the token offset where
/// system message(s) + tool definitions end in the tokenized prompt.
///
/// Uses the **two-probe technique**: tokenize two probes with different user
/// messages through the same chat template and tools, then find where they
/// diverge. The common prefix = system + tools boundary.
///
/// This is robust for ANY template format (ChatML, Llama, custom) because
/// it compares token-by-token between two known-different inputs rather than
/// searching for sentinel tokens or template-specific markers.
struct StablePrefixDetector {

    /// Returns the token offset where the stable prefix (system + tools) ends.
    /// Returns nil if:
    /// - systemPrompt is nil or empty
    /// - Probes produce zero common prefix
    /// - Common prefix doesn't match the start of fullTokens
    static func detect(
        systemPrompt: String?,
        toolSpecs: [ToolSpec]?,
        additionalContext: [String: any Sendable]? = nil,
        fullTokens: [Int],
        tokenizer: any Tokenizer
    ) throws -> Int? {
        guard let systemPrompt, !systemPrompt.isEmpty else {
            return nil
        }

        // Tokenize two probes with different user content but identical
        // system prompt and tools. The common prefix is the stable boundary.
        // Probe strings must differ from the FIRST character so the common
        // prefix stops exactly at the start of user content, not inside it.
        let probeA = try tokenizeProbe(
            systemPrompt: systemPrompt,
            userContent: "A_prefix_probe",
            toolSpecs: toolSpecs,
            additionalContext: additionalContext,
            tokenizer: tokenizer
        )
        let probeB = try tokenizeProbe(
            systemPrompt: systemPrompt,
            userContent: "Z_prefix_probe",
            toolSpecs: toolSpecs,
            additionalContext: additionalContext,
            tokenizer: tokenizer
        )

        // Common prefix length = where system + tools end and user content begins
        let commonLength = zip(probeA, probeB).prefix(while: ==).count

        guard commonLength > 0 else {
            return nil
        }

        // Verify: the common prefix must match the start of fullTokens.
        // If the template interleaves tools after user content or renders
        // differently for different message counts, this catches mismatches.
        guard fullTokens.count >= commonLength,
              fullTokens[0..<commonLength].elementsEqual(probeA[0..<commonLength])
        else {
            return nil
        }

        return commonLength
    }

    // MARK: - Private

    private static func tokenizeProbe(
        systemPrompt: String,
        userContent: String,
        toolSpecs: [ToolSpec]?,
        additionalContext: [String: any Sendable]?,
        tokenizer: any Tokenizer
    ) throws -> [Int] {
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": systemPrompt],
            ["role": "user", "content": userContent],
        ]
        return try tokenizer.applyChatTemplate(
            messages: messages, tools: toolSpecs, additionalContext: additionalContext
        )
    }
}
