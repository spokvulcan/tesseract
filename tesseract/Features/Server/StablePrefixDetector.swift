import CryptoKit
import Foundation
import MLXLMCommon

/// Detects the stable-prefix boundary: the token offset where
/// system message(s) + tool definitions end in the tokenized prompt.
///
/// Uses the **two-probe technique**: tokenize two probes with different user
/// messages through the same chat template and tools, then find where they
/// diverge. The common prefix = system + tools boundary.
///
/// Template-agnostic — works for ChatML, Llama, or any format because it
/// compares token sequences token-by-token.
struct StablePrefixDetector {

    /// Memo of past probe results, keyed by the probe inputs (system prompt,
    /// canonical tools, additional context): value = the common prefix length
    /// plus a hash of the common prefix tokens, so the fullTokens
    /// verification below runs identically on hits and misses. A production
    /// server issues the same stable prefix on nearly every request, so the
    /// two Jinja renders + BPE encodes (tens of ms at 10K+ tokens of
    /// system+tools) would otherwise be paid per request. A hit whose stored
    /// prefix hash doesn't match fullTokens falls through to a fresh
    /// two-probe detect — a memo entry from a different template can cost a
    /// cache hit, never produce a wrong boundary.
    nonisolated(unsafe) private static var memo: [String: (commonLength: Int, prefixHash: String)] =
        [:]
    nonisolated private static let memoLock = NSLock()

    /// Test hook: drop all memoized probe results.
    nonisolated static func resetMemo() {
        memoLock.withLock { memo.removeAll() }
    }

    /// Returns the token offset where the stable prefix (system + tools) ends.
    /// Returns nil if:
    /// - systemPrompt is nil or empty
    /// - Probes produce zero common prefix
    /// - Common prefix doesn't match the start of fullTokens
    /// - Common prefix is suspiciously short for a large prompt (suggests
    ///   non-deterministic template rendering, e.g. swift-jinja tojson variance)
    nonisolated static func detect(
        systemPrompt: String?,
        toolSpecs: [ToolSpec]?,
        additionalContext: [String: any Sendable]? = nil,
        fullTokens: [Int],
        tokenizer: any Tokenizer
    ) throws -> Int? {
        guard let systemPrompt, !systemPrompt.isEmpty else {
            return nil
        }

        let memoKey = Self.memoKey(
            systemPrompt: systemPrompt, toolSpecs: toolSpecs, additionalContext: additionalContext)
        if let cached = (memoLock.withLock { memo[memoKey] }),
            fullTokens.count >= cached.commonLength,
            tokenHash(fullTokens[0..<cached.commonLength]) == cached.prefixHash,
            passesRatioGuard(commonLength: cached.commonLength, fullTokens: fullTokens)
        {
            return cached.commonLength
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
            fullTokens[0..<commonLength].elementsEqual(probeA[0..<commonLength]),
            passesRatioGuard(commonLength: commonLength, fullTokens: fullTokens)
        else {
            return nil
        }

        memoLock.withLock {
            // Bounded: probe keys rotate only when the operator changes the
            // system prompt / tool set / template context.
            if memo.count > 256 { memo.removeAll() }
            memo[memoKey] = (
                commonLength, tokenHash(probeA[0..<commonLength])
            )
        }
        return commonLength
    }

    // MARK: - Private

    /// Reject suspiciously-short stable prefixes for large prompts.
    /// swift-jinja's `tojson` filter occasionally produces non-deterministic
    /// key ordering for nested dicts, causing probes to diverge inside the
    /// tools block. On large prompts (>1000 tokens), a legitimate stable
    /// prefix should cover at least a third of the tokens — system prompts
    /// and tool definitions dominate those prompts. Smaller ratios indicate
    /// probe divergence is an artifact, not a real user-content boundary.
    /// Skip this check for small prompts (unit tests, trivial messages)
    /// where the user content itself may be a large fraction.
    private nonisolated static func passesRatioGuard(
        commonLength: Int, fullTokens: [Int]
    ) -> Bool {
        fullTokens.count <= 1000 || commonLength >= fullTokens.count / 3
    }

    private nonisolated static func memoKey(
        systemPrompt: String,
        toolSpecs: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) -> String {
        var parts = [sha256Hex(Data(systemPrompt.utf8))]
        if let toolSpecs,
            JSONSerialization.isValidJSONObject(toolSpecs),
            let data = try? JSONSerialization.data(
                withJSONObject: toolSpecs, options: [.sortedKeys])
        {
            parts.append(sha256Hex(data))
        }
        if let additionalContext,
            JSONSerialization.isValidJSONObject(additionalContext),
            let data = try? JSONSerialization.data(
                withJSONObject: additionalContext, options: [.sortedKeys])
        {
            parts.append(sha256Hex(data))
        }
        return parts.joined(separator: "|")
    }

    /// Hash of a token-ID slice as little-endian Int32 bytes (vocab IDs fit
    /// in 32 bits). Used for the fullTokens prefix verification on memo hits.
    private nonisolated static func tokenHash(_ tokens: ArraySlice<Int>) -> String {
        var data = Data(capacity: tokens.count * 4)
        for token in tokens {
            var v = Int32(token)
            data.append(Data(bytes: &v, count: 4))
        }
        return sha256Hex(data)
    }

    private nonisolated static func tokenHash(_ tokens: [Int]) -> String {
        tokenHash(tokens[...])
    }

    private nonisolated static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private nonisolated static func tokenizeProbe(
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
