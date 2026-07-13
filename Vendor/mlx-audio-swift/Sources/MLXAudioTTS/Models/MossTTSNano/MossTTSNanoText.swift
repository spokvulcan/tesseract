import Foundation
import MLXAudioCore

public protocol MossTextTokenizing {
    func encode(_ text: String) -> [Int]
    func decode(_ tokenIDs: [Int]) -> String
}

public final class MossSentencePieceTokenizer: MossTextTokenizing {
    private let tokenizer: SentencePieceTokenizer

    public init(modelURL: URL) throws {
        self.tokenizer = try SentencePieceTokenizer.from(sentencePieceModelURL: modelURL)
    }

    public func encode(_ text: String) -> [Int] {
        let normalized = Self.normalizeSentencePieceWhitespace(text)
        guard !normalized.isEmpty else { return [] }
        return tokenizer.encodeWithByteFallback(normalized)
    }

    public func decode(_ tokenIDs: [Int]) -> String {
        tokenizer.decode(tokenIDs)
    }

    private static func normalizeSentencePieceWhitespace(_ text: String) -> String {
        var normalized = ""
        var previousWasWhitespace = true
        for scalar in text.unicodeScalars {
            if CharacterSet.whitespacesAndNewlines.contains(scalar) {
                if !previousWasWhitespace {
                    normalized.append(" ")
                    previousWasWhitespace = true
                }
            } else {
                normalized.unicodeScalars.append(scalar)
                previousWasWhitespace = false
            }
        }
        if normalized.last == " " {
            normalized.removeLast()
        }
        return normalized
    }
}

public let mossUserRolePrefix = "user\n"
public let mossUserTemplateReferencePrefix = "<user_inst>\n- Reference(s):\n"
public let mossUserTemplateAfterReference = """

- Instruction:
None
- Tokens:
None
- Quality:
None
- Sound Event:
None
- Ambient Sound:
None
- Language:
None
- Text:
"""
public let mossUserTemplateSuffix = "\n</user_inst>"
public let mossAssistantTurnPrefix = "\n"
public let mossAssistantRolePrefix = "assistant\n"

private let mossSentenceEndPunctuation = Set(".!?。！？；;")
private let mossClauseSplitPunctuation = Set(",，、；;：:")
private let mossClosingPunctuation = Set("\"'\"')]}）】》」』")

public func mossLoadTokenizer(modelDirectory: URL) throws -> MossSentencePieceTokenizer {
    try MossSentencePieceTokenizer(modelURL: modelDirectory.appendingPathComponent("tokenizer.model"))
}

public func mossEncodeText(_ tokenizer: MossTextTokenizing, _ text: String) -> [Int] {
    tokenizer.encode(text)
}

public func mossBuildUserPromptPrefix(
    tokenizer: MossTextTokenizing,
    config: MossTTSNanoConfig
) -> [Int] {
    [config.imStartTokenID]
        + mossEncodeText(tokenizer, mossUserRolePrefix)
        + mossEncodeText(tokenizer, mossUserTemplateReferencePrefix)
}

public func mossBuildUserPromptAfterReference(tokenizer: MossTextTokenizing) -> [Int] {
    mossEncodeText(tokenizer, mossUserTemplateAfterReference)
}

public func mossBuildAssistantPromptPrefix(
    tokenizer: MossTextTokenizing,
    config: MossTTSNanoConfig
) -> [Int] {
    mossEncodeText(tokenizer, mossUserTemplateSuffix)
        + [config.imEndTokenID]
        + mossEncodeText(tokenizer, mossAssistantTurnPrefix)
        + [config.imStartTokenID]
        + mossEncodeText(tokenizer, mossAssistantRolePrefix)
}

public func mossBuildPromptTokenIDs(
    tokenizer: MossTextTokenizing,
    config: MossTTSNanoConfig,
    textTokenIDs: [Int]
) -> [Int] {
    mossBuildUserPromptPrefix(tokenizer: tokenizer, config: config)
        + mossEncodeText(tokenizer, "None")
        + mossBuildUserPromptAfterReference(tokenizer: tokenizer)
        + textTokenIDs
        + mossBuildAssistantPromptPrefix(tokenizer: tokenizer, config: config)
}

public func mossContainsCJK(_ text: String) -> Bool {
    text.unicodeScalars.contains { scalar in
        let value = scalar.value
        return (0x4E00...0x9FFF).contains(value)
            || (0x3400...0x4DBF).contains(value)
            || (0x3040...0x30FF).contains(value)
            || (0xAC00...0xD7AF).contains(value)
    }
}

public func mossPrepareTextForSentenceChunking(_ text: String) throws -> String {
    var normalized = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !normalized.isEmpty else {
        throw MossTTSNanoError.invalidInput("Text prompt cannot be empty.")
    }

    normalized = normalized
        .replacingOccurrences(of: "\r", with: " ")
        .replacingOccurrences(of: "\n", with: " ")
    while normalized.contains("  ") {
        normalized = normalized.replacingOccurrences(of: "  ", with: " ")
    }

    if mossContainsCJK(normalized) {
        if let last = normalized.last, !mossSentenceEndPunctuation.contains(last) {
            normalized.append("。")
        }
        return normalized
    }

    if let first = normalized.first,
       String(first).lowercased() == String(first),
       String(first).uppercased() != String(first) {
        normalized = String(first).uppercased() + normalized.dropFirst()
    }
    if let last = normalized.last, last.isLetter || last.isNumber {
        normalized.append(".")
    }
    if normalized.split(whereSeparator: \.isWhitespace).count < 5 {
        normalized = "        " + normalized
    }
    return normalized
}

public func mossSplitTextByPunctuation(_ text: String, punctuation: Set<Character>) -> [String] {
    var sentences: [String] = []
    var current: [Character] = []
    let chars = Array(text)
    var index = 0

    while index < chars.count {
        let character = chars[index]
        current.append(character)
        if punctuation.contains(character) {
            var lookahead = index + 1
            while lookahead < chars.count, mossClosingPunctuation.contains(chars[lookahead]) {
                current.append(chars[lookahead])
                lookahead += 1
            }
            let sentence = String(current).trimmingCharacters(in: .whitespacesAndNewlines)
            if !sentence.isEmpty {
                sentences.append(sentence)
            }
            current.removeAll(keepingCapacity: true)
            while lookahead < chars.count, chars[lookahead].isWhitespace {
                lookahead += 1
            }
            index = lookahead
            continue
        }
        index += 1
    }

    let tail = String(current).trimmingCharacters(in: .whitespacesAndNewlines)
    if !tail.isEmpty {
        sentences.append(tail)
    }
    return sentences
}

public func mossJoinSentenceParts(_ left: String, _ right: String) -> String {
    if left.isEmpty { return right }
    if right.isEmpty { return left }
    if mossContainsCJK(left) || mossContainsCJK(right) {
        return left + right
    }
    return "\(left) \(right)"
}

public func mossSplitTextByTokenBudget(
    tokenizer: MossTextTokenizing,
    text: String,
    maxTokens: Int
) -> [String] {
    var remaining = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !remaining.isEmpty else { return [] }

    let safeMaxTokens = max(1, maxTokens)
    let preferredBoundaryChars = mossClauseSplitPunctuation
        .union(mossSentenceEndPunctuation)
        .union([" "])
    var pieces: [String] = []

    while !remaining.isEmpty {
        if mossEncodeText(tokenizer, remaining).count <= safeMaxTokens {
            pieces.append(remaining)
            break
        }

        let chars = Array(remaining)
        var low = 1
        var high = chars.count
        var bestPrefixLength = 1
        while low <= high {
            let middle = (low + high) / 2
            let candidate = String(chars.prefix(middle)).trimmingCharacters(in: .whitespacesAndNewlines)
            if candidate.isEmpty {
                low = middle + 1
                continue
            }
            if mossEncodeText(tokenizer, candidate).count <= safeMaxTokens {
                bestPrefixLength = middle
                low = middle + 1
            } else {
                high = middle - 1
            }
        }

        var cutIndex = bestPrefixLength
        let prefixChars = Array(chars.prefix(bestPrefixLength))
        let scanStart = prefixChars.count - 1
        let scanEnd = max(-1, prefixChars.count - 25)
        if scanStart > scanEnd {
            for scanIndex in stride(from: scanStart, to: scanEnd, by: -1) {
                if preferredBoundaryChars.contains(prefixChars[scanIndex]) {
                    cutIndex = scanIndex + 1
                    break
                }
            }
        }

        var piece = String(chars.prefix(cutIndex)).trimmingCharacters(in: .whitespacesAndNewlines)
        if piece.isEmpty {
            piece = String(chars.prefix(bestPrefixLength)).trimmingCharacters(in: .whitespacesAndNewlines)
            cutIndex = bestPrefixLength
        }
        pieces.append(piece)
        remaining = String(chars.dropFirst(cutIndex)).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    return pieces
}

public func mossSplitTextIntoBestSentences(
    tokenizer: MossTextTokenizing,
    text: String,
    maxTokens: Int = 75
) throws -> [String] {
    let normalized = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !normalized.isEmpty else { return [] }

    let safeMaxTokens = max(1, maxTokens)
    let prepared = try mossPrepareTextForSentenceChunking(normalized)
    let sentenceCandidates = mossSplitTextByPunctuation(prepared, punctuation: mossSentenceEndPunctuation)
    let candidates = sentenceCandidates.isEmpty ? [prepared] : sentenceCandidates

    var sentenceSlices: [(tokenCount: Int, text: String)] = []
    for sentence in candidates {
        let normalizedSentence = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalizedSentence.isEmpty else { continue }
        let sentenceTokenCount = mossEncodeText(tokenizer, normalizedSentence).count
        if sentenceTokenCount <= safeMaxTokens {
            sentenceSlices.append((sentenceTokenCount, normalizedSentence))
            continue
        }

        let clauseCandidates = mossSplitTextByPunctuation(normalizedSentence, punctuation: mossClauseSplitPunctuation)
        let clauses = clauseCandidates.count <= 1 ? [normalizedSentence] : clauseCandidates
        for clause in clauses {
            let normalizedClause = clause.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !normalizedClause.isEmpty else { continue }
            let clauseTokenCount = mossEncodeText(tokenizer, normalizedClause).count
            if clauseTokenCount <= safeMaxTokens {
                sentenceSlices.append((clauseTokenCount, normalizedClause))
                continue
            }
            for piece in mossSplitTextByTokenBudget(
                tokenizer: tokenizer,
                text: normalizedClause,
                maxTokens: safeMaxTokens
            ) {
                let normalizedPiece = piece.trimmingCharacters(in: .whitespacesAndNewlines)
                if !normalizedPiece.isEmpty {
                    sentenceSlices.append((mossEncodeText(tokenizer, normalizedPiece).count, normalizedPiece))
                }
            }
        }
    }

    var chunks: [String] = []
    var currentChunk = ""
    var currentTokenCount = 0
    for slice in sentenceSlices {
        if currentChunk.isEmpty {
            currentChunk = slice.text
            currentTokenCount = slice.tokenCount
            continue
        }
        if currentTokenCount + slice.tokenCount > safeMaxTokens {
            chunks.append(currentChunk.trimmingCharacters(in: .whitespacesAndNewlines))
            currentChunk = slice.text
            currentTokenCount = slice.tokenCount
        } else {
            currentChunk = mossJoinSentenceParts(currentChunk, slice.text)
            currentTokenCount = mossEncodeText(tokenizer, currentChunk).count
        }
    }
    if !currentChunk.isEmpty {
        chunks.append(currentChunk.trimmingCharacters(in: .whitespacesAndNewlines))
    }
    return chunks.count > 1 ? chunks : [normalized]
}

public func mossLightweightNormalizeText(_ text: String) -> String {
    let collapsed = text
        .replacingOccurrences(of: "\r", with: " ")
        .replacingOccurrences(of: "\n", with: " ")
        .trimmingCharacters(in: .whitespacesAndNewlines)
    return collapsed.replacingOccurrences(
        of: #"\s+"#,
        with: " ",
        options: .regularExpression
    )
}
