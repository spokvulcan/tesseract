//
//  TextSegmenter.swift
//  tesseract
//

import Foundation
import NaturalLanguage

struct TextSegment: Sendable {
    let index: Int
    let text: String
}

enum TextSegmenter {
    private enum Defaults {
        static let targetTokensPerSegment = 200
        static let tokensPerWordEstimate: Double = 1.3
    }

    static func segment(_ text: String, targetTokens: Int = Defaults.targetTokensPerSegment) -> [TextSegment] {
        let sentences = splitIntoSentences(text)
        guard sentences.count > 1 else {
            return [TextSegment(index: 0, text: text)]
        }

        var segments: [TextSegment] = []
        var currentSentences: [String] = []
        var currentTokenEstimate = 0

        for sentence in sentences {
            let sentenceTokens = estimateTokens(sentence)

            // If adding this sentence would exceed the budget and we have content, finalize segment
            if currentTokenEstimate + sentenceTokens > targetTokens && !currentSentences.isEmpty {
                let segmentText = currentSentences.joined()
                segments.append(TextSegment(index: segments.count, text: segmentText))
                currentSentences = []
                currentTokenEstimate = 0
            }

            currentSentences.append(sentence)
            currentTokenEstimate += sentenceTokens
        }

        // Flush remaining
        if !currentSentences.isEmpty {
            let segmentText = currentSentences.joined()
            segments.append(TextSegment(index: segments.count, text: segmentText))
        }

        return segments
    }

    static func isLongForm(_ text: String) -> Bool {
        let tokens = estimateTokens(text)
        return tokens > Defaults.targetTokensPerSegment
    }

    // MARK: - Private

    private static func estimateTokens(_ text: String) -> Int {
        let words = text.split(whereSeparator: { $0.isWhitespace || $0.isNewline }).count
        return Int(Double(words) * Defaults.tokensPerWordEstimate)
    }

    private static func splitIntoSentences(_ text: String) -> [String] {
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text

        var sentences: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let sentence = String(text[range])
            sentences.append(sentence)
            return true
        }

        // If tokenizer found nothing, return the whole text
        if sentences.isEmpty {
            return [text]
        }

        return sentences
    }
}
