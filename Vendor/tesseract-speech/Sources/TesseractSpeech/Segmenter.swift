// TesseractSpeech — sentence segmentation, absorbed from the app's v1
// TextSegmenter (ADR-0038: segmentation moves behind the seam). Pure.

import Foundation
import NaturalLanguage

struct TextSegment: Sendable, Equatable {
    let index: Int
    let text: String
}

enum Segmenter {
    private enum Defaults {
        static let targetTokensPerSegment = 200
        static let tokensPerWordEstimate: Double = 1.3
    }

    static func segment(_ text: String, targetTokens: Int = Defaults.targetTokensPerSegment)
        -> [TextSegment]
    {
        let sentences = splitIntoSentences(text)
        guard sentences.count > 1 else {
            return [TextSegment(index: 0, text: text)]
        }

        var segments: [TextSegment] = []
        var currentSentences: [String] = []
        var currentTokenEstimate = 0

        for sentence in sentences {
            let sentenceTokens = estimateTokens(sentence)

            if currentTokenEstimate + sentenceTokens > targetTokens && !currentSentences.isEmpty {
                segments.append(TextSegment(index: segments.count, text: currentSentences.joined()))
                currentSentences = []
                currentTokenEstimate = 0
            }

            currentSentences.append(sentence)
            currentTokenEstimate += sentenceTokens
        }

        if !currentSentences.isEmpty {
            segments.append(TextSegment(index: segments.count, text: currentSentences.joined()))
        }

        return segments
    }

    private static func estimateTokens(_ text: String) -> Int {
        let words = text.split { $0.isWhitespace || $0.isNewline }.count
        return Int(Double(words) * Defaults.tokensPerWordEstimate)
    }

    private static func splitIntoSentences(_ text: String) -> [String] {
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text

        var sentences: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            sentences.append(String(text[range]))
            return true
        }

        return sentences.isEmpty ? [text] : sentences
    }
}
