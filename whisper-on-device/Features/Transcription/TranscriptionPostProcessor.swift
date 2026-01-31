//
//  TranscriptionPostProcessor.swift
//  whisper-on-device
//

import Foundation

struct TranscriptionPostProcessor: Sendable {
    func process(_ text: String) -> String {
        var result = text

        // Trim whitespace
        result = result.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !result.isEmpty else { return result }

        // Remove duplicate spaces
        result = removeDuplicateSpaces(result)

        // Fix common Whisper artifacts
        result = removeWhisperArtifacts(result)

        // Normalize punctuation spacing
        result = normalizePunctuation(result)

        // Ensure proper capitalization
        result = capitalizeAppropriately(result)

        return result
    }

    // MARK: - Private

    private func removeDuplicateSpaces(_ text: String) -> String {
        text.replacingOccurrences(
            of: "\\s+",
            with: " ",
            options: .regularExpression
        )
    }

    private func removeWhisperArtifacts(_ text: String) -> String {
        var result = text

        // Remove repeated words/phrases (e.g., "the the" -> "the")
        let words = result.components(separatedBy: .whitespaces)
        var cleanedWords: [String] = []

        for (index, word) in words.enumerated() {
            // Skip if this word is the same as the previous one
            if index > 0 && word.lowercased() == words[index - 1].lowercased() {
                continue
            }
            cleanedWords.append(word)
        }

        result = cleanedWords.joined(separator: " ")

        // Remove common Whisper hallucinations
        let hallucinations = [
            "Thank you for watching.",
            "Thanks for watching.",
            "Please subscribe.",
            "Like and subscribe.",
            "[Music]",
            "[Applause]",
            "[Laughter]",
            "(upbeat music)",
            "(gentle music)",
        ]

        for hallucination in hallucinations {
            result = result.replacingOccurrences(of: hallucination, with: "")
        }

        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func normalizePunctuation(_ text: String) -> String {
        var result = text

        // Remove space before punctuation
        result = result.replacingOccurrences(
            of: "\\s+([.,!?;:])",
            with: "$1",
            options: .regularExpression
        )

        // Ensure space after punctuation (except at end)
        result = result.replacingOccurrences(
            of: "([.,!?;:])([A-Za-z])",
            with: "$1 $2",
            options: .regularExpression
        )

        // Fix multiple punctuation marks
        result = result.replacingOccurrences(
            of: "([.!?]){2,}",
            with: "$1",
            options: .regularExpression
        )

        return result
    }

    private func capitalizeAppropriately(_ text: String) -> String {
        var result = text

        // Capitalize first letter
        if let first = result.first {
            result = first.uppercased() + String(result.dropFirst())
        }

        // Capitalize after sentence-ending punctuation
        result = result.replacingOccurrences(
            of: "([.!?])\\s+([a-z])",
            with: "$1 $2",
            options: .regularExpression
        )

        // Use a different approach for capitalization after punctuation
        var chars = Array(result)
        var capitalizeNext = false

        for i in 0..<chars.count {
            if capitalizeNext && chars[i].isLetter {
                chars[i] = Character(chars[i].uppercased())
                capitalizeNext = false
            }

            if chars[i] == "." || chars[i] == "!" || chars[i] == "?" {
                capitalizeNext = true
            }
        }

        result = String(chars)

        // Capitalize standalone "I"
        result = result.replacingOccurrences(
            of: "\\bi\\b",
            with: "I",
            options: .regularExpression
        )

        return result
    }
}
