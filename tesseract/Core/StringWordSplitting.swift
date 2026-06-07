//
//  StringWordSplitting.swift
//  tesseract
//
//  One definition of "a word" for the Speech feature: whitespace- and
//  newline-separated, empty runs omitted. Shared by Word Timeline (the rendered word
//  model) and TextSegmenter (token-count estimation) so the two can't drift on what
//  counts as a word boundary.
//

import Foundation

extension StringProtocol {
    /// Split into whitespace- and newline-separated words, omitting empty runs.
    /// `nonisolated` so the `nonisolated` `WordTimeline` value can call it under the
    /// project's default-MainActor isolation.
    nonisolated func splitIntoWords() -> [SubSequence] {
        split(omittingEmptySubsequences: true, whereSeparator: { $0.isWhitespace || $0.isNewline })
    }
}
