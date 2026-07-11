//
//  ProofreadVerdict.swift
//  tesseract
//
//  The pure value layer of the **Proofread Pass** (map #283, ADR-0034):
//  the verdict vocabulary, the word-level diff variants narrate from, the
//  acceptance guard that keeps a small model from replacing the user's
//  words, and the model-reply parser. All `nonisolated` pure values — no
//  model, no actor, fully unit-testable.
//

import Foundation

/// One word-level change the proofreader made, for overlay narration.
/// `original` empty = an insertion; `replacement` empty = a deletion.
nonisolated struct WordEdit: Equatable, Sendable {
    let original: String
    let replacement: String
}

/// The Proofread Pass's structured outcome for one transcription.
nonisolated enum ProofreadVerdict: Equatable, Sendable {
    /// The model improved the text; `edits` is the word-swap diff
    /// (raw → corrected) variants narrate.
    case corrected(text: String, edits: [WordEdit])
    /// The model found nothing to fix (or its output failed the acceptance
    /// guard — fail-open keeps the user's words).
    case unchanged
    /// A wrong-words take: the model judged the transcription garbage.
    /// Nothing is committed; the raw text stays available for
    /// "insert raw anyway".
    case rejected(reason: String)
}

/// Word-level diff between the raw and the corrected transcription — a
/// longest-common-subsequence over whitespace-split tokens, folded into
/// paired `WordEdit`s (adjacent delete+insert runs pair up as swaps).
nonisolated enum WordDiff {

    static func edits(from raw: String, to corrected: String) -> [WordEdit] {
        let old = raw.split(separator: " ").map(String.init)
        let new = corrected.split(separator: " ").map(String.init)
        guard old != new else { return [] }

        // LCS table (transcriptions are short — a dictation is rarely more
        // than a few hundred words, so the quadratic table is fine).
        var lcs = [[Int]](
            repeating: [Int](repeating: 0, count: new.count + 1), count: old.count + 1)
        for i in stride(from: old.count - 1, through: 0, by: -1) {
            for j in stride(from: new.count - 1, through: 0, by: -1) {
                lcs[i][j] =
                    old[i] == new[j]
                    ? lcs[i + 1][j + 1] + 1
                    : max(lcs[i + 1][j], lcs[i][j + 1])
            }
        }

        // Walk the table, buffering delete/insert runs; flush pairs them
        // positionally into swaps, with leftovers as pure deletes/inserts.
        var edits: [WordEdit] = []
        var deleted: [String] = []
        var inserted: [String] = []
        func flush() {
            let paired = min(deleted.count, inserted.count)
            for k in 0..<paired {
                edits.append(WordEdit(original: deleted[k], replacement: inserted[k]))
            }
            for k in paired..<deleted.count {
                edits.append(WordEdit(original: deleted[k], replacement: ""))
            }
            for k in paired..<inserted.count {
                edits.append(WordEdit(original: "", replacement: inserted[k]))
            }
            deleted = []
            inserted = []
        }

        var i = 0
        var j = 0
        while i < old.count, j < new.count {
            if old[i] == new[j] {
                flush()
                i += 1
                j += 1
            } else if lcs[i + 1][j] >= lcs[i][j + 1] {
                deleted.append(old[i])
                i += 1
            } else {
                inserted.append(new[j])
                j += 1
            }
        }
        deleted.append(contentsOf: old[i...])
        inserted.append(contentsOf: new[j...])
        flush()
        return edits
    }
}

/// The acceptance guard: a 0.8B model must never *replace* the user's words
/// wholesale. A correction is acceptable only when it stays close to the raw
/// text — otherwise the pass fails open to the raw transcription.
nonisolated enum ProofreadGuard {
    /// Corrections may touch at most this share of the raw words.
    static let maxEditedWordShare = 0.5
    /// Corrected length must stay within this factor of the raw length.
    static let maxLengthRatio = 1.8

    static func acceptable(raw: String, corrected: String) -> Bool {
        guard !corrected.isEmpty else { return false }
        let rawCount = max(1, raw.split(separator: " ").count)
        let correctedCount = corrected.split(separator: " ").count
        let ratio = Double(correctedCount) / Double(rawCount)
        guard ratio <= maxLengthRatio, ratio >= 1 / maxLengthRatio else { return false }
        let touched = WordDiff.edits(from: raw, to: corrected).count
        return Double(touched) / Double(rawCount) <= maxEditedWordShare
    }
}

/// Parses the model's reply into a verdict. The prompt contract is plain
/// text out, with a single `REJECT:` escape hatch — no JSON, because a
/// no-think 0.8B holds a text contract far more reliably.
nonisolated enum ProofreadReply {
    static let rejectPrefix = "REJECT:"

    static func parse(_ reply: String, raw: String) -> ProofreadVerdict {
        let trimmed = reply.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix(rejectPrefix) {
            let reason = trimmed.dropFirst(rejectPrefix.count)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return .rejected(reason: reason.isEmpty ? "Unintelligible transcription" : reason)
        }
        guard trimmed != raw, !trimmed.isEmpty else { return .unchanged }
        guard ProofreadGuard.acceptable(raw: raw, corrected: trimmed) else {
            // The model wandered — keep the user's words (fail-open).
            return .unchanged
        }
        return .corrected(text: trimmed, edits: WordDiff.edits(from: raw, to: trimmed))
    }
}
