//
//  MemorySpeech.swift
//  tesseract
//
//  What counts as something the owner *said* (ADR-0035 §1).
//
//  An episode is testimony — the immutable layer every belief is ultimately
//  answerable to. So it had better contain his words and not the app's.
//
//  It did not. A skill invocation sends the skill's whole body as the user
//  message (`<skill name="proofread" location="…">…</skill>` followed by
//  whatever he actually typed — `ChatSession.executeSkill`), and memory recorded
//  that verbatim. 28 of the first 207 episodes in the owner's own store were
//  hundreds of lines of skill instructions with a sentence of his at the bottom;
//  eight beliefs had been distilled from them; and the retrieval block quoted
//  them back to the model as "things that were actually said, verbatim" —
//  complete with the container paths they carried.
//
//  So the machine wrapper comes off at the door, and it comes off in exactly one
//  place, used by both the live capture and the backfill: history and future
//  have to agree about what he said.
//

import Foundation

nonisolated enum MemorySpeech: Sendable {

    /// The part of a message the owner actually contributed, or nil if there is
    /// none — a bare skill fire with no arguments is the app talking to itself,
    /// and it is not an episode.
    ///
    /// Wrappers are stripped, never summarised: what is left is verbatim, which
    /// is the whole point of the layer.
    static func spoken(_ text: String) -> String? {
        var remainder = text
        for tag in ["skill", "memory"] {
            remainder = strip(tag: tag, from: remainder)
        }
        let trimmed = remainder.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    /// Remove every `<tag …> … </tag>` block, keeping what surrounds them.
    ///
    /// Scanned by hand rather than by regex because the body is arbitrary text
    /// (markdown, code, quotes) and a lazy regex over a few hundred lines of it
    /// is the kind of thing that quietly eats the message it was meant to
    /// preserve. An unterminated open tag drops the rest of the message — a
    /// truncated wrapper is not testimony either.
    private static func strip(tag: String, from text: String) -> String {
        let open = "<\(tag)"
        let close = "</\(tag)>"
        var out = ""
        var rest = Substring(text)

        while let start = rest.range(of: open) {
            // Only a real tag: `<skill name=…>` or `<skill>`, not `<skills>`.
            let after = rest[start.upperBound...].first
            guard after == " " || after == ">" || after == "\n" else {
                out += rest[..<start.upperBound]
                rest = rest[start.upperBound...]
                continue
            }
            out += rest[..<start.lowerBound]
            guard let end = rest.range(of: close, range: start.upperBound..<rest.endIndex) else {
                return out
            }
            rest = rest[end.upperBound...]
        }
        out += rest
        return out
    }
}
