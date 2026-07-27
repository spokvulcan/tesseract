// tokdiff-replacement.swift — Split-step replacement code for
// swift-transformers 1.3.3 (Sources/Tokenizers).
//
// ***************************************************************************
// * VERDICT: DO NOT SHIP. This is the candidate that was evaluated, kept   *
// * for the record. It produces DIFFERENT final token ids than production  *
// * on two text classes (measured over a 3.0M-token differential corpus,   *
// * see tokdiff-run1.log):                                                 *
// *                                                                        *
// *   D1 CRLF clusters:  "!\r\n\r\n!"  P=[0,317,317,0]  ICU=[0,845,0]      *
// *                      "\r\r\r\n"    P=[201,201,317]  ICU=[23574]        *
// *   D2 VS16 clusters:  "🤖❤️!"  P=[9008,97,244,209726,0]                 *
// *                      ICU=[9008,97,244,158938,29545,0]                  *
// *                                                                        *
// * Root cause: Swift Regex matches character classes at extended          *
// * grapheme-cluster granularity; NSRegularExpression (ICU) matches at     *
// * code-point granularity. No pattern mutation can faithfully re-express  *
// * UAX#29 cluster semantics (CRLF as one unit, base+VS16 as one unit)     *
// * inside an ICU pattern. The quirk-mutated variant B (which folds in the *
// * [^\r\n\p{L}\p{N}] negated-class quirk) diverges IDENTICALLY to A —     *
// * the \n-before-letter class was the ONLY class B needed to fix, and it  *
// * is not enough.                                                         *
// ***************************************************************************
//
// What this file contains, for the record:
//   1. The exact diff-shaped code for arm A ("clean ICU") — the form that
//      WOULD have been upstreamable as "same ids, faster" had it matched.
//   2. The one-line arm-B mutation, for completeness.
//
// Split-phase timing on the 32K bench prompt (ABBA, 7 reps, medians):
//   production loop: 76.17 ms   single-pass ICU: 6.24 ms   = 12.2x faster
// (per-corpus-class ratios 4.4x–13.3x; no pathological item found where ICU
//  was slower than the loop — see SPLIT-PHASE TIME table in tokdiff-run1.log)

import Foundation

// ============================================================================
// 1. String+PreTokenization.swift — add one method to the String extension.
//    Reproduces split(by:options:includeSeparators:true, omittingEmpty-
//    Subsequences:true) piece-for-piece when the engines agree, from ONE
//    precompiled NSRegularExpression pass instead of one
//    String.range(of:.regularExpression) call per match.
// ============================================================================

extension String {
    /// Isolated-behavior split driven by a single precompiled-regex pass.
    /// Emits gap substrings and match substrings in order — the same pieces
    /// as `split(by: pattern, includeSeparators: true)` whenever the regex
    /// engines agree on match boundaries.
    func split(byMatchesOf regex: NSRegularExpression) -> [String] {
        let selfRange = NSRange(startIndex..<endIndex, in: self)
        let matches = regex.matches(in: self, options: [], range: selfRange)
        var result: [String] = []
        var start = startIndex
        for match in matches {
            // Same conversion semantics as split(by captureRegex:).
            guard let matchRange = Range(match.range, in: self) else { continue }
            if start < matchRange.lowerBound {
                result.append(String(self[start..<matchRange.lowerBound]))
            }
            result.append(String(self[matchRange]))
            start = matchRange.upperBound
        }
        if start < endIndex {
            result.append(String(self[start...]))
        }
        return result
    }
}

// ============================================================================
// 2. StringSplitPattern (String+PreTokenization.swift) — carry a compiled
//    regex alongside the pattern string; use it for the .regexp case.
// ============================================================================

enum StringSplitPattern {
    case regexp(regexp: String)
    case string(pattern: String)

    func split(_ text: String, invert: Bool = true) -> [String] {
        switch self {
        case let .regexp(regexp):
            // Arm A: single-pass ICU. Identical pieces to the old loop
            // whenever Swift Regex and ICU agree on boundaries.
            if let regex = StringSplitPatternCache.shared.regex(for: regexp) {
                return text.split(byMatchesOf: regex)
            }
            return text.split(by: regexp, includeSeparators: true)
        case let .string(substring):
            return text.split(by: substring, options: [], includeSeparators: !invert)
        }
    }

    static func from(config: Config) -> StringSplitPattern? {
        if let pattern = config.pattern.String.string() {
            return .string(pattern: pattern)
        }
        if let pattern = config.pattern.Regex.string() {
            return .regexp(regexp: pattern)
        }
        return nil
    }
}

/// Compile-once cache so the NSRegularExpression is built once per pattern
/// (SplitPreTokenizer instances are rebuilt per tokenizer init, and the
/// pattern string is a compile-time constant of the tokenizer.json).
final class StringSplitPatternCache: @unchecked Sendable {
    static let shared = StringSplitPatternCache()
    private var cache: [String: NSRegularExpression] = [:]
    private let lock = NSLock()

    func regex(for pattern: String) -> NSRegularExpression? {
        lock.lock()
        defer { lock.unlock() }
        if let cached = cache[pattern] { return cached }
        guard let compiled = try? NSRegularExpression(pattern: pattern, options: []) else {
            return nil
        }
        cache[pattern] = compiled
        return compiled
    }
}

// ============================================================================
// 3. Arm B variant (quirk-preserving) — same code, but the pattern fed to
//    NSRegularExpression is first rewritten to fold in the Swift-Regex
//    negated-class quirk: in a negated class whose source contains the
//    adjacent pair "\r\n", Swift Regex lets \r and \n individually escape
//    the negation (proven: tokquirk.swift battery — [^\n\p{L}] and
//    [^\r\p{L}] behave correctly; [^\r\n...] does not; \s-based negated
//    classes are unaffected). The exact fold for the Qwen pattern:
//
//        "[^\\r\\n\\p{L}\\p{N}]"  ->  "[^\\p{L}\\p{N}]"
//
//    Applied once, at cache-fill time:
//
//        let icuPattern = pattern.replacingOccurrences(
//            of: #"[^\r\n\p{L}\p{N}]"#, with: #"[^\p{L}\p{N}]"#)
//
//    Measured result: B reproduces Swift's split pieces for the
//    \n/\r-before-letter class piece-for-piece, but diverges IDENTICALLY
//    to A on the CRLF-cluster and VS16-cluster classes — so it is NOT a
//    shippable form either, just a muddier one.
// ============================================================================
