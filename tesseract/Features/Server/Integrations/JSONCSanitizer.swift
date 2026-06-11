//
//  JSONCSanitizer.swift
//  tesseract
//

import Foundation

/// Reduces JSONC to strict JSON: strips `//` and `/* */` comments and
/// trailing commas, leaving string contents untouched. OpenCode parses every
/// config file as JSONC regardless of extension, so the Config Merge must
/// accept the same input or it would misclassify a legal config as corrupt.
///
/// Comments are not preserved through a merge — the output is re-serialized
/// JSON; the setup script's backup keeps the original.
nonisolated enum JSONCSanitizer {

    static func sanitize(_ text: String) -> String {
        stripTrailingCommas(stripComments(text))
    }

    // MARK: - Private

    private static func stripComments(_ text: String) -> String {
        let scalars = Array(text.unicodeScalars)
        var result = String.UnicodeScalarView()
        result.reserveCapacity(scalars.count)
        var i = 0
        var inString = false
        while i < scalars.count {
            let c = scalars[i]
            if inString {
                result.append(c)
                if c == "\\", i + 1 < scalars.count {
                    result.append(scalars[i + 1])
                    i += 2
                    continue
                }
                if c == "\"" { inString = false }
                i += 1
                continue
            }
            if c == "\"" {
                inString = true
                result.append(c)
                i += 1
                continue
            }
            if c == "/", i + 1 < scalars.count, scalars[i + 1] == "/" {
                while i < scalars.count, scalars[i] != "\n" { i += 1 }
                continue
            }
            if c == "/", i + 1 < scalars.count, scalars[i + 1] == "*" {
                i += 2
                while i + 1 < scalars.count, !(scalars[i] == "*" && scalars[i + 1] == "/") {
                    i += 1
                }
                i = min(i + 2, scalars.count)
                continue
            }
            result.append(c)
            i += 1
        }
        return String(result)
    }

    private static func stripTrailingCommas(_ text: String) -> String {
        let scalars = Array(text.unicodeScalars)
        var result = String.UnicodeScalarView()
        result.reserveCapacity(scalars.count)
        var i = 0
        var inString = false
        while i < scalars.count {
            let c = scalars[i]
            if inString {
                result.append(c)
                if c == "\\", i + 1 < scalars.count {
                    result.append(scalars[i + 1])
                    i += 2
                    continue
                }
                if c == "\"" { inString = false }
                i += 1
                continue
            }
            if c == "\"" {
                inString = true
                result.append(c)
                i += 1
                continue
            }
            if c == "," {
                var j = i + 1
                while j < scalars.count,
                    CharacterSet.whitespacesAndNewlines.contains(scalars[j])
                { j += 1 }
                if j < scalars.count, scalars[j] == "}" || scalars[j] == "]" {
                    i += 1
                    continue
                }
            }
            result.append(c)
            i += 1
        }
        return String(result)
    }
}
