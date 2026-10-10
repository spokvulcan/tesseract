//
//  LinkedTitle.swift
//  tesseract
//
//  A title as Today shows it. A web link caught into a reminder ("Research
//  https://github.com/…") reads short, its host and path without "www.", the
//  query or a closing slash, cut at forty characters, and opens with a
//  click. The words around it stay as written. Pure: a title in, runs out;
//  `Text(linkedTitle:)` draws them, the links in the page's accent.
//

import SwiftUI

nonisolated enum LinkedTitle {

    /// The longest a link reads; past it, it is cut with "…".
    static let maxLinkLength = 40

    enum Run: Sendable, Equatable {
        case text(String)
        /// How the link reads, and where it goes.
        case link(String, URL)
    }

    /// What ends a sentence after a link, not the link itself.
    private static let closing: Set<Character> = [".", ",", ";", ":", "!", "?", "'", "\""]

    static func runs(_ title: String) -> [Run] {
        var runs: [Run] = []
        var rest = title[...]
        while let match = rest.firstMatch(of: /(?i)https?:\/\/[^\s<>"]+/) {
            let start = match.range.lowerBound
            // Punctuation closing the sentence is the sentence's, and so is a
            // closing bracket the link never opened.
            var end = match.range.upperBound
            while end > start {
                let last = rest[rest.index(before: end)]
                let link = rest[start..<end]
                let unopened =
                    last == ")"
                    && link.count(where: { $0 == ")" }) > link.count(where: { $0 == "(" })
                guard closing.contains(last) || unopened else { break }
                end = rest.index(before: end)
            }
            let raw = String(rest[start..<end])
            guard let url = URL(string: raw), let shown = shortForm(url) else {
                runs.append(.text(String(rest[..<end])))
                rest = rest[end...]
                continue
            }
            if start > rest.startIndex {
                runs.append(.text(String(rest[..<start])))
            }
            runs.append(.link(shown, url))
            rest = rest[end...]
        }
        if !rest.isEmpty { runs.append(.text(String(rest))) }
        // Neighbouring words from a link that didn't parse read as one run.
        return runs.reduce(into: []) { merged, run in
            if case .text(let next) = run, case .text(let previous)? = merged.last {
                merged[merged.count - 1] = .text(previous + next)
            } else {
                merged.append(run)
            }
        }
    }

    /// The title with its links short and clickable, for a `Text`.
    static func attributed(_ title: String) -> AttributedString {
        runs(title).reduce(into: AttributedString()) { text, run in
            switch run {
            case .text(let words):
                text += AttributedString(words)
            case .link(let shown, let url):
                var link = AttributedString(shown)
                link.link = url
                text += link
            }
        }
    }

    /// "github.com/lithos-ai/lithos-metal": the host without "www." and the
    /// path without a closing slash, cut at `maxLinkLength`.
    private static func shortForm(_ url: URL) -> String? {
        guard var host = url.host(percentEncoded: false)?.lowercased(), !host.isEmpty else {
            return nil
        }
        if host.hasPrefix("www.") { host.removeFirst(4) }
        var path = url.path(percentEncoded: false)
        while path.hasSuffix("/") { path.removeLast() }
        let shown = host + path
        guard shown.count > maxLinkLength else { return shown }
        return shown.prefix(maxLinkLength - 1) + "…"
    }
}

extension Text {
    /// A title with its links short and one click from opening, drawn in
    /// `linkColor` (the accent, as Today's other clickable words; a done
    /// task's fade with it) rather than the system's link blue.
    init(linkedTitle title: String, linkColor: Color = .accentColor) {
        var text = LinkedTitle.attributed(title)
        let links = text.runs.filter { $0.link != nil }.map(\.range)
        for range in links {
            text[range].foregroundColor = linkColor
        }
        self.init(text)
    }
}
