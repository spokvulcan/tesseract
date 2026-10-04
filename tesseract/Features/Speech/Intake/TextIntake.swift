//
//  TextIntake.swift
//  tesseract
//
//  Getting text in (#515): a shared web page, a PDF, a Markdown or plain text
//  file, each turned into the text the Reader shows and the voice reads.
//  Both are the same text: the Reader finds each spoken word by counting
//  words forward (ADR-0076), so whatever isn't read (Markdown's marks, a
//  page's menus) isn't shown either. Pure; the share extension and the app
//  both use it.
//

import Foundation
import PDFKit
import SwiftReadability
import SwiftSoup
import UniformTypeIdentifiers

/// A text on its way into the Library.
nonisolated struct IncomingText: Codable, Equatable, Sendable {
    /// Its own title, when it has one (a page's, a PDF's).
    var title: String?
    var text: String
}

nonisolated enum TextIntake {
    enum Failure: Error, Equatable {
        /// A file this app can't read: not text, Markdown or a PDF.
        case unsupported(String)
        /// It was readable, but held no words: a scanned PDF, an empty page.
        case noText
    }

    static let markdown = UTType(
        importedAs: "net.daringfireball.markdown", conformingTo: .plainText)

    /// The kinds of file the app opens.
    static let fileTypes: [UTType] = [.plainText, markdown, .pdf]

    // MARK: - Files

    /// The text of a `.txt`, `.md` or `.pdf` file.
    static func text(fromFile url: URL) throws -> IncomingText {
        let type =
            (try? url.resourceValues(forKeys: [.contentTypeKey]).contentType)
            ?? UTType(filenameExtension: url.pathExtension) ?? .data
        let name = url.deletingPathExtension().lastPathComponent
        let incoming: IncomingText
        if type.conforms(to: .pdf) {
            guard let document = PDFDocument(url: url) else { throw Failure.unsupported(name) }
            incoming = text(fromPDF: document, fallbackTitle: name)
        } else if type.conforms(to: markdown)
            || ["md", "markdown"].contains(url.pathExtension.lowercased())
        {
            incoming = IncomingText(title: nil, text: plainText(fromMarkdown: try decoded(url)))
        } else if type.conforms(to: .text) {
            incoming = IncomingText(title: nil, text: normalized(try decoded(url)))
        } else {
            throw Failure.unsupported(name)
        }
        guard hasWords(incoming.text) else { throw Failure.noText }
        return incoming
    }

    /// A text file's characters: UTF-8, or whatever encoding the file says.
    private static func decoded(_ url: URL) throws -> String {
        if let text = try? String(contentsOf: url, encoding: .utf8) { return text }
        var encoding = String.Encoding.utf8
        return try String(contentsOf: url, usedEncoding: &encoding)
    }

    // MARK: - PDF

    /// A PDF's text, page after page, its lines joined back into paragraphs.
    static func text(fromPDF document: PDFDocument, fallbackTitle: String? = nil) -> IncomingText {
        var pages: [String] = []
        for index in 0..<document.pageCount {
            if let text = document.page(at: index)?.string, hasWords(text) { pages.append(text) }
        }
        let title = (document.documentAttributes?[PDFDocumentAttribute.titleAttribute] as? String)
            .flatMap { $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : $0 }
        return IncomingText(
            title: title ?? fallbackTitle, text: reflowed(pages.joined(separator: "\n\n")))
    }

    /// Lines a PDF broke for the page, joined into paragraphs: a line runs on
    /// unless it is blank, or ends a sentence well short of a full line. A
    /// word hyphenated across lines is joined back.
    static func reflowed(_ text: String) -> String {
        let lines = text.components(separatedBy: .newlines).map {
            $0.trimmingCharacters(in: .whitespaces)
        }
        let lengths = lines.map(\.count).filter { $0 > 0 }.sorted()
        let typical = lengths.isEmpty ? 0 : lengths[lengths.count * 3 / 4]
        var paragraphs: [String] = []
        var current = ""
        for (index, line) in lines.enumerated() {
            guard !line.isEmpty else {
                if !current.isEmpty { paragraphs.append(current) }
                current = ""
                continue
            }
            if current.hasSuffix("-"), let first = line.first, first.isLowercase {
                current.removeLast()
                current += line
            } else {
                current += current.isEmpty ? line : " " + line
            }
            let next = index + 1 < lines.count ? lines[index + 1] : ""
            let endsSentence = line.last.map { ".!?:;\"”’)".contains($0) } ?? false
            if endsSentence, Double(line.count) < Double(typical) * 0.8, !next.isEmpty {
                paragraphs.append(current)
                current = ""
            }
        }
        if !current.isEmpty { paragraphs.append(current) }
        return paragraphs.joined(separator: "\n\n")
    }

    // MARK: - Markdown

    /// Markdown as it reads: its marks gone, each block its own paragraph,
    /// a link its words.
    static func plainText(fromMarkdown markdown: String) -> String {
        let options = AttributedString.MarkdownParsingOptions(
            allowsExtendedAttributes: false, interpretedSyntax: .full,
            failurePolicy: .returnPartiallyParsedIfPossible)
        guard let parsed = try? AttributedString(markdown: markdown, options: options) else {
            return normalized(markdown)
        }
        var blocks: [String] = []
        var current = ""
        var currentBlock: Int?
        for run in parsed.runs {
            let block = run.presentationIntent?.components.first?.identity
            if block != currentBlock {
                if !current.isEmpty { blocks.append(current) }
                current = ""
                currentBlock = block
            }
            current += String(parsed[run.range].characters)
        }
        if !current.isEmpty { blocks.append(current) }
        return normalized(blocks.joined(separator: "\n\n"))
    }

    // MARK: - Web pages

    /// A page's readable text, as Safari's reader would pick it: the
    /// article, without menus, ads or comments. Nil when the page has no
    /// article to read.
    static func text(fromHTML html: String, url: URL?, title: String? = nil) -> IncomingText? {
        let reader = Readability(html: html, url: url ?? URL(string: "about:blank")!)
        guard let article = try? reader.parse(serializer: { paragraphs(of: $0) }) else {
            return nil
        }
        let text = article.content.joined(separator: "\n\n")
        guard hasWords(text) else { return nil }
        let pageTitle = article.title.flatMap { $0.isEmpty ? nil : $0 } ?? title
        return IncomingText(title: pageTitle, text: text)
    }

    private static let blockTags: Set<String> = [
        "p", "div", "section", "article", "main", "header", "footer", "aside", "h1", "h2", "h3",
        "h4", "h5", "h6", "li", "ul", "ol", "dl", "dt", "dd", "blockquote", "pre", "figcaption",
        "table", "tr", "td", "th", "hr",
    ]
    private static let skippedTags: Set<String> = [
        "script", "style", "noscript", "svg", "button", "nav", "form", "iframe",
    ]

    /// An element's text, one paragraph per block, whitespace as a reader
    /// sees it.
    static func paragraphs(of element: Element) -> [String] {
        var result: [String] = []
        var inline = ""
        func flush() {
            let paragraph = inline.split(whereSeparator: \.isWhitespace).joined(separator: " ")
            if !paragraph.isEmpty { result.append(paragraph) }
            inline = ""
        }
        func visit(_ node: Node) {
            if let text = node as? TextNode {
                inline += text.text()
                return
            }
            guard let element = node as? Element else { return }
            let tag = element.tagName().lowercased()
            if skippedTags.contains(tag) { return }
            if tag == "br" {
                flush()
                return
            }
            let isBlock = blockTags.contains(tag)
            if isBlock { flush() }
            for child in element.getChildNodes() { visit(child) }
            if isBlock { flush() }
        }
        visit(element)
        flush()
        return result
    }

    // MARK: - Helpers

    /// Line endings made `\n`, runs of blank lines cut to one, trailing
    /// spaces dropped.
    static func normalized(_ text: String) -> String {
        let lines = text.replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
            .components(separatedBy: "\n")
            .map { $0.replacingOccurrences(of: "\\s+$", with: "", options: .regularExpression) }
        var result: [String] = []
        for line in lines {
            if line.isEmpty, result.last?.isEmpty ?? true { continue }
            result.append(line)
        }
        while result.last?.isEmpty == true { result.removeLast() }
        return result.joined(separator: "\n")
    }

    /// Something the voice can read: a letter or a digit.
    static func hasWords(_ text: String) -> Bool {
        text.unicodeScalars.contains { CharacterSet.alphanumerics.contains($0) }
    }
}
