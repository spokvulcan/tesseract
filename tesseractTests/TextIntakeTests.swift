//
//  TextIntakeTests.swift
//  tesseractTests
//
//  Getting text in (#515): a saved web page's article, a PDF's paragraphs,
//  Markdown without its marks, and the share extension's inbox.
//

import Foundation
import PDFKit
import Testing
import UniformTypeIdentifiers

@testable import Tesseract_Agent

@MainActor
struct TextIntakeTests {

    /// A news page as Safari saves it: menus, an ad, the article, comments.
    private static let page = """
        <!doctype html><html><head><title>Harbor News — The quiet night</title></head>
        <body>
        <nav><a href="/">Home</a> <a href="/world">World</a> <a href="/sport">Sport</a></nav>
        <div class="ad-banner">Buy one boat, get one free!</div>
        <article>
          <h1>The quiet night at the harbor</h1>
          <p>The river watched the harbor all night long. Nothing moved on the water, and the
          lights of the town went out one by one.</p>
          <p>A small boat returned home before dawn. Its keeper tied it to the pier and climbed
          the old stone steps, as he had done for <em>forty years</em>.</p>
          <p>By morning the fog had lifted, and the fishermen were already at work, mending
          their nets in the cold air and talking about the weather.</p>
        </article>
        <section class="comments"><p>Great story!</p></section>
        <footer>© Harbor News</footer>
        </body></html>
        """

    @Test func aPageGivesItsArticleAndNothingElse() throws {
        let incoming = try #require(
            TextIntake.text(fromHTML: Self.page, url: URL(string: "https://news.example/night")))
        let paragraphs = incoming.text.components(separatedBy: "\n\n")
        #expect(
            paragraphs.contains(
                "The river watched the harbor all night long. Nothing moved on the water, and the lights of the town went out one by one."
            ))
        #expect(incoming.text.contains("as he had done for forty years."))
        #expect(!incoming.text.contains("Buy one boat"))
        #expect(!incoming.text.contains("Home"))
        #expect(!incoming.text.contains("Great story"))
        #expect(incoming.title?.contains("quiet night") == true)
    }

    @Test func aPageWithNothingToReadGivesNothing() {
        #expect(
            TextIntake.text(fromHTML: "<html><body><nav>Home</nav></body></html>", url: nil) == nil)
    }

    @Test func markdownReadsWithoutItsMarks() {
        let markdown = """
            # The Harbor

            The river watched the **harbor** all night. See [the map](https://example.com/map).

            - A small boat
            - The keeper's `steps`

            > Nothing moved.
            """
        #expect(
            TextIntake.plainText(fromMarkdown: markdown) == """
                The Harbor

                The river watched the harbor all night. See the map.

                A small boat

                The keeper's steps

                Nothing moved.
                """)
    }

    @Test func aPDFsLinesJoinBackIntoParagraphs() {
        // A paragraph's short last line ends it; a hyphen at a line's end
        // joins the word back.
        let pdfText = """
            The river watched the harbor all night long. Nothing moved
            on the water, and the lights of the town went out one by
            one.
            A small boat returned home before dawn, and its keeper tied
            it to the pier.

            By morning the fog had lifted, and the fish-
            ermen were already at work, mending their nets.
            """
        #expect(
            TextIntake.reflowed(pdfText) == """
                The river watched the harbor all night long. Nothing moved on the water, and the lights of the town went out one by one.

                A small boat returned home before dawn, and its keeper tied it to the pier.

                By morning the fog had lifted, and the fishermen were already at work, mending their nets.
                """)
    }

    @Test func aPDFFileGivesItsTextAndTitle() throws {
        let url = makeTempDir("intake").appendingPathComponent("harbor.pdf")
        try Self.makePDF(
            lines: ["The river watched the harbor.", "A small boat returned home."], at: url)
        let incoming = try TextIntake.text(fromFile: url)
        #expect(incoming.text.contains("The river watched the harbor."))
        #expect(incoming.text.contains("A small boat returned home."))
        #expect(incoming.title == "harbor")
    }

    @Test func textAndMarkdownFilesAreRead() throws {
        let folder = makeTempDir("intake")
        let text = folder.appendingPathComponent("note.txt")
        try "Line one.\r\nLine two.\r\n\r\n\r\n\r\nThe end.  \n".write(
            to: text, atomically: true, encoding: .utf8)
        #expect(try TextIntake.text(fromFile: text).text == "Line one.\nLine two.\n\nThe end.")

        let markdown = folder.appendingPathComponent("note.md")
        try "## A heading\n\nSome *text*.".write(to: markdown, atomically: true, encoding: .utf8)
        #expect(try TextIntake.text(fromFile: markdown).text == "A heading\n\nSome text.")
    }

    @Test func aFileWithNoWordsIsRefused() throws {
        let empty = makeTempDir("intake").appendingPathComponent("empty.txt")
        try "   \n\n".write(to: empty, atomically: true, encoding: .utf8)
        #expect(throws: TextIntake.Failure.noText) { try TextIntake.text(fromFile: empty) }
    }

    @Test func theInboxHandsItsTextsToTheLibraryOldestFirst() throws {
        let inbox = LibraryInbox(directory: makeTempDir("inbox"))
        let now = Date.now
        try inbox.drop(
            IncomingText(title: "First", text: "The first text."), at: now.addingTimeInterval(-10))
        try inbox.drop(IncomingText(title: nil, text: "The second text."), at: now)
        let library = ReaderLibrary(directory: makeTempDir("library"))

        let added = library.takeIn(from: inbox)
        #expect(added.map(\.title) == ["First", "The second text."])
        #expect(library.entries.map(\.title) == ["The second text.", "First"])
        #expect(inbox.pending().isEmpty)
        #expect(library.takeIn(from: inbox).isEmpty)
    }

    // MARK: - The share sheet

    @Test func safarisPageResultsAreReadAsTheirArticle() async throws {
        let results: NSDictionary = [
            NSExtensionJavaScriptPreprocessingResultsKey: [
                "title": "Harbor News", "url": "https://news.example/night", "html": Self.page,
            ]
        ]
        let item = NSExtensionItem()
        item.attachments = [
            NSItemProvider(item: results, typeIdentifier: UTType.propertyList.identifier),
            NSItemProvider(
                item: URL(string: "https://news.example/night")! as NSURL,
                typeIdentifier: UTType.url.identifier),
        ]
        let incoming = try await SharedContent.text(from: [item])
        // The heading is the title, not the first line, as in Safari's reader.
        #expect(incoming.text.hasPrefix("The river watched the harbor all night long."))
        #expect(incoming.title?.contains("quiet night") == true)
        #expect(!incoming.text.contains("Buy one boat"))
    }

    @Test func sharedPlainTextIsReadAsItIs() async throws {
        let item = NSExtensionItem()
        item.attachments = [
            NSItemProvider(
                item: "A selection from a note.\r\nSecond line." as NSString,
                typeIdentifier: UTType.plainText.identifier)
        ]
        let incoming = try await SharedContent.text(from: [item])
        #expect(
            incoming == IncomingText(title: nil, text: "A selection from a note.\nSecond line."))
    }

    @Test func aSharedPDFIsRead() async throws {
        let url = makeTempDir("intake").appendingPathComponent("shared.pdf")
        try Self.makePDF(lines: ["The keeper climbed the steps."], at: url)
        let item = NSExtensionItem()
        item.attachments = [NSItemProvider(contentsOf: url)!]
        let incoming = try await SharedContent.text(from: [item])
        #expect(incoming.text.contains("The keeper climbed the steps."))
    }

    @Test func sharingSomethingWithoutTextIsRefused() async {
        let item = NSExtensionItem()
        item.attachments = [
            NSItemProvider(item: Data([0, 1, 2]) as NSData, typeIdentifier: UTType.png.identifier)
        ]
        await #expect(throws: SharedContent.Failure.nothingToRead) {
            try await SharedContent.text(from: [item])
        }
    }

    /// A one-page PDF with `lines` of text, drawn as a PDF context would.
    private static func makePDF(lines: [String], at url: URL) throws {
        var box = CGRect(x: 0, y: 0, width: 612, height: 792)
        guard let context = CGContext(url as CFURL, mediaBox: &box, nil) else {
            throw CocoaError(.fileWriteUnknown)
        }
        context.beginPDFPage(nil)
        let font = CTFontCreateWithName("Helvetica" as CFString, 14, nil)
        for (index, line) in lines.enumerated() {
            let string = NSAttributedString(string: line, attributes: [.font: font])
            let ctLine = CTLineCreateWithAttributedString(string)
            context.textPosition = CGPoint(x: 72, y: 700 - CGFloat(index) * 24)
            CTLineDraw(ctLine, context)
        }
        context.endPDFPage()
        context.closePDF()
    }
}
