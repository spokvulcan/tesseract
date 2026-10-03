//
//  SharedContent.swift
//  tesseract
//
//  What the share sheet hands "Read in Tesseract", turned into a text: the
//  readable text of a page Safari shared (Safari runs the extension's
//  script on the page, so the app never fetches it), a PDF, a text file, or
//  plain text.
//

import Foundation
import PDFKit
import UniformTypeIdentifiers

nonisolated enum SharedContent {
    enum Failure: Error, Equatable {
        /// Nothing shared that has text: an image, a bare link from an app
        /// that isn't Safari.
        case nothingToRead
    }

    /// The text in `items`, the first kind that has one: a page, a PDF, a
    /// text file, plain text.
    static func text(from items: [NSExtensionItem]) async throws -> IncomingText {
        let providers = items.flatMap { $0.attachments ?? [] }
        for provider in providers
        where provider.hasItemConformingToTypeIdentifier(UTType.propertyList.identifier) {
            if let page = try? await page(from: provider) { return page }
        }
        for provider in providers
        where provider.hasItemConformingToTypeIdentifier(UTType.pdf.identifier) {
            if let data = try? await data(of: .pdf, from: provider),
                let document = PDFDocument(data: data)
            {
                let incoming = TextIntake.text(
                    fromPDF: document, fallbackTitle: provider.suggestedName)
                if TextIntake.hasWords(incoming.text) { return incoming }
            }
        }
        for provider in providers
        where provider.hasItemConformingToTypeIdentifier(UTType.plainText.identifier) {
            let item = try? await provider.loadItem(forTypeIdentifier: UTType.plainText.identifier)
            if let url = item as? URL, url.isFileURL,
                let incoming = try? TextIntake.text(fromFile: url)
            {
                return incoming
            }
            let text =
                (item as? String) ?? (item as? Data).flatMap { String(data: $0, encoding: .utf8) }
            if let text, TextIntake.hasWords(text) {
                return IncomingText(title: nil, text: TextIntake.normalized(text))
            }
        }
        throw Failure.nothingToRead
    }

    /// The results of the extension's script on a Safari page: its title,
    /// address and HTML.
    private static func page(from provider: NSItemProvider) async throws -> IncomingText? {
        let item = try await provider.loadItem(forTypeIdentifier: UTType.propertyList.identifier)
        guard let dictionary = item as? NSDictionary,
            let results = dictionary[NSExtensionJavaScriptPreprocessingResultsKey]
                as? [String: Any],
            let html = results["html"] as? String
        else { return nil }
        let url = (results["url"] as? String).flatMap(URL.init(string:))
        return TextIntake.text(fromHTML: html, url: url, title: results["title"] as? String)
    }

    private static func data(of type: UTType, from provider: NSItemProvider) async throws -> Data {
        try await withCheckedThrowingContinuation { continuation in
            _ = provider.loadDataRepresentation(for: type) { data, error in
                if let data {
                    continuation.resume(returning: data)
                } else {
                    continuation.resume(throwing: error ?? Failure.nothingToRead)
                }
            }
        }
    }
}
