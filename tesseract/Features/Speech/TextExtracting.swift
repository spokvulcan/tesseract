//
//  TextExtracting.swift
//  tesseract
//

/// Where the speech hotkey's text comes from: the selection in whatever app
/// is in front. The Mac's adapter copies it through the pasteboard
/// (`TextExtractor`); tests use an in-memory one.
@MainActor
protocol TextExtracting: AnyObject {
    func extractSelectedText() async throws -> String
}
