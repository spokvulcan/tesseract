//
//  TextInjectorTests.swift
//  tesseractTests
//
//  Exercises the **Clipboard Loan** contract of `TextInjector` against a real,
//  uniquely-named `NSPasteboard` — genuine changeCount semantics, no mocking.
//  The focus probe and paste trigger are seamed so the external-app path runs
//  deterministically and no synthetic Cmd+V ever escapes to the test machine.
//
//  The regression that motivated this suite: the 10 MB snapshot cap silently
//  skipped saving screenshot-sized clipboards (a full-screen TIFF is 20–80 MB),
//  so the dictation squatted on the pasteboard and the screenshot was lost.
//

import AppKit
import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
@Suite("TextInjector clipboard loan")
struct TextInjectorTests {
    private static let transientMarker = NSPasteboard.PasteboardType(
        "org.nspasteboard.TransientType")

    /// An injector wired to a fresh unique pasteboard, never our own app,
    /// with the synthetic paste stubbed out.
    private func makeInjector(
        pasteboard: NSPasteboard, maxSavedClipboardBytes: Int = 128 * 1024 * 1024,
        onPaste: @escaping @MainActor () -> Void = {}
    ) -> TextInjector {
        TextInjector(
            pasteboard: pasteboard,
            maxSavedClipboardBytes: maxSavedClipboardBytes,
            ownAppFocused: { false },
            paste: onPaste)
    }

    @Test func restoresSmallTextAfterPaste() async throws {
        let pasteboard = NSPasteboard.withUniqueName()
        defer { pasteboard.releaseGlobally() }
        pasteboard.clearContents()
        pasteboard.setString("https://example.com", forType: .string)

        var pasteCount = 0
        let injector = makeInjector(pasteboard: pasteboard, onPaste: { pasteCount += 1 })
        try await injector.inject("dictated text")

        // The transport phase: the transcript is on the pasteboard, pasted once.
        #expect(pasteCount == 1)
        #expect(pasteboard.string(forType: .string) == "dictated text")

        await injector.clipboardReturnTask?.value
        #expect(pasteboard.string(forType: .string) == "https://example.com")
    }

    /// The regression test for the mis-sized cap: a screenshot-scale payload
    /// (well over the old 10 MB ceiling) must be saved and restored.
    @Test func restoresScreenshotSizedPayload() async throws {
        let pasteboard = NSPasteboard.withUniqueName()
        defer { pasteboard.releaseGlobally() }
        pasteboard.clearContents()
        let screenshot = Data(count: 24 * 1024 * 1024)
        pasteboard.setData(screenshot, forType: .tiff)

        let injector = makeInjector(pasteboard: pasteboard)
        try await injector.inject("dictated text")
        await injector.clipboardReturnTask?.value

        #expect(pasteboard.data(forType: .tiff) == screenshot)
        #expect(pasteboard.string(forType: .string) == nil)
    }

    /// Over the cap there is nothing to restore — the loan return must clear
    /// the transcript so a later Cmd+V cannot re-paste the dictation.
    @Test func clearsTranscriptWhenSnapshotOverCap() async throws {
        let pasteboard = NSPasteboard.withUniqueName()
        defer { pasteboard.releaseGlobally() }
        pasteboard.clearContents()
        pasteboard.setString("payload beyond the tiny test cap", forType: .string)

        let injector = makeInjector(pasteboard: pasteboard, maxSavedClipboardBytes: 8)
        try await injector.inject("dictated text")
        await injector.clipboardReturnTask?.value

        #expect(pasteboard.string(forType: .string) == nil)
        #expect(pasteboard.pasteboardItems?.isEmpty ?? true)
    }

    /// An empty pre-dictation clipboard is a loan of emptiness: the return
    /// clears the transcript rather than leaving it to be re-pasted.
    @Test func clearsTranscriptWhenClipboardWasEmpty() async throws {
        let pasteboard = NSPasteboard.withUniqueName()
        defer { pasteboard.releaseGlobally() }
        pasteboard.clearContents()

        let injector = makeInjector(pasteboard: pasteboard)
        try await injector.inject("dictated text")
        #expect(pasteboard.string(forType: .string) == "dictated text")

        await injector.clipboardReturnTask?.value
        #expect(pasteboard.string(forType: .string) == nil)
        #expect(pasteboard.pasteboardItems?.isEmpty ?? true)
    }

    /// The changeCount guard: content the user copies inside the return window
    /// wins — the loan return must neither restore nor clear over it.
    @Test func userCopyDuringReturnWindowWins() async throws {
        let pasteboard = NSPasteboard.withUniqueName()
        defer { pasteboard.releaseGlobally() }
        pasteboard.clearContents()
        pasteboard.setString("before", forType: .string)

        let injector = makeInjector(pasteboard: pasteboard)
        try await injector.inject("dictated text")

        // No suspension between inject returning and this write, so it lands
        // before the deferred return task can run its changeCount check.
        pasteboard.clearContents()
        pasteboard.setString("user copy", forType: .string)

        await injector.clipboardReturnTask?.value
        #expect(pasteboard.string(forType: .string) == "user copy")
    }

    /// Restore mode off is dictate-to-clipboard: the transcript stays, no
    /// return task runs, and the privacy markers ride the write.
    @Test func restoreOffLeavesTranscriptWithPrivacyMarkers() async throws {
        let pasteboard = NSPasteboard.withUniqueName()
        defer { pasteboard.releaseGlobally() }
        pasteboard.clearContents()
        pasteboard.setString("before", forType: .string)

        let injector = makeInjector(pasteboard: pasteboard)
        injector.restoreClipboard = false
        try await injector.inject("dictated text")

        #expect(injector.clipboardReturnTask == nil)
        #expect(pasteboard.string(forType: .string) == "dictated text")
        #expect(pasteboard.types?.contains(Self.transientMarker) == true)
    }
}
