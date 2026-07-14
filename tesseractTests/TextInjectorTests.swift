//
//  TextInjectorTests.swift
//  tesseractTests
//
//  Exercises the **Clipboard Loan** contract of `TextInjector` against a real,
//  uniquely-named `NSPasteboard` — genuine changeCount semantics, no mocking.
//  The focus probe and paste trigger are seamed so the external-app path runs
//  deterministically and no synthetic Cmd+V ever escapes to the test machine.
//
//  The regression that motivated this suite: the snapshot cap silently
//  skipped saving screenshot clipboards. The deeper cause: a screenshot is
//  one item declaring a ~2 MB PNG, but the pasteboard-level type list also
//  offers server-derived representations (TIFF at 4+ bytes/pixel, each
//  doubled by its legacy-name alias), so a type-level snapshot read tens of
//  megabytes — past any fixed cap for HDR captures — for megabytes of real
//  content. The loan snapshots item-declared content only; the pasteboard
//  server re-derives the rest after restore.
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

    /// Real PNG bytes — genuine image content, so the pasteboard server
    /// synthesizes the derived representations a screenshot clipboard has.
    private static func makePNG() throws -> Data {
        let image = NSImage(size: NSSize(width: 200, height: 120), flipped: false) { rect in
            NSColor.systemTeal.setFill()
            rect.fill()
            return true
        }
        let tiff = try #require(image.tiffRepresentation)
        let rep = try #require(NSBitmapImageRep(data: tiff))
        return try #require(rep.representation(using: .png, properties: [:]))
    }

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

    /// The screenshot regression test. Real PNG content on the pasteboard
    /// makes the server synthesize derived representations (TIFF + legacy
    /// aliases, ~245x the content size), and the cap here sits between the
    /// two: a type-level snapshot overflows it and the screenshot is lost; the
    /// item-level snapshot saves the PNG and the loan returns it — with the
    /// derived TIFF available again to whoever pastes.
    @Test func snapshotSavesContentNotDerivedRepresentations() async throws {
        let pasteboard = NSPasteboard.withUniqueName()
        defer { pasteboard.releaseGlobally() }
        let png = try Self.makePNG()
        pasteboard.clearContents()
        pasteboard.setData(png, forType: .png)
        #expect(pasteboard.data(forType: .tiff) != nil)  // derivation is live

        let injector = makeInjector(
            pasteboard: pasteboard, maxSavedClipboardBytes: 64 * 1024)
        try await injector.inject("dictated text")
        await injector.clipboardReturnTask?.value

        #expect(pasteboard.data(forType: .png) == png)
        #expect(pasteboard.data(forType: .tiff) != nil)
        #expect(pasteboard.string(forType: .string) == nil)
    }

    /// A multi-item clipboard (several files copied at once) must come back
    /// as the same items, not flattened into one.
    @Test func restoresMultiItemClipboard() async throws {
        let pasteboard = NSPasteboard.withUniqueName()
        defer { pasteboard.releaseGlobally() }
        pasteboard.clearContents()
        let items = ["first", "second"].map { text -> NSPasteboardItem in
            let item = NSPasteboardItem()
            item.setString(text, forType: .string)
            return item
        }
        pasteboard.writeObjects(items)

        let injector = makeInjector(pasteboard: pasteboard)
        try await injector.inject("dictated text")
        await injector.clipboardReturnTask?.value

        let restored = pasteboard.pasteboardItems ?? []
        #expect(restored.map { $0.string(forType: .string) } == ["first", "second"])
    }

    /// A payload well over the old 10 MB ceiling must be saved and restored.
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
