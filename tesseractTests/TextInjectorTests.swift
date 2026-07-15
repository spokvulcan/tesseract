//
//  TextInjectorTests.swift
//  tesseractTests
//
//  Exercises the **Clipboard Loan** contract of `TextInjector` primarily
//  against real, uniquely-named `NSPasteboard`s for genuine changeCount
//  semantics. A narrow pasteboard fake covers AppKit failure results that a
//  real server cannot produce deterministically. The focus probe, direct
//  insertion, and paste trigger are seamed so no synthetic Cmd+V or inserted
//  text escapes to the test machine.
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
private final class TestPasteboard: ClipboardPasteboard {
    private(set) var mutationCount = 0
    private(set) var changeCount = 41
    private(set) var restoreWriteAttempts = 0
    private var items: [NSPasteboardItem]?
    private var restoreFailuresRemaining: Int
    private var stringWriteFailuresRemaining: Int
    private var foreignWriteAfterTranscript: String?
    private var setStringCalls = 0

    var pasteboardItems: [NSPasteboardItem]? { items }

    init(
        items: [NSPasteboardItem]?, restoreFailures: Int = 0,
        stringWriteFailures: Int = 0, foreignWriteAfterTranscript: String? = nil
    ) {
        self.items = items
        restoreFailuresRemaining = restoreFailures
        stringWriteFailuresRemaining = stringWriteFailures
        self.foreignWriteAfterTranscript = foreignWriteAfterTranscript
    }

    func clearContents() -> Int {
        mutationCount += 1
        changeCount += 1
        items = []
        return changeCount
    }

    func setString(_ string: String, forType dataType: NSPasteboard.PasteboardType) -> Bool {
        mutationCount += 1
        if stringWriteFailuresRemaining > 0 {
            stringWriteFailuresRemaining -= 1
            return false
        }
        let item: NSPasteboardItem
        if let first = items?.first {
            item = first
        } else {
            item = NSPasteboardItem()
            items = [item]
        }
        let written = item.setString(string, forType: dataType)
        setStringCalls += 1
        // Simulate another process taking the pasteboard the instant the
        // transcript's three writes (text + both privacy markers) complete —
        // the narrowest window between transport write and Cmd+V.
        if setStringCalls == 3, let foreign = foreignWriteAfterTranscript {
            foreignWriteAfterTranscript = nil
            changeCount += 1
            let foreignItem = NSPasteboardItem()
            _ = foreignItem.setString(foreign, forType: .string)
            items = [foreignItem]
        }
        return written
    }

    func writeObjects(_ objects: [any NSPasteboardWriting]) -> Bool {
        mutationCount += 1
        restoreWriteAttempts += 1
        if restoreFailuresRemaining > 0 {
            restoreFailuresRemaining -= 1
            return false
        }
        items = objects.compactMap { $0 as? NSPasteboardItem }
        return items?.count == objects.count
    }

    func string(forType type: NSPasteboard.PasteboardType) -> String? {
        items?.first?.string(forType: type)
    }
}

@MainActor
@Suite("TextInjector clipboard loan")
struct TextInjectorTests {
    private static let transientMarker = NSPasteboard.PasteboardType(
        "org.nspasteboard.TransientType")
    private static let concealedMarker = NSPasteboard.PasteboardType(
        "org.nspasteboard.ConcealedType")

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
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("text-injector-files-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let urls = ["first.txt", "second.txt"].map { directory.appendingPathComponent($0) }
        for (index, url) in urls.enumerated() {
            try Data("file \(index)".utf8).write(to: url)
        }
        pasteboard.clearContents()
        #expect(pasteboard.writeObjects(urls.map { $0 as NSURL }))

        let injector = makeInjector(pasteboard: pasteboard)
        try await injector.inject("dictated text")
        await injector.clipboardReturnTask?.value

        let restored =
            (pasteboard.readObjects(
                forClasses: [NSURL.self],
                options: [.urlReadingFileURLsOnly: true]) as? [NSURL])?.map { $0 as URL }
        #expect(restored == urls)
    }

    @Test func restoresWebURLClipboard() async throws {
        let pasteboard = NSPasteboard.withUniqueName()
        defer { pasteboard.releaseGlobally() }
        let url = try #require(NSURL(string: "https://example.com/path?q=1"))
        pasteboard.clearContents()
        #expect(pasteboard.writeObjects([url]))

        let injector = makeInjector(pasteboard: pasteboard)
        try await injector.inject("dictated text")
        await injector.clipboardReturnTask?.value

        let restored =
            pasteboard.readObjects(forClasses: [NSURL.self], options: nil)
            as? [NSURL]
        #expect(restored == [url])
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

    @Test func cancellationBeforeInjectionDoesNotMutateClipboard() async throws {
        let pasteboard = NSPasteboard.withUniqueName()
        defer { pasteboard.releaseGlobally() }
        pasteboard.clearContents()
        pasteboard.setString("before", forType: .string)

        var pasted = false
        let injector = makeInjector(pasteboard: pasteboard, onPaste: { pasted = true })
        let caller = Task { try await injector.inject("dictated text") }
        caller.cancel()
        _ = await caller.result

        await injector.clipboardReturnTask?.value
        #expect(pasted == false)
        #expect(pasteboard.string(forType: .string) == "before")
    }

    /// Once Cmd+V is posted, cancellation of the owning commit task must not
    /// cancel the independent loan return.
    @Test func loanReturnOutlivesCallerCancellation() async throws {
        let pasteboard = NSPasteboard.withUniqueName()
        defer { pasteboard.releaseGlobally() }
        pasteboard.clearContents()
        pasteboard.setString("before", forType: .string)

        var caller: Task<Void, Error>?
        let injector = makeInjector(
            pasteboard: pasteboard,
            onPaste: { caller?.cancel() })
        caller = Task {
            try await injector.inject("dictated text")
            try await Task.sleep(for: .seconds(10))
        }

        let callerTask = try #require(caller)
        _ = await callerTask.result
        await injector.clipboardReturnTask?.value

        #expect(pasteboard.string(forType: .string) == "before")
    }

    /// One loan at a time: an injection arriving while the previous return is
    /// still pending settles that loan first, so its own snapshot sees the
    /// restored original — never the previous transcript — and the original
    /// content survives the whole burst.
    @Test func rapidReinjectionPreservesOriginalClipboard() async throws {
        let pasteboard = NSPasteboard.withUniqueName()
        defer { pasteboard.releaseGlobally() }
        pasteboard.clearContents()
        pasteboard.setString("before", forType: .string)

        var pastedTexts: [String?] = []
        let injector = makeInjector(
            pasteboard: pasteboard,
            onPaste: { pastedTexts.append(pasteboard.string(forType: .string)) })
        try await injector.inject("first take")
        // The first call returns with its loan still out, so the next call
        // deterministically enters inside the read/return window.
        try await injector.inject("second take")

        await injector.clipboardReturnTask?.value
        #expect(pastedTexts == ["first take", "second take"])
        #expect(pasteboard.string(forType: .string) == "before")
    }

    /// Waiting for the previous loan can span a focus change. The route must
    /// be chosen from the current frontmost app after that wait, not a stale
    /// pre-wait sample.
    @Test func focusIsReevaluatedAfterWaitingForPreviousLoan() async throws {
        let pasteboard = NSPasteboard.withUniqueName()
        defer { pasteboard.releaseGlobally() }
        pasteboard.clearContents()
        pasteboard.setString("before", forType: .string)

        var ownAppFocused = false
        var pastedTexts: [String?] = []
        let injector = TextInjector(
            pasteboard: pasteboard,
            ownAppFocused: { ownAppFocused },
            insertText: { _ in false },  // own app focused, but no editable field
            paste: { pastedTexts.append(pasteboard.string(forType: .string)) })
        try await injector.inject("first take")

        let focusChange = Task {
            try await Task.sleep(for: .milliseconds(10))
            ownAppFocused = true
        }
        try await injector.inject("second take")
        _ = await focusChange.result

        #expect(pastedTexts == ["first take"])
        #expect(pasteboard.string(forType: .string) == "second take")
    }

    /// The restore must replay each item's type declaration order — it is the
    /// source app's fidelity preference, and consumers may walk it in order.
    @Test func restoredItemPreservesTypeOrder() async throws {
        let pasteboard = NSPasteboard.withUniqueName()
        defer { pasteboard.releaseGlobally() }
        pasteboard.clearContents()
        let declared = ["com.example.rich", "com.example.mid", "com.example.plain"]
            .map { NSPasteboard.PasteboardType($0) }
        let item = NSPasteboardItem()
        for (index, type) in declared.enumerated() {
            item.setData(Data([UInt8(index)]), forType: type)
        }
        pasteboard.writeObjects([item])

        let injector = makeInjector(pasteboard: pasteboard)
        try await injector.inject("dictated text")
        await injector.clipboardReturnTask?.value

        let restored = try #require(pasteboard.pasteboardItems?.first)
        #expect(restored.types == declared)
        #expect(restored.data(forType: declared[0]) == Data([0]))
    }

    /// A retrieval failure is not an empty clipboard. The loan must abort
    /// before its first mutation so unseen user content cannot be destroyed.
    @Test func unreadableClipboardIsNeverMutated() async {
        let pasteboard = TestPasteboard(items: nil)
        var pasted = false
        let injector = TextInjector(
            pasteboard: pasteboard,
            ownAppFocused: { false },
            paste: { pasted = true })

        await #expect(throws: DictationError.self) {
            try await injector.inject("dictated text")
        }

        #expect(pasteboard.mutationCount == 0)
        #expect(pasted == false)
        #expect(injector.clipboardReturnTask == nil)
    }

    /// A transient AppKit write failure must not turn the non-atomic
    /// clear-then-write restore into permanent clipboard data loss.
    @Test func transientRestoreFailureRetriesOriginalClipboard() async throws {
        let original = NSPasteboardItem()
        _ = original.setString("before", forType: .string)
        let pasteboard = TestPasteboard(items: [original], restoreFailures: 1)
        let injector = TextInjector(
            pasteboard: pasteboard,
            ownAppFocused: { false },
            paste: {})

        try await injector.inject("dictated text")
        await injector.clipboardReturnTask?.value

        #expect(pasteboard.restoreWriteAttempts == 2)
        #expect(pasteboard.string(forType: .string) == "before")
    }

    /// Failed transcript transport is not a successful paste. The saved
    /// clipboard is rolled back and Cmd+V is never posted.
    @Test func transcriptWriteFailureRollsBackWithoutPasting() async {
        let original = NSPasteboardItem()
        _ = original.setString("before", forType: .string)
        let pasteboard = TestPasteboard(items: [original], stringWriteFailures: 1)
        var pasted = false
        let injector = TextInjector(
            pasteboard: pasteboard,
            ownAppFocused: { false },
            paste: { pasted = true })

        await #expect(throws: DictationError.self) {
            try await injector.inject("dictated text")
        }
        await injector.clipboardReturnTask?.value

        #expect(pasted == false)
        #expect(pasteboard.string(forType: .string) == "before")
    }

    /// A foreign writer taking the pasteboard between the transcript write
    /// and the paste must abort the injection — Cmd+V would deliver *their*
    /// content as the dictation — and the rollback must not stomp their copy.
    @Test func foreignWriteBeforePasteAbortsWithoutStompingIt() async throws {
        let original = NSPasteboardItem()
        _ = original.setString("before", forType: .string)
        let pasteboard = TestPasteboard(
            items: [original], foreignWriteAfterTranscript: "their copy")
        var pasted = false
        let injector = TextInjector(
            pasteboard: pasteboard,
            ownAppFocused: { false },
            paste: { pasted = true })

        await #expect(throws: DictationError.self) {
            try await injector.inject("dictated text")
        }
        await injector.clipboardReturnTask?.value

        #expect(pasted == false)
        #expect(pasteboard.string(forType: .string) == "their copy")
    }

    /// Exhausting the immediate retry budget must retain the snapshot. The
    /// next clipboard use recovers it before taking another loan.
    @Test func persistentRestoreFailureIsRecoveredBeforeNextInjection() async throws {
        let original = NSPasteboardItem()
        _ = original.setString("before", forType: .string)
        let pasteboard = TestPasteboard(items: [original], restoreFailures: 3)
        let injector = TextInjector(
            pasteboard: pasteboard,
            ownAppFocused: { false },
            paste: {})

        try await injector.inject("first take")
        await injector.clipboardReturnTask?.value
        #expect(pasteboard.restoreWriteAttempts == 3)

        try await injector.inject("second take")
        await injector.clipboardReturnTask?.value

        #expect(pasteboard.restoreWriteAttempts == 5)
        #expect(pasteboard.string(forType: .string) == "before")
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
        #expect(pasteboard.types?.contains(Self.concealedMarker) == true)
    }
}
