//
//  PasteboardImageReaderTests.swift
//  tesseractTests
//
//  Tests the Image Gesture test and payload reading against real (named,
//  private) NSPasteboards: what counts as image content, what doesn't, and
//  that a copied file's textual sidecar never masks the image (issue #167).
//

import AppKit
import Foundation
import Testing
import UniformTypeIdentifiers

@testable import Tesseract_Agent

@MainActor
struct PasteboardImageReaderTests {

    /// A private pasteboard per test — never touches NSPasteboard.general.
    private func makePasteboard() -> NSPasteboard {
        NSPasteboard(name: NSPasteboard.Name("test-image-gesture-\(UUID().uuidString)"))
    }

    private func writeTempFile(named name: String, contents: Data) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("reader-tests-\(UUID().uuidString)", isDirectory: true)
            .appendingPathComponent(name)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try contents.write(to: url)
        return url
    }

    @Test
    func stringOnlyPasteboardIsNotAnImageGesture() async {
        let pasteboard = makePasteboard()
        pasteboard.clearContents()
        pasteboard.setString("just some text", forType: .string)

        #expect(PasteboardImageReader.containsImageContent(pasteboard) == false)
        let payload = await PasteboardImageReader.read(pasteboard)
        #expect(payload.isEmpty)
    }

    @Test
    func rawPNGDataIsAnImageGesture() async {
        let pasteboard = makePasteboard()
        pasteboard.clearContents()
        pasteboard.setData(ImageTestFixtures.tinyPNGData, forType: .png)

        #expect(PasteboardImageReader.containsImageContent(pasteboard))
        let payload = await PasteboardImageReader.read(pasteboard)
        #expect(payload.attachments.count == 1)
        #expect(payload.attachments.first?.mimeType == "image/png")
        #expect(payload.rejections.isEmpty)
    }

    @Test
    func copiedImageFileAttachesEvenWithAFilenameStringSidecar() async throws {
        // A Finder file-copy carries both the file URL and the filename as a
        // string — the sidecar must not stop the gesture from reading the image.
        let url = try writeTempFile(named: "shot.png", contents: ImageTestFixtures.tinyPNGData)
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }

        let pasteboard = makePasteboard()
        pasteboard.clearContents()
        pasteboard.writeObjects([url as NSURL])
        pasteboard.setString(url.lastPathComponent, forType: .string)

        #expect(PasteboardImageReader.containsImageContent(pasteboard))
        let payload = await PasteboardImageReader.read(pasteboard)
        #expect(payload.attachments.count == 1)
        #expect(payload.attachments.first?.filename == "shot.png")
    }

    @Test
    func copiedNonImageFileIsNotAnImageGesture() async throws {
        let url = try writeTempFile(named: "notes.txt", contents: Data("hello".utf8))
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }

        let pasteboard = makePasteboard()
        pasteboard.clearContents()
        pasteboard.writeObjects([url as NSURL])
        pasteboard.setString(url.lastPathComponent, forType: .string)

        #expect(PasteboardImageReader.containsImageContent(pasteboard) == false)
        let payload = await PasteboardImageReader.read(pasteboard)
        #expect(payload.isEmpty)
    }

    @Test
    func oversizeRawImageYieldsARejectionNotSilence() async {
        // Oversize content is still an Image Gesture — it must surface as a
        // rejection so the composer can voice it, never as an empty payload.
        // Deliberately undecodable bytes: a decodable oversize image may be
        // legitimately rescued by the NSImage re-encode tier.
        let pasteboard = makePasteboard()
        pasteboard.clearContents()
        pasteboard.setData(Data(count: ImageIngest.maxBytes + 1), forType: .png)

        #expect(PasteboardImageReader.containsImageContent(pasteboard))
        let payload = await PasteboardImageReader.read(pasteboard)
        #expect(payload.attachments.isEmpty)
        #expect(payload.rejections.count == 1)
        if case .oversize = payload.rejections[0] {
        } else {
            Issue.record("expected .oversize, got \(payload.rejections[0])")
        }
    }

    @Test
    func ingestFileURLsSplitsAttachmentsAndRejections() throws {
        let image = try writeTempFile(named: "ok.png", contents: ImageTestFixtures.tinyPNGData)
        let text = try writeTempFile(named: "no.txt", contents: Data("nope".utf8))
        defer {
            try? FileManager.default.removeItem(at: image.deletingLastPathComponent())
            try? FileManager.default.removeItem(at: text.deletingLastPathComponent())
        }

        let payload = PasteboardImageReader.ingest(fileURLs: [image, text])

        #expect(payload.attachments.count == 1)
        #expect(payload.rejections.count == 1)
    }
}
