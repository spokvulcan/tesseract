//
//  ImageIngestTests.swift
//  tesseractTests
//
//  Pins the shared image-ingest core (PRD #112, slice #115): the pure verdict
//  every entry point funnels through. Fixtures cover each accepted raster type,
//  the oversize and non-image rejections, unsupported/unresolvable types, a
//  decoded image normalized to PNG, and mixed-batch separation.
//

import AppKit
import Foundation
import Testing

@testable import Tesseract_Agent

struct ImageIngestTests {

    /// Re-encode the 1×1 fixture into a concrete raster type so "each accepted
    /// type" is exercised with genuinely that-typed bytes (not a relabeled PNG).
    private func encode(_ fileType: NSBitmapImageRep.FileType) -> Data {
        let rep = NSBitmapImageRep(data: ImageTestFixtures.tinyPNGData)!
        return rep.representation(using: fileType, properties: [:])!
    }

    // MARK: - Accepted

    @Test
    func acceptsPNGKeepingOriginalBytesAndMIME() throws {
        let data = ImageTestFixtures.tinyPNGData
        let attachment = try ImageIngest.ingest(
            data: data, typeIdentifier: "public.png", filename: "shot.png"
        ).get()

        #expect(attachment.data == data)       // original bytes preserved
        #expect(attachment.mimeType == "image/png")
        #expect(attachment.filename == "shot.png")
    }

    @Test
    func resolvesIdentifierFromMIMEUTIAndExtension() throws {
        let data = ImageTestFixtures.tinyPNGData
        #expect(try ImageIngest.ingest(data: data, typeIdentifier: "image/png").get().mimeType == "image/png")
        #expect(try ImageIngest.ingest(data: data, typeIdentifier: "public.png").get().mimeType == "image/png")
        #expect(try ImageIngest.ingest(data: data, typeIdentifier: "png").get().mimeType == "image/png")
    }

    @Test
    func acceptsEachSupportedRasterType() throws {
        let cases: [(NSBitmapImageRep.FileType, String)] = [
            (.png, "image/png"), (.jpeg, "image/jpeg"), (.gif, "image/gif"), (.tiff, "image/tiff"),
        ]
        for (fileType, mime) in cases {
            let attachment = try ImageIngest.ingest(data: encode(fileType), typeIdentifier: mime).get()
            #expect(attachment.mimeType == mime)
        }
    }

    @Test
    func normalizesDecodedImageToPNG() throws {
        let nsImage = NSImage(data: ImageTestFixtures.tinyPNGData)!
        let attachment = try ImageIngest.ingest(image: nsImage).get()

        #expect(attachment.mimeType == "image/png")
        #expect(ImageIngest.isDecodableImage(attachment.data))
    }

    // MARK: - Rejected

    @Test
    func rejectsOversizeWithByteCount() {
        let big = Data(count: ImageIngest.maxBytes + 1)
        #expect(ImageIngest.ingest(data: big, typeIdentifier: "image/png")
                == .failure(.oversize(bytes: ImageIngest.maxBytes + 1)))
    }

    @Test
    func rejectsNonImageBytesOfAnAcceptedType() {
        #expect(ImageIngest.ingest(data: Data([0, 1, 2, 3]), typeIdentifier: "image/png")
                == .failure(.notAnImage))
    }

    @Test
    func rejectsUnsupportedType() {
        let result = ImageIngest.ingest(data: ImageTestFixtures.tinyPNGData, typeIdentifier: "public.plain-text")
        guard case .failure(.unsupportedType) = result else {
            Issue.record("expected .unsupportedType, got \(result)"); return
        }
    }

    @Test
    func rejectsUnresolvableType() {
        let result = ImageIngest.ingest(data: ImageTestFixtures.tinyPNGData, typeIdentifier: "not a real type")
        guard case .failure(.unsupportedType) = result else {
            Issue.record("expected .unsupportedType, got \(result)"); return
        }
    }

    // MARK: - Batch cap (slice #117)

    private func attachments(_ n: Int) -> [ImageAttachment] {
        (0..<n).map { _ in ImageAttachment(data: ImageTestFixtures.tinyPNGData, mimeType: "image/png") }
    }

    @Test
    func capBatchFillsRemainingRoomUpToLimit() {
        let batch = attachments(5)
        // 6 already queued, limit 8 → only 2 fit.
        #expect(ImageIngest.capBatch(batch, alreadyQueued: 6, limit: 8).count == 2)
    }

    @Test
    func capBatchAcceptsAllWhenUnderLimit() {
        let batch = attachments(3)
        #expect(ImageIngest.capBatch(batch, alreadyQueued: 0, limit: 8).count == 3)
    }

    @Test
    func capBatchDropsEverythingWhenFull() {
        #expect(ImageIngest.capBatch(attachments(4), alreadyQueued: 8, limit: 8).isEmpty)
        // Defensive: already over the limit yields no negative room.
        #expect(ImageIngest.capBatch(attachments(4), alreadyQueued: 10, limit: 8).isEmpty)
    }

    // MARK: - Multi-input

    @Test
    func mixedBatchSeparatesAcceptedFromRejected() {
        let png = ImageTestFixtures.tinyPNGData
        let inputs: [(Data, String)] = [
            (png, "image/png"),        // accepted
            (Data([1, 2, 3]), "image/png"),   // not an image
            (png, "text/plain"),       // unsupported type
        ]
        let accepted = inputs
            .map { ImageIngest.ingest(data: $0.0, typeIdentifier: $0.1) }
            .compactMap { try? $0.get() }

        #expect(accepted.count == 1)
    }
}
