//
//  ImageGesture.swift
//  tesseract
//
//  The **Image Gesture** reading layer (issue #167): one payload shape and one
//  set of readers for every inbound image delivery — ⌘V paste, a drag onto the
//  composer, a drag onto the window, the file picker. A gesture is keyed on
//  *content, not outcome*: once a pasteboard or provider set carries image
//  content, it resolves as an image action and the caller must never fall back
//  to inserting the payload's textual sidecar (file name, path, source URL).
//

import AppKit
import Foundation
import UniformTypeIdentifiers

/// What one Image Gesture yielded: the attachments that ingested cleanly plus
/// the typed rejections for everything that didn't. A payload with both empty
/// means the source carried no image content at all (not an Image Gesture).
nonisolated struct ImageGesturePayload: Sendable {
    var attachments: [ImageAttachment] = []
    var rejections: [ImageIngest.Rejection] = []

    var isEmpty: Bool { attachments.isEmpty && rejections.isEmpty }
}

// MARK: - Pasteboard reading (⌘V and AppKit drags)

/// The impure pasteboard edge shared by ⌘V and composer drags: decides whether
/// a pasteboard carries image content (the Image Gesture test) and reads it
/// into a payload. Tries the richest source first: copied image *file* URLs
/// (⌘C on a file in Finder), raw image data of a supported type, a decoded
/// `NSImage` (copied in Preview or a browser), then file promises (the
/// screenshot floating thumbnail, Photos). The first source that yields a
/// result wins, so a single image never double-attaches.
enum PasteboardImageReader {

    private static let imageURLOptions: [NSPasteboard.ReadingOptionKey: Any] = [
        .urlReadingContentsConformToTypes: [UTType.image.identifier]
    ]

    /// The supported image set as pasteboard types — one derivation shared by
    /// the content probe and the composer's drag registration, so the two can
    /// never drift apart.
    static let supportedPasteboardTypes: [NSPasteboard.PasteboardType] =
        ImageIngest.supportedUTTypes.map { NSPasteboard.PasteboardType($0.identifier) }

    /// The gesture test: does this pasteboard carry image content in any form
    /// we can read? Deliberately cheap — type/conformance checks only, no byte
    /// reads or decodes — so it can run inside menu validation and drag
    /// tracking.
    static func containsImageContent(_ pasteboard: NSPasteboard) -> Bool {
        if pasteboard.canReadObject(forClasses: [NSURL.self], options: imageURLOptions) {
            return true
        }
        if pasteboard.availableType(from: supportedPasteboardTypes) != nil { return true }
        if NSImage.canInit(with: pasteboard) { return true }
        if !imagePromiseReceivers(pasteboard).isEmpty { return true }
        return false
    }

    /// Read the pasteboard's image content into a payload. Async only for the
    /// file-promise tier, which must wait for the promising app to materialize
    /// the file.
    static func read(_ pasteboard: NSPasteboard) async -> ImageGesturePayload {
        // 1. Image file URLs.
        if let urls = pasteboard.readObjects(forClasses: [NSURL.self], options: imageURLOptions)
            as? [URL],
            !urls.isEmpty
        {
            let payload = ingest(fileURLs: urls)
            if !payload.isEmpty { return payload }
        }

        // 2. Raw image data of a supported type. Failures don't return early:
        // an oversize TIFF may sit next to a PNG that fits, and the decoded-
        // image tier below can still re-encode something attachable.
        var rejections: [ImageIngest.Rejection] = []
        for utType in ImageIngest.supportedUTTypes {
            let type = NSPasteboard.PasteboardType(utType.identifier)
            guard let data = pasteboard.data(forType: type) else { continue }
            switch ImageIngest.ingest(
                data: data, typeIdentifier: utType.identifier, filename: "pasted-image"
            ) {
            case .success(let attachment):
                return ImageGesturePayload(attachments: [attachment])
            case .failure(let rejection):
                rejections.append(rejection)
            }
        }

        // 3. Decoded image with no file backing.
        if let image = NSImage(pasteboard: pasteboard) {
            switch ImageIngest.ingest(image: image, filename: "pasted-image.png") {
            case .success(let attachment):
                return ImageGesturePayload(attachments: [attachment])
            case .failure(let rejection):
                rejections.append(rejection)
            }
        }
        if !rejections.isEmpty {
            return ImageGesturePayload(rejections: [rejections[0]])
        }

        // 4. File promises — the screenshot floating thumbnail and Photos
        // promise a file that doesn't exist yet; receive it into a scratch
        // directory and ingest the materialized bytes.
        let receivers = imagePromiseReceivers(pasteboard)
        if !receivers.isEmpty {
            return await receivePromisedImages(receivers)
        }

        return ImageGesturePayload()
    }

    /// Ingest file URLs into a payload — shared by the pasteboard tier, the
    /// materialized-promise tier, and the file picker.
    nonisolated static func ingest(fileURLs urls: [URL]) -> ImageGesturePayload {
        var payload = ImageGesturePayload()
        for url in urls {
            guard let data = try? Data(contentsOf: url) else {
                payload.rejections.append(.notAnImage)
                continue
            }
            let uti =
                (try? url.resourceValues(forKeys: [.contentTypeKey]).contentType?.identifier)
                ?? UTType(filenameExtension: url.pathExtension)?.identifier
                ?? url.pathExtension
            switch ImageIngest.ingest(
                data: data, typeIdentifier: uti, filename: url.lastPathComponent
            ) {
            case .success(let attachment): payload.attachments.append(attachment)
            case .failure(let rejection): payload.rejections.append(rejection)
            }
        }
        return payload
    }

    // MARK: - File promises

    /// The pasteboard's file-promise receivers whose promised types include an
    /// image. Reading receivers only inspects the promised type list — no file
    /// is materialized until `receivePromisedImages`.
    private static func imagePromiseReceivers(_ pasteboard: NSPasteboard)
        -> [NSFilePromiseReceiver]
    {
        guard
            let receivers = pasteboard.readObjects(
                forClasses: [NSFilePromiseReceiver.self], options: nil
            ) as? [NSFilePromiseReceiver]
        else { return [] }
        return receivers.filter { receiver in
            receiver.fileTypes.contains { UTType($0)?.conforms(to: .image) == true }
        }
    }

    /// Ask each receiver to materialize its promised files into a scratch
    /// directory, then ingest the bytes. The directory is removed afterwards —
    /// `ImageIngest` copies the bytes into the attachment.
    private static func receivePromisedImages(
        _ receivers: [NSFilePromiseReceiver]
    ) async -> ImageGesturePayload {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("image-gesture-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        var payload = ImageGesturePayload()
        for receiver in receivers {
            let expected = max(1, receiver.fileTypes.count)
            let urls: [URL] = await withCheckedContinuation { continuation in
                var received: [URL] = []
                var completed = 0
                receiver.receivePromisedFiles(
                    atDestination: directory, options: [:], operationQueue: .main
                ) { url, error in
                    if error == nil { received.append(url) }
                    completed += 1
                    if completed == expected { continuation.resume(returning: received) }
                }
            }
            let ingested = ingest(fileURLs: urls)
            payload.attachments.append(contentsOf: ingested.attachments)
            payload.rejections.append(contentsOf: ingested.rejections)
            payload.rejections.append(
                contentsOf: repeatElement(.notAnImage, count: max(0, expected - urls.count)))
        }
        return payload
    }
}

// MARK: - Item-provider reading (SwiftUI window drop)

/// The impure `NSItemProvider` edge for the full-window SwiftUI drop (slice
/// #117 delivery of the same Image Gesture).
enum ImageItemProviderReader {

    static func load(_ providers: [NSItemProvider]) async -> ImageGesturePayload {
        // Loads run serially: NSItemProvider is not Sendable, so a task-group
        // fan-out can't cross the isolation boundary, and drops are small
        // (cap is 8) — determinism over concurrency machinery here.
        var payload = ImageGesturePayload()
        for provider in providers {
            let preferredUTI =
                provider.registeredTypeIdentifiers.first {
                    UTType($0)?.conforms(to: .image) == true
                } ?? UTType.image.identifier
            let data: Data? = await withCheckedContinuation { continuation in
                provider.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) {
                    data, _ in
                    continuation.resume(returning: data)
                }
            }
            guard let data else {
                payload.rejections.append(.notAnImage)
                continue
            }
            switch ImageIngest.ingest(
                data: data, typeIdentifier: preferredUTI, filename: "dropped-image"
            ) {
            case .success(let attachment): payload.attachments.append(attachment)
            case .failure(let rejection): payload.rejections.append(rejection)
            }
        }
        return payload
    }
}
