//
//  ImageIngest.swift
//  tesseract
//

import Foundation
import ImageIO
import AppKit
import UniformTypeIdentifiers

/// The single shared, pure image-ingest core (PRD #112, slice #115). Every image
/// entry point — file picker, ⌘V paste, drag-and-drop — funnels its raw input
/// through here so all sources behave identically: the same supported-type set,
/// the same 10 MB cap, the same normalization, the same typed rejections.
///
/// Pure and `nonisolated`: it takes abstracted inputs (encoded bytes + a declared
/// type identifier, or an already-decoded `NSImage`) and returns an
/// `ImageAttachment` or a typed `Rejection`. The impure edges (NSPasteboard,
/// NSOpenPanel, NSItemProvider) live in the callers; only this verdict layer is
/// unit-tested, with fixtures.
nonisolated enum ImageIngest {

    /// Why an input was not turned into an attachment.
    enum Rejection: Error, Equatable {
        /// Declared type is not an image we accept (or couldn't be resolved).
        case unsupportedType(String)
        /// Bytes exceed the per-image cap.
        case oversize(bytes: Int)
        /// Declared an accepted type but the bytes don't decode as an image.
        case notAnImage
    }

    /// 10 MB per image — matches the composer's long-standing picker cap.
    static let maxBytes = 10 * 1024 * 1024

    /// The accepted image MIME types: PNG, JPEG, TIFF, GIF, WebP, HEIC/HEIF.
    static let supportedMIMETypes: Set<String> = [
        "image/png", "image/jpeg", "image/tiff",
        "image/gif", "image/webp", "image/heic", "image/heif",
    ]

    /// The same accepted set as `UTType`s, in a stable order — for the file
    /// picker's allowed types and the pasteboard probe. Derived from
    /// `supportedMIMETypes` so picker, paste, and ingest can never drift apart.
    static let supportedUTTypes: [UTType] =
        supportedMIMETypes
        .sorted()
        .compactMap { UTType(mimeType: $0) }

    /// Ingest encoded bytes carrying a declared type (a UTI like `public.png`, a
    /// MIME like `image/png`, or a bare extension). Keeps the original bytes for
    /// directly-usable types — no re-encode, so the on-disk Image Digest is the
    /// content the user actually pasted.
    static func ingest(
        data: Data, typeIdentifier: String, filename: String? = nil
    ) -> Result<ImageAttachment, Rejection> {
        guard let mime = normalizedMIME(forIdentifier: typeIdentifier) else {
            return .failure(.unsupportedType(typeIdentifier))
        }
        guard supportedMIMETypes.contains(mime) else {
            return .failure(.unsupportedType(mime))
        }
        guard data.count <= maxBytes else {
            return .failure(.oversize(bytes: data.count))
        }
        guard isDecodableImage(data) else {
            return .failure(.notAnImage)
        }
        return .success(ImageAttachment(data: data, mimeType: mime, filename: filename))
    }

    /// Ingest an already-decoded image (e.g. an `NSImage` copied in Preview or a
    /// browser, where the pasteboard has no file-backed bytes) by encoding to
    /// PNG. Subject to the same size cap after encoding.
    static func ingest(
        image: NSImage, filename: String? = nil
    ) -> Result<ImageAttachment, Rejection> {
        guard let png = pngData(from: image) else { return .failure(.notAnImage) }
        guard png.count <= maxBytes else { return .failure(.oversize(bytes: png.count)) }
        return .success(
            ImageAttachment(
                data: png, mimeType: "image/png", filename: filename ?? "pasted-image.png"
            ))
    }

    /// Cap a batch of candidate attachments to the room remaining under `limit`,
    /// given how many are already queued (slice #117). One testable rule for the
    /// composer's ~8-image cap, shared by the drop/paste/picker paths rather than
    /// duplicated at each call site.
    static func capBatch(
        _ attachments: [ImageAttachment], alreadyQueued: Int, limit: Int
    ) -> [ImageAttachment] {
        let room = max(0, limit - alreadyQueued)
        return Array(attachments.prefix(room))
    }

    // MARK: - Helpers

    /// Resolve a declared identifier to a canonical MIME type. Accepts a MIME
    /// (passed through), a UTI, or a bare filename extension.
    static func normalizedMIME(forIdentifier identifier: String) -> String? {
        if identifier.contains("/") { return identifier }  // already a MIME
        if let utType = UTType(identifier) ?? UTType(filenameExtension: identifier) {
            return utType.preferredMIMEType
        }
        return nil
    }

    /// True when the bytes decode as at least one image frame (ImageIO sniffs the
    /// real container, so a mislabeled non-image is rejected).
    static func isDecodableImage(_ data: Data) -> Bool {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return false }
        return CGImageSourceGetCount(source) > 0
    }

    /// PNG-encode a decoded image via a bitmap rep.
    static func pngData(from image: NSImage) -> Data? {
        guard let tiff = image.tiffRepresentation,
            let rep = NSBitmapImageRep(data: tiff),
            let png = rep.representation(using: .png, properties: [:])
        else { return nil }
        return png
    }
}
