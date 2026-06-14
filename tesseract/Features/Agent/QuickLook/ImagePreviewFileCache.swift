//
//  ImagePreviewFileCache.swift
//  tesseract
//

import Foundation
import UniformTypeIdentifiers

/// The filesystem seam behind `ImagePreviewFileCache` (ADR-0001's two-adapter
/// rule): the production adapter writes real sandbox temp files; an in-memory
/// peer in tests records writes so dedup and cleanup are assertable hermetically.
nonisolated protocol ImagePreviewFileSystem: AnyObject {
    func fileExists(at url: URL) -> Bool
    func createDirectory(at url: URL) throws
    func write(_ data: Data, to url: URL) throws
    func contentsOfDirectory(at url: URL) -> [URL]
    func removeItem(at url: URL) throws
}

/// FileManager-backed adapter — the real sandbox temp filesystem.
nonisolated final class DiskImagePreviewFileSystem: ImagePreviewFileSystem {
    func fileExists(at url: URL) -> Bool {
        FileManager.default.fileExists(atPath: url.path)
    }
    func createDirectory(at url: URL) throws {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }
    func write(_ data: Data, to url: URL) throws {
        try data.write(to: url, options: .atomic)
    }
    func contentsOfDirectory(at url: URL) -> [URL] {
        (try? FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)) ?? []
    }
    func removeItem(at url: URL) throws {
        try FileManager.default.removeItem(at: url)
    }
}

/// The **ImagePreviewFileCache** port (PRD #112, slice #114): materializes an
/// `ImageAttachment`'s bytes to a sandbox temp file Quick Look can open by URL,
/// keyed by the image's **Image Digest** (SHA-256 over the exact bytes, reused
/// from `ImageDigest`). Write-once: identical bytes across messages share one
/// file (digest dedup); distinct images get distinct files. Pre-warm a preview
/// set ahead of a click so opening is near-instant; cleared on conversation
/// reset (the OS reclaims the sandbox temp dir otherwise).
@MainActor
final class ImagePreviewFileCache {

    private let root: URL
    private let fs: ImagePreviewFileSystem
    private var directoryEnsured = false

    /// Default sandbox temp root, mirroring the app's other temp-dir users.
    static func defaultRoot() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("quicklook-previews", isDirectory: true)
    }

    init(
        root: URL = ImagePreviewFileCache.defaultRoot(),
        fileSystem: ImagePreviewFileSystem = DiskImagePreviewFileSystem()
    ) {
        self.root = root
        self.fs = fileSystem
    }

    /// The temp-file URL for one attachment, writing its bytes on first request
    /// only. Idempotent: a second call for the same bytes finds the file present
    /// and writes nothing (digest dedup).
    @discardableResult
    func url(for attachment: ImageAttachment) throws -> URL {
        let digest = ImageDigest(imageBytes: attachment.data).hexString
        let url = root
            .appendingPathComponent(digest, isDirectory: false)
            .appendingPathExtension(Self.fileExtension(forMIME: attachment.mimeType))
        if !fs.fileExists(at: url) {
            ensureDirectory()
            try fs.write(attachment.data, to: url)
        }
        return url
    }

    /// Materialize a whole preview set in order, returning the file URLs. Bytes
    /// that fail to write are dropped rather than aborting the set.
    @discardableResult
    func materialize(_ attachments: [ImageAttachment]) -> [URL] {
        attachments.compactMap { try? url(for: $0) }
    }

    /// Pre-warm a single image (fire-and-forget from the transcript decode path)
    /// so its file already exists by the time it's clicked.
    func prewarm(_ attachment: ImageAttachment) {
        _ = try? url(for: attachment)
    }

    /// Remove every materialized preview file. Called on conversation reset so
    /// the sandbox temp dir doesn't accumulate stale images mid-session.
    func clear() {
        for url in fs.contentsOfDirectory(at: root) {
            try? fs.removeItem(at: url)
        }
    }

    private func ensureDirectory() {
        guard !directoryEnsured else { return }
        try? fs.createDirectory(at: root)
        directoryEnsured = true
    }

    /// Map a MIME type to a file extension so Quick Look picks the right viewer.
    /// Falls back to `img` for unknown types (Quick Look still sniffs content).
    static func fileExtension(forMIME mime: String) -> String {
        UTType(mimeType: mime)?.preferredFilenameExtension ?? "img"
    }
}
