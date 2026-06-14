//
//  ImagePreviewFileCacheTests.swift
//  tesseractTests
//
//  Pins the ImagePreviewFileCache port (PRD #112, slice #114): digest-keyed
//  temp files, write-once dedup, pre-warm reuse, and cleanup. Exercised through
//  two adapters (ADR-0001): an in-memory peer that records writes (the dedup
//  analogue of `InMemorySettingsStore.writes`) and the real temp-dir filesystem.
//

import Foundation
import Testing

@testable import Tesseract_Agent

/// In-memory `ImagePreviewFileSystem` peer recording writes for assertions.
nonisolated final class InMemoryImagePreviewFileSystem: ImagePreviewFileSystem {
    var storage: [URL: Data] = [:]
    private(set) var writes: [URL] = []
    private(set) var createdDirectories: [URL] = []

    func fileExists(at url: URL) -> Bool { storage[url] != nil }
    func createDirectory(at url: URL) throws { createdDirectories.append(url) }
    func write(_ data: Data, to url: URL) throws { storage[url] = data; writes.append(url) }
    func contentsOfDirectory(at url: URL) -> [URL] { Array(storage.keys) }
    func removeItem(at url: URL) throws { storage.removeValue(forKey: url) }
}

@MainActor
struct ImagePreviewFileCacheTests {

    private let root = URL(fileURLWithPath: "/previews", isDirectory: true)

    /// Real PNG bytes plus a unique tail → a distinct Image Digest per `byte`.
    private func makeAttachment(byte: UInt8) -> ImageAttachment {
        ImageAttachment(data: ImageTestFixtures.tinyPNGData + Data([byte]), mimeType: "image/png")
    }

    // MARK: - In-memory peer

    @Test
    func identicalBytesShareOneFileAndWriteOnce() throws {
        let fs = InMemoryImagePreviewFileSystem()
        let cache = ImagePreviewFileCache(root: root, fileSystem: fs)
        let original = makeAttachment(byte: 1)
        // Same bytes, different attachment id (e.g. the same image in two messages).
        let copy = ImageAttachment(data: original.data, mimeType: "image/png")

        let urlA = try cache.url(for: original)
        let urlB = try cache.url(for: copy)

        #expect(urlA == urlB)             // digest dedup → one file
        #expect(fs.writes.count == 1)     // write-once
    }

    @Test
    func distinctBytesGetDistinctFiles() throws {
        let fs = InMemoryImagePreviewFileSystem()
        let cache = ImagePreviewFileCache(root: root, fileSystem: fs)

        let urlA = try cache.url(for: makeAttachment(byte: 1))
        let urlB = try cache.url(for: makeAttachment(byte: 2))

        #expect(urlA != urlB)
        #expect(fs.writes.count == 2)
    }

    @Test
    func prewarmedImageOpensWithNoSecondWrite() throws {
        let fs = InMemoryImagePreviewFileSystem()
        let cache = ImagePreviewFileCache(root: root, fileSystem: fs)
        let image = makeAttachment(byte: 9)

        cache.prewarm(image)
        #expect(fs.writes.count == 1)

        let url = try cache.url(for: image)   // the later click
        #expect(fs.writes.count == 1)         // file already present — instant open
        #expect(fs.storage[url] != nil)
    }

    @Test
    func materializePreservesOrder() {
        let fs = InMemoryImagePreviewFileSystem()
        let cache = ImagePreviewFileCache(root: root, fileSystem: fs)
        let a = makeAttachment(byte: 1), b = makeAttachment(byte: 2), c = makeAttachment(byte: 3)

        let urls = cache.materialize([a, b, c])

        #expect(urls.count == 3)
        #expect(urls == [a, b, c].map { try! cache.url(for: $0) })
    }

    @Test
    func clearRemovesEveryFile() throws {
        let fs = InMemoryImagePreviewFileSystem()
        let cache = ImagePreviewFileCache(root: root, fileSystem: fs)
        _ = try cache.url(for: makeAttachment(byte: 1))
        _ = try cache.url(for: makeAttachment(byte: 2))
        #expect(fs.storage.count == 2)

        cache.clear()
        #expect(fs.storage.isEmpty)
    }

    // MARK: - Real temp-dir adapter

    @Test
    func realTempDirWritesDedupsAndCleansUp() throws {
        let realRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("qltest-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: realRoot) }

        let cache = ImagePreviewFileCache(root: realRoot)   // real DiskImagePreviewFileSystem
        let image = makeAttachment(byte: 7)

        let url = try cache.url(for: image)
        #expect(FileManager.default.fileExists(atPath: url.path))
        #expect((try? Data(contentsOf: url)) == image.data)

        // Same bytes → same file, no error on the second materialization.
        let again = try cache.url(for: ImageAttachment(data: image.data, mimeType: "image/png"))
        #expect(again == url)

        cache.clear()
        #expect(FileManager.default.fileExists(atPath: url.path) == false)
    }
}
