//
//  ModelFingerprintTests.swift
//  tesseractTests
//
//  Stability across repeated calls and sensitivity to content / mtime /
//  size changes are the two properties we care about — the fingerprint is
//  folded into `CachePartitionKey.modelFingerprint` to guarantee a weight
//  swap under the same `modelID` cannot surface stale SSD-resident
//  snapshots.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct ModelFingerprintTests {

    // MARK: - Helpers

    /// Create a scratch directory inside the process's temp dir.
    /// Cleaned up at the end of the test via `defer`.
    private func makeScratchDir() throws -> URL {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("ModelFingerprintTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }

    private func writeFile(_ data: Data, at url: URL, modificationDate: Date? = nil) throws {
        try data.write(to: url)
        if let modificationDate {
            try FileManager.default.setAttributes(
                [.modificationDate: modificationDate],
                ofItemAtPath: url.path
            )
        }
    }

    // MARK: - Happy path

    @Test
    func fingerprintIsStableForIdenticalDirectory() throws {
        let dir = try makeScratchDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let frozenDate = Date(timeIntervalSince1970: 1_700_000_000)
        try writeFile(Data("{\"model\":\"test\"}".utf8), at: dir.appendingPathComponent("config.json"), modificationDate: frozenDate)
        try writeFile(Data("{\"version\":\"1\"}".utf8), at: dir.appendingPathComponent("tokenizer.json"), modificationDate: frozenDate)
        try writeFile(Data(repeating: 0xAA, count: 1024), at: dir.appendingPathComponent("model.safetensors"), modificationDate: frozenDate)

        let fp1 = try ModelFingerprint.computeFingerprint(modelDir: dir)
        let fp2 = try ModelFingerprint.computeFingerprint(modelDir: dir)
        #expect(fp1 == fp2, "fingerprint must be deterministic across calls")
        #expect(fp1.count == 64, "hex SHA-256 digest is 64 characters")
    }

    @Test
    func fingerprintDifferentWhenConfigContentChanges() throws {
        let dir = try makeScratchDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let date = Date(timeIntervalSince1970: 1_700_000_000)
        let configURL = dir.appendingPathComponent("config.json")
        let tokenizerURL = dir.appendingPathComponent("tokenizer.json")
        let safetensorsURL = dir.appendingPathComponent("model.safetensors")

        try writeFile(Data("{\"model\":\"A\"}".utf8), at: configURL, modificationDate: date)
        try writeFile(Data("{\"version\":\"1\"}".utf8), at: tokenizerURL, modificationDate: date)
        try writeFile(Data(repeating: 0xAA, count: 512), at: safetensorsURL, modificationDate: date)

        let fpA = try ModelFingerprint.computeFingerprint(modelDir: dir)

        // Rewrite config.json with different content, preserve mtime on the
        // safetensors file so we isolate the config-sensitivity signal from
        // the mtime-sensitivity signal.
        try writeFile(Data("{\"model\":\"B\"}".utf8), at: configURL, modificationDate: date)
        try FileManager.default.setAttributes([.modificationDate: date], ofItemAtPath: safetensorsURL.path)

        let fpB = try ModelFingerprint.computeFingerprint(modelDir: dir)
        #expect(fpA != fpB, "config.json content change must change the fingerprint")
    }

    @Test
    func fingerprintDifferentWhenTokenizerContentChanges() throws {
        let dir = try makeScratchDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let date = Date(timeIntervalSince1970: 1_700_000_000)
        let tokenizerURL = dir.appendingPathComponent("tokenizer.json")
        let safetensorsURL = dir.appendingPathComponent("model.safetensors")

        try writeFile(Data("{\"model\":\"test\"}".utf8), at: dir.appendingPathComponent("config.json"), modificationDate: date)
        try writeFile(Data("{\"version\":\"1\"}".utf8), at: tokenizerURL, modificationDate: date)
        try writeFile(Data(repeating: 0xAA, count: 512), at: safetensorsURL, modificationDate: date)

        let fp1 = try ModelFingerprint.computeFingerprint(modelDir: dir)

        try writeFile(Data("{\"version\":\"2\"}".utf8), at: tokenizerURL, modificationDate: date)
        try FileManager.default.setAttributes([.modificationDate: date], ofItemAtPath: safetensorsURL.path)

        let fp2 = try ModelFingerprint.computeFingerprint(modelDir: dir)
        #expect(fp1 != fp2, "tokenizer.json content change must change the fingerprint")
    }

    @Test
    func fingerprintDifferentWhenSafetensorsMtimeChanges() throws {
        let dir = try makeScratchDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let configDate = Date(timeIntervalSince1970: 1_700_000_000)
        try writeFile(Data("{}".utf8), at: dir.appendingPathComponent("config.json"), modificationDate: configDate)
        try writeFile(Data("{}".utf8), at: dir.appendingPathComponent("tokenizer.json"), modificationDate: configDate)

        let weightURL = dir.appendingPathComponent("model.safetensors")
        let payload = Data(repeating: 0xAA, count: 1024)

        try writeFile(payload, at: weightURL, modificationDate: Date(timeIntervalSince1970: 1_700_000_100))
        let fp1 = try ModelFingerprint.computeFingerprint(modelDir: dir)

        // Same bytes, different mtime — should still flip the fingerprint so
        // a weight swap that happens to produce an identical byte stream but
        // a different file (different inode, different mtime) is detected.
        try writeFile(payload, at: weightURL, modificationDate: Date(timeIntervalSince1970: 1_700_000_200))
        let fp2 = try ModelFingerprint.computeFingerprint(modelDir: dir)

        #expect(fp1 != fp2, "mtime change on a safetensors file must change the fingerprint")
    }

    @Test
    func fingerprintDifferentWhenSafetensorsSizeChanges() throws {
        let dir = try makeScratchDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let date = Date(timeIntervalSince1970: 1_700_000_000)
        try writeFile(Data("{}".utf8), at: dir.appendingPathComponent("config.json"), modificationDate: date)
        try writeFile(Data("{}".utf8), at: dir.appendingPathComponent("tokenizer.json"), modificationDate: date)

        let weightURL = dir.appendingPathComponent("model.safetensors")
        try writeFile(Data(repeating: 0xAA, count: 512), at: weightURL, modificationDate: date)
        let fp1 = try ModelFingerprint.computeFingerprint(modelDir: dir)

        try writeFile(Data(repeating: 0xAA, count: 1024), at: weightURL, modificationDate: date)
        let fp2 = try ModelFingerprint.computeFingerprint(modelDir: dir)

        #expect(fp1 != fp2, "file size change must change the fingerprint")
    }

    @Test
    func fingerprintOrderIndependentAcrossSafetensorsEnumeration() throws {
        // Two directories with the same logical contents but created in
        // different orders must produce the same fingerprint — the fingerprint
        // sorts file entries by name, so filesystem enumeration order cannot
        // leak in.
        let dir1 = try makeScratchDir()
        let dir2 = try makeScratchDir()
        defer {
            try? FileManager.default.removeItem(at: dir1)
            try? FileManager.default.removeItem(at: dir2)
        }

        let date = Date(timeIntervalSince1970: 1_700_000_000)

        // Create files in order A, B in dir1
        try writeFile(Data("{}".utf8), at: dir1.appendingPathComponent("config.json"), modificationDate: date)
        try writeFile(Data("{}".utf8), at: dir1.appendingPathComponent("tokenizer.json"), modificationDate: date)
        try writeFile(Data(repeating: 0x11, count: 128), at: dir1.appendingPathComponent("a.safetensors"), modificationDate: date)
        try writeFile(Data(repeating: 0x22, count: 128), at: dir1.appendingPathComponent("b.safetensors"), modificationDate: date)

        // Create files in order B, A in dir2
        try writeFile(Data("{}".utf8), at: dir2.appendingPathComponent("config.json"), modificationDate: date)
        try writeFile(Data("{}".utf8), at: dir2.appendingPathComponent("tokenizer.json"), modificationDate: date)
        try writeFile(Data(repeating: 0x22, count: 128), at: dir2.appendingPathComponent("b.safetensors"), modificationDate: date)
        try writeFile(Data(repeating: 0x11, count: 128), at: dir2.appendingPathComponent("a.safetensors"), modificationDate: date)

        let fp1 = try ModelFingerprint.computeFingerprint(modelDir: dir1)
        let fp2 = try ModelFingerprint.computeFingerprint(modelDir: dir2)
        #expect(fp1 == fp2, "fingerprint must be independent of create-order / enumeration order")
    }

    @Test
    func fingerprintIgnoresNonSafetensorsFiles() throws {
        let dir = try makeScratchDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let date = Date(timeIntervalSince1970: 1_700_000_000)
        try writeFile(Data("{}".utf8), at: dir.appendingPathComponent("config.json"), modificationDate: date)
        try writeFile(Data("{}".utf8), at: dir.appendingPathComponent("tokenizer.json"), modificationDate: date)
        try writeFile(Data(repeating: 0x11, count: 128), at: dir.appendingPathComponent("model.safetensors"), modificationDate: date)

        let fp1 = try ModelFingerprint.computeFingerprint(modelDir: dir)

        // Drop a README.md and a spurious .txt into the dir. Neither should
        // affect the fingerprint — the spec is explicitly config + tokenizer
        // + *.safetensors metadata only.
        try writeFile(Data("# hi".utf8), at: dir.appendingPathComponent("README.md"), modificationDate: date)
        try writeFile(Data("ignore me".utf8), at: dir.appendingPathComponent("notes.txt"), modificationDate: date)

        let fp2 = try ModelFingerprint.computeFingerprint(modelDir: dir)
        #expect(fp1 == fp2, "non-safetensors / non-config files must not affect the fingerprint")
    }

    @Test
    func fingerprintThrowsForMissingDirectory() throws {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("does-not-exist-\(UUID().uuidString)", isDirectory: true)
        do {
            _ = try ModelFingerprint.computeFingerprint(modelDir: missing)
            Issue.record("expected ModelFingerprint.Error.missingDirectory")
        } catch ModelFingerprint.Error.missingDirectory {
            // expected
        } catch {
            Issue.record("unexpected error: \(error)")
        }
    }

    @Test
    func fingerprintSucceedsWithMissingOptionalFiles() throws {
        // A directory with only safetensors (no config/tokenizer) must still
        // produce a fingerprint rather than throwing — these optional files
        // contribute empty byte slices to the hash.
        let dir = try makeScratchDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let date = Date(timeIntervalSince1970: 1_700_000_000)
        try writeFile(Data(repeating: 0xFF, count: 256), at: dir.appendingPathComponent("model.safetensors"), modificationDate: date)

        let fp = try ModelFingerprint.computeFingerprint(modelDir: dir)
        #expect(fp.count == 64)
    }
}
