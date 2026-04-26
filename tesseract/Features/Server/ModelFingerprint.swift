//
//  ModelFingerprint.swift
//  tesseract
//
//  Stable fingerprint for a locally-downloaded model directory. Used as the
//  weight-identity term of `CachePartitionKey` so a weight swap under the
//  same `modelID` cannot surface stale persisted prefix-cache snapshots.
//
//  Hash input: SHA-256 over config.json bytes + tokenizer.json bytes +
//  sorted list of (filename, size, mtime) for every *.safetensors in the
//  directory. Full weight-byte hashing is intentionally skipped — it would
//  add ~100 ms to the load path for a scenario (weight swap under stable
//  modelID) that is manual-ops-only.
//

import CryptoKit
import Foundation

enum ModelFingerprint {

    enum Error: LocalizedError {
        case missingDirectory(URL)
        case unableToEnumerate(URL, underlying: Swift.Error)
        case unreadableFile(URL, underlying: Swift.Error)

        var errorDescription: String? {
            switch self {
            case .missingDirectory(let url):
                return "Model directory does not exist: \(url.path)"
            case .unableToEnumerate(let url, let underlying):
                return "Failed to enumerate model directory \(url.path): \(underlying.localizedDescription)"
            case .unreadableFile(let url, let underlying):
                return "Failed to read model file \(url.path): \(underlying.localizedDescription)"
            }
        }
    }

    /// Compute a lowercase-hex SHA-256 fingerprint for `modelDir`.
    ///
    /// `nonisolated` so `LLMActor.loadModel` can call it without a MainActor
    /// hop — the default actor inference would otherwise place static
    /// members of unmarked enums on MainActor.
    ///
    /// Missing optional files (e.g. a model shipped without a
    /// `tokenizer.json`) contribute an empty byte slice rather than
    /// failing the whole computation.
    nonisolated static func computeFingerprint(modelDir: URL) throws -> String {
        let fm = FileManager.default
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: modelDir.path, isDirectory: &isDirectory),
              isDirectory.boolValue
        else {
            throw Error.missingDirectory(modelDir)
        }

        var hasher = SHA256()

        // 1. config.json — hash raw bytes if present; empty if absent.
        let configURL = modelDir.appendingPathComponent("config.json", isDirectory: false)
        hasher.update(data: try readIfPresent(configURL))

        // Fixed separator avoids collision between (config="AB", tokenizer="") and
        // (config="A", tokenizer="B"). Matches the convention used by hashable
        // multi-field canonicalization elsewhere in the codebase.
        hasher.update(data: Data([0x00]))

        // 2. tokenizer.json — hash raw bytes if present; empty if absent.
        let tokenizerURL = modelDir.appendingPathComponent("tokenizer.json", isDirectory: false)
        hasher.update(data: try readIfPresent(tokenizerURL))
        hasher.update(data: Data([0x00]))

        // 3. Sorted list of (filename, size, mtime) for every *.safetensors.
        //    Sorting by filename gives a deterministic order independent of
        //    filesystem enumeration order. Each tuple is encoded as
        //    "<name>\0<size>\0<mtimeNs>\n" before feeding into SHA-256.
        let tensorEntries = try collectSafetensorsEntries(in: modelDir)
        for entry in tensorEntries {
            hasher.update(data: Data(entry.name.utf8))
            hasher.update(data: Data([0x00]))
            hasher.update(data: withUnsafeBytes(of: entry.size.littleEndian) { Data($0) })
            hasher.update(data: Data([0x00]))
            hasher.update(data: withUnsafeBytes(of: entry.mtimeNanoseconds.littleEndian) { Data($0) })
            hasher.update(data: Data([0x0a]))
        }

        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Content-only fingerprint: SHA-256 over `config.json` + `tokenizer.json`
    /// + sorted `(filename, size, full byte content)` of every `*.safetensors`.
    /// Unlike `computeFingerprint`, this hash is stable across machines for the
    /// same HuggingFace checkpoint — it does not fold in the local filesystem
    /// mtime, and it reads weight bytes rather than trusting their mtime.
    ///
    /// Used for shipped-artifact lookup (e.g. TriAttention calibration stats
    /// keyed by content-identity of the checkpoint). The local SSD prefix
    /// cache continues to use the cheaper `computeFingerprint`.
    ///
    /// Cost: streams safetensors content through SHA-256 in 8 MiB chunks. On
    /// Apple Silicon expect ~500 MB/s → a few seconds for 4B, tens of seconds
    /// for 27B. Only call this when the result is actually needed (e.g.
    /// TriAttention requested + model is PARO + not vision mode).
    nonisolated static func computeContentFingerprint(modelDir: URL) throws -> String {
        let fm = FileManager.default
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: modelDir.path, isDirectory: &isDirectory),
              isDirectory.boolValue
        else {
            throw Error.missingDirectory(modelDir)
        }

        var hasher = SHA256()

        let configURL = modelDir.appendingPathComponent("config.json", isDirectory: false)
        hasher.update(data: try readIfPresent(configURL))
        hasher.update(data: Data([0x00]))

        let tokenizerURL = modelDir.appendingPathComponent("tokenizer.json", isDirectory: false)
        hasher.update(data: try readIfPresent(tokenizerURL))
        hasher.update(data: Data([0x00]))

        let tensorEntries = try collectSafetensorsEntries(in: modelDir)
        for entry in tensorEntries {
            hasher.update(data: Data(entry.name.utf8))
            hasher.update(data: Data([0x00]))
            hasher.update(data: withUnsafeBytes(of: entry.size.littleEndian) { Data($0) })
            hasher.update(data: Data([0x00]))
            let entryURL = modelDir.appendingPathComponent(entry.name, isDirectory: false)
            try hashFileContents(url: entryURL, into: &hasher)
            hasher.update(data: Data([0x0a]))
        }

        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - Private

    private struct SafetensorsEntry {
        let name: String
        let size: Int64
        let mtimeNanoseconds: Int64
    }

    /// Stream `url`'s bytes through `hasher` in 8 MiB chunks so we can hash
    /// multi-GiB weight files without loading them into memory. Wrapped in
    /// `autoreleasepool` so the per-chunk `Data` objects drop between reads.
    nonisolated private static func hashFileContents(url: URL, into hasher: inout SHA256) throws {
        let handle: FileHandle
        do {
            handle = try FileHandle(forReadingFrom: url)
        } catch {
            throw Error.unreadableFile(url, underlying: error)
        }
        defer { try? handle.close() }

        let chunkSize = 8 * 1024 * 1024
        while true {
            let chunk: Data?
            do {
                chunk = try autoreleasepool { try handle.read(upToCount: chunkSize) }
            } catch {
                throw Error.unreadableFile(url, underlying: error)
            }
            guard let chunk, !chunk.isEmpty else { break }
            hasher.update(data: chunk)
        }
    }

    nonisolated private static func readIfPresent(_ url: URL) throws -> Data {
        do {
            return try Data(contentsOf: url, options: [.mappedIfSafe])
        } catch let error as CocoaError
            where error.code == .fileReadNoSuchFile || error.code == .fileNoSuchFile
        {
            return Data()
        } catch {
            throw Error.unreadableFile(url, underlying: error)
        }
    }

    /// Lowercase-hex SHA-256 over the bytes of a single file. Used by the
    /// DFlash bind path to mix the draft's `config.json` identity into the
    /// effective `modelFingerprint` so a draft swap forces a partition
    /// flush. Returns `nil` if the file cannot be read; caller decides
    /// whether to treat that as fatal.
    nonisolated static func sha256OfFile(_ url: URL) -> String? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    nonisolated private static func collectSafetensorsEntries(in modelDir: URL) throws -> [SafetensorsEntry] {
        let fm = FileManager.default
        let contents: [URL]
        do {
            contents = try fm.contentsOfDirectory(
                at: modelDir,
                includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey],
                options: [.skipsHiddenFiles]
            )
        } catch {
            throw Error.unableToEnumerate(modelDir, underlying: error)
        }

        let safetensors = contents.filter { $0.pathExtension.lowercased() == "safetensors" }

        var entries: [SafetensorsEntry] = []
        entries.reserveCapacity(safetensors.count)
        for url in safetensors {
            let values: URLResourceValues
            do {
                values = try url.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
            } catch {
                throw Error.unreadableFile(url, underlying: error)
            }
            let size = Int64(values.fileSize ?? 0)
            let mtimeNs: Int64
            if let date = values.contentModificationDate {
                mtimeNs = Int64(date.timeIntervalSince1970 * 1_000_000_000)
            } else {
                mtimeNs = 0
            }
            entries.append(SafetensorsEntry(
                name: url.lastPathComponent,
                size: size,
                mtimeNanoseconds: mtimeNs
            ))
        }

        entries.sort { $0.name < $1.name }
        return entries
    }
}
