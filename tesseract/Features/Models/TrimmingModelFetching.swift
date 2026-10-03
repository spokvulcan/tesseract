//
//  TrimmingModelFetching.swift
//  tesseract
//
//  Downloading only the part of a safetensors file a device runs (#515): the
//  phone's voice needs the codec's decoder, not the encoder stored after it in
//  the same file (225 MB that only voice cloning reads).
//

import Foundation

/// A **Model Fetching** adapter that can also fetch the start of a file.
protocol RangedModelFetching: ModelFetching {
    /// The first `length` bytes of `path`.
    func fetchPrefix(of path: String, from repo: String, length: Int) async throws -> Data

    /// The first `length` bytes of `path`, into `destination`.
    func fetchFile(at path: String, from repo: String, to destination: URL, length: Int)
        async throws
}

/// Keeping the tensors of a safetensors file whose names start with a prefix.
///
/// The file is a little-endian UInt64 header length, a JSON header naming each
/// tensor's byte range in the data that follows, then the data. When the kept
/// tensors' bytes come first, the file can be fetched as its header and that
/// range, and its header rewritten to name only them, padded with spaces to
/// its old length (the format allows it) so no offset moves.
nonisolated struct SafetensorsTrim: Equatable, Sendable {
    /// Bytes to fetch from the start of the file.
    let length: Int
    /// The header to write over the fetched one: the same length, the kept
    /// tensors only.
    let header: Data

    enum Failure: Error, Equatable {
        case notSafetensors
        /// Some tensor not kept sits among the kept ones' bytes.
        case keptTensorsNotFirst
    }

    /// Bytes to fetch to read the header: its length.
    static let lengthPrefix = 8

    /// The header's length, from the file's first eight bytes.
    static func headerLength(_ prefix: Data) throws -> Int {
        guard prefix.count >= lengthPrefix else { throw Failure.notSafetensors }
        let value = prefix.prefix(lengthPrefix).enumerated().reduce(UInt64(0)) {
            $0 | UInt64($1.element) << (8 * UInt64($1.offset))
        }
        guard value > 0, value < 100_000_000 else { throw Failure.notSafetensors }
        return Int(value)
    }

    /// The trim keeping `prefix`'s tensors, from the file's first
    /// `lengthPrefix + headerLength` bytes.
    init(start: Data, keeping prefix: String) throws {
        let headerLength = try Self.headerLength(start)
        guard start.count >= Self.lengthPrefix + headerLength,
            let object = try? JSONSerialization.jsonObject(
                with: start.subdata(in: Self.lengthPrefix..<(Self.lengthPrefix + headerLength))),
            let entries = object as? [String: Any]
        else { throw Failure.notSafetensors }

        var kept: [String: Any] = [:]
        var keptEnd = 0
        var droppedStart = Int.max
        for (name, entry) in entries {
            if name == "__metadata__" {
                kept[name] = entry
                continue
            }
            guard let tensor = entry as? [String: Any],
                let offsets = tensor["data_offsets"] as? [Int], offsets.count == 2
            else { throw Failure.notSafetensors }
            if name.hasPrefix(prefix) {
                kept[name] = entry
                keptEnd = max(keptEnd, offsets[1])
            } else {
                droppedStart = min(droppedStart, offsets[0])
            }
        }
        guard keptEnd <= droppedStart else { throw Failure.keptTensorsNotFirst }

        var header = try JSONSerialization.data(withJSONObject: kept, options: [.sortedKeys])
        guard header.count <= headerLength else { throw Failure.notSafetensors }
        header.append(Data(repeating: 0x20, count: headerLength - header.count))
        self.length = Self.lengthPrefix + headerLength + keptEnd
        self.header = header
    }

    /// Writes the kept header over the fetched file's.
    func rewrite(fileAt url: URL) throws {
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.seek(toOffset: UInt64(Self.lengthPrefix))
        try handle.write(contentsOf: header)
    }
}

/// Model Fetching that keeps only part of some files: for each listed path,
/// the tensors whose names start with its prefix. It lists those files at
/// their trimmed size, so the download manager's size checks hold, and
/// fetches them as their header and the kept bytes. A file whose kept
/// tensors aren't stored first is fetched whole.
final class TrimmingModelFetching<Base: RangedModelFetching>: ModelFetching {
    private let base: Base
    private let trims: [String: String]
    private var plans: [String: SafetensorsTrim] = [:]
    /// The bytes each repo's last listing came to, trims applied: a
    /// download's total, for its progress.
    private(set) var listedBytes: [String: Int] = [:]

    /// `trims`: file path to the tensor-name prefix to keep.
    init(base: Base, trims: [String: String]) {
        self.base = base
        self.trims = trims
    }

    func listFiles(in repo: String, recursive: Bool) async throws -> [RemoteModelFile] {
        var files = try await base.listFiles(in: repo, recursive: recursive)
        for (index, file) in files.enumerated() {
            guard let plan = try await plan(for: file.path, in: repo) else { continue }
            files[index] = RemoteModelFile(path: file.path, size: plan.length)
        }
        listedBytes[repo] = files.reduce(0) { $0 + ($1.size ?? 0) }
        return files
    }

    func fetchFile(at path: String, from repo: String, to destination: URL) async throws {
        guard let plan = try await plan(for: path, in: repo) else {
            try await base.fetchFile(at: path, from: repo, to: destination)
            return
        }
        try await base.fetchFile(at: path, from: repo, to: destination, length: plan.length)
        try plan.rewrite(fileAt: destination)
    }

    /// The trim for `path`, read from the remote file's header once; nil for
    /// a file kept whole.
    private func plan(for path: String, in repo: String) async throws -> SafetensorsTrim? {
        guard let prefix = trims[path] else { return nil }
        let key = "\(repo)/\(path)"
        if let plan = plans[key] { return plan }
        let lengthBytes = try await base.fetchPrefix(
            of: path, from: repo, length: SafetensorsTrim.lengthPrefix)
        let headerLength = try SafetensorsTrim.headerLength(lengthBytes)
        let start = try await base.fetchPrefix(
            of: path, from: repo, length: SafetensorsTrim.lengthPrefix + headerLength)
        guard let plan = try? SafetensorsTrim(start: start, keeping: prefix) else { return nil }
        plans[key] = plan
        return plan
    }
}
