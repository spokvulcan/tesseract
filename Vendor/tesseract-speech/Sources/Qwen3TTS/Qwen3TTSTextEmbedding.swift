import Darwin
import Foundation
@preconcurrency import MLX
import MLXNN

/// The talker's text embedding table: 151,936 rows of the talker's text
/// width, 622 MB of bf16 in every shipped checkpoint, a third of the talker.
///
/// A prompt uses a few hundred rows, once, so an unquantized table is read
/// from the checkpoint file row by row (`pread`, no mapping) and never held
/// in memory; the page cache keeps the rows a voice uses warm. A quantized
/// table, which the reader can't slice, stays an in-memory embedding.
package final class Qwen3TTSTextEmbedding: @unchecked Sendable {
    package let dimensions: Int
    package let count: Int
    private let storage: Storage

    private enum Storage {
        case file(descriptor: Int32, offset: UInt64, rowBytes: Int, dtype: DType)
        case memory(Embedding)
    }

    deinit {
        if case .file(let descriptor, _, _, _) = storage { close(descriptor) }
    }

    /// Embeds `ids` as `[1, ids.count, dimensions]`.
    package func callAsFunction(_ ids: [Int32]) throws -> MLXArray {
        switch storage {
        case .memory(let embedding):
            return embedding(MLXArray(ids).reshaped(1, -1))
        case .file(let descriptor, let offset, let rowBytes, let dtype):
            var bytes = Data(count: ids.count * rowBytes)
            try bytes.withUnsafeMutableBytes { buffer in
                for (i, id) in ids.enumerated() {
                    guard id >= 0, Int(id) < count else {
                        throw AudioGenerationError.invalidInput(
                            "Text token \(id) is outside the embedding table (\(count) rows).")
                    }
                    let destination = buffer.baseAddress! + i * rowBytes
                    let position = off_t(offset + UInt64(id) * UInt64(rowBytes))
                    var done = 0
                    while done < rowBytes {
                        let n = pread(descriptor, destination + done, rowBytes - done, position + off_t(done))
                        guard n > 0 else {
                            throw AudioGenerationError.modelNotInitialized(
                                "Reading text embedding row \(id) failed (errno \(errno)).")
                        }
                        done += n
                    }
                }
            }
            return MLXArray(bytes, [1, ids.count, dimensions], dtype: dtype)
        }
    }

    /// An in-memory table (a quantized checkpoint, or tests).
    package init(embedding: Embedding, count: Int, dimensions: Int) {
        self.count = count
        self.dimensions = dimensions
        storage = .memory(embedding)
    }

    /// The table stored as `key` in one of `directory`'s safetensors files,
    /// read on demand. Nil when no file stores it as a plain bf16, fp16 or
    /// fp32 matrix.
    package init?(key: String, directory: URL) {
        for file in (try? Qwen3TTSWeights.safetensorsFiles(in: directory)) ?? [] {
            guard let entry = Self.header(of: file)?[key] else { continue }
            let dtype: DType
            switch entry.dtype {
            case "BF16": dtype = .bfloat16
            case "F16": dtype = .float16
            case "F32": dtype = .float32
            default: return nil
            }
            guard entry.shape.count == 2,
                entry.end - entry.begin == UInt64(entry.shape[0] * entry.shape[1] * dtype.size)
            else { return nil }
            let descriptor = open(file.path, O_RDONLY)
            guard descriptor >= 0 else { return nil }
            count = entry.shape[0]
            dimensions = entry.shape[1]
            storage = .file(
                descriptor: descriptor, offset: entry.dataStart + entry.begin,
                rowBytes: dimensions * dtype.size, dtype: dtype)
            return
        }
        return nil
    }

    private struct Entry {
        var dtype: String
        var shape: [Int]
        var begin: UInt64
        var end: UInt64
        var dataStart: UInt64
    }

    /// A safetensors header: an 8-byte little-endian length, then that many
    /// bytes of JSON naming each tensor's dtype, shape and byte range.
    private static func header(of file: URL) -> [String: Entry]? {
        guard let handle = try? FileHandle(forReadingFrom: file) else { return nil }
        defer { try? handle.close() }
        guard let lengthBytes = try? handle.read(upToCount: 8), lengthBytes.count == 8 else {
            return nil
        }
        let length = lengthBytes.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
        guard length < 100_000_000, let json = try? handle.read(upToCount: Int(length)),
            let object = try? JSONSerialization.jsonObject(with: json) as? [String: Any]
        else { return nil }
        var entries: [String: Entry] = [:]
        for (name, value) in object {
            guard let info = value as? [String: Any], let dtype = info["dtype"] as? String,
                let shape = info["shape"] as? [Int], let range = info["data_offsets"] as? [Int],
                range.count == 2
            else { continue }
            entries[name] = Entry(
                dtype: dtype, shape: shape, begin: UInt64(range[0]), end: UInt64(range[1]),
                dataStart: 8 + length)
        }
        return entries
    }
}
