//
//  TrimmingModelFetchingTests.swift
//  tesseractTests
//
//  The phone's voice downloads only the codec's decoder from a file that also
//  holds its encoder (#515): a safetensors file whose kept tensors come first
//  is fetched as its header and their bytes, and its header rewritten in
//  place. Through the download manager with the in-memory fetching peer.
//

import Foundation
import MLX
import Testing

@testable import Tesseract_Agent

@MainActor
struct TrimmingModelFetchingTests {

    /// A safetensors file of float32 tensors, stored in this order.
    static func safetensors(_ tensors: [(String, [Float])]) throws -> Data {
        var header: [String: Any] = [:]
        var data = Data()
        for (name, values) in tensors {
            let start = data.count
            for value in values {
                withUnsafeBytes(of: value.bitPattern.littleEndian) { data.append(contentsOf: $0) }
            }
            header[name] = [
                "dtype": "F32", "shape": [values.count], "data_offsets": [start, data.count],
            ]
        }
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        json.append(Data(repeating: 0x20, count: (8 - json.count % 8) % 8))
        var file = Data()
        withUnsafeBytes(of: UInt64(json.count).littleEndian) { file.append(contentsOf: $0) }
        return file + json + data
    }

    static let codec: [(String, [Float])] = [
        ("decoder.conv.weight", [1, 2, 3, 4]),
        ("decoder.norm.weight", [5, 6]),
        ("encoder.conv.weight", [7, 8, 9, 10, 11, 12]),
    ]

    /// The trimmed file is the header and the decoder's bytes; MLX loads it,
    /// decoder only, values intact.
    @Test func aTrimKeepsThePrefixedTensors() throws {
        let file = try Self.safetensors(Self.codec)
        let trim = try SafetensorsTrim(start: file, keeping: "decoder.")
        let headerLength = try SafetensorsTrim.headerLength(file)
        #expect(trim.length == 8 + headerLength + 6 * 4)
        #expect(trim.header.count == headerLength)

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("trim-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try file.prefix(trim.length).write(to: url)
        try trim.rewrite(fileAt: url)

        let arrays = try MLX.loadArrays(url: url)
        #expect(Set(arrays.keys) == ["decoder.conv.weight", "decoder.norm.weight"])
        #expect(arrays["decoder.conv.weight"]?.asArray(Float.self) == [1, 2, 3, 4])
        #expect(arrays["decoder.norm.weight"]?.asArray(Float.self) == [5, 6])
    }

    /// Kept tensors stored after one that isn't can't be fetched as a prefix.
    @Test func keptTensorsStoredLaterAreNotTrimmed() throws {
        let file = try Self.safetensors([Self.codec[2], Self.codec[0]])
        #expect(throws: SafetensorsTrim.Failure.keptTensorsNotFirst) {
            try SafetensorsTrim(start: file, keeping: "decoder.")
        }
    }

    /// The manager lists the file at its trimmed size, fetches only those
    /// bytes, counts the entry downloaded, and a second pass finds nothing
    /// to repair. Other files come whole.
    @Test func theDownloadManagerFetchesOnlyTheKeptBytes() async throws {
        let codec = try Self.safetensors(Self.codec)
        let repo = "fixture/voice"
        let base = InMemoryModelFetching(repos: [
            repo: [
                .init(path: "config.json", size: 2, contents: Data("{}".utf8)),
                .init(
                    path: "speech_tokenizer/model.safetensors", size: codec.count, contents: codec),
            ]
        ])
        let fetching = TrimmingModelFetching(
            base: base, trims: ["speech_tokenizer/model.safetensors": "decoder."])
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("trim-lifecycle-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let model = ModelDefinition(
            id: "voice", displayName: "Voice", description: "", category: .textToSpeech,
            source: .huggingFace(repo: repo, requiredExtension: "safetensors"),
            sizeDescription: "", dependencies: [])
        let manager = ModelDownloadManager(
            fetching: fetching, storageRoot: root, definitions: [model])

        manager.download(modelID: "voice")
        try await Self.waitUntilSettled(manager, "voice")
        let trimmed = try SafetensorsTrim(start: codec, keeping: "decoder.")
        guard case .downloaded(let size) = manager.status(for: "voice") else {
            Issue.record("not downloaded: \(manager.status(for: "voice"))")
            return
        }
        #expect(size == Int64(2 + trimmed.length))
        #expect(
            base.fetchedFiles.contains(
                "\(repo)/speech_tokenizer/model.safetensors (first \(trimmed.length) bytes)"))
        #expect(base.fetchedFiles.contains("\(repo)/config.json"))

        let fetched = base.fetchedFiles.count
        manager.verifyAndRepair(modelID: "voice")
        try await Self.waitUntilSettled(manager, "voice")
        #expect(base.fetchedFiles.count == fetched, "nothing to repair")
    }

    static func waitUntilSettled(_ manager: ModelDownloadManager, _ id: String) async throws {
        for _ in 0..<200 {
            switch manager.status(for: id) {
            case .downloading, .verifying: try await Task.sleep(for: .milliseconds(10))
            default: return
            }
        }
        Issue.record("\(id) never settled")
    }
}
