import Foundation
import Testing

@testable import Tesseract_Agent

struct PlaceholderContainerEncodingTests {
    private func fixture(_ blob: Data, suffix: Int? = nil) -> (
        SnapshotPayload, PersistedSnapshotDescriptor
    ) {
        let payload = SnapshotPayload(
            tokenOffset: 4, checkpointType: .leaf,
            layers: [
                .init(
                    className: "KVCacheSimple",
                    state: [.init(data: blob, dtype: "uint8", shape: [blob.count])],
                    metaState: ["4"], offset: 4, suffixBaseOffset: suffix)
            ])
        let descriptor = PersistedSnapshotDescriptor(
            snapshotID: "fixed", partitionDigest: "abcd1234", pathFromRoot: [1, 2, 3, 4],
            tokenOffset: 4, checkpointType: "leaf", bytes: blob.count,
            segmentBaseOffset: suffix ?? 0, createdAt: 100, lastAccessAt: 101,
            fileRelativePath: "fixed.safetensors", schemaVersion: 9)
        return (payload, descriptor)
    }

    @Test func chunksBorrowPayloadStorageWithinTheWriteBound() throws {
        let blob = Data(repeating: 0xAB, count: 9_009)
        let (payload, descriptor) = fixture(blob)
        let encoding = try PlaceholderContainerEncoding(payload: payload, descriptor: descriptor)
        var consumed = 0
        blob.withUnsafeBytes { (source: UnsafeRawBufferPointer) in
            encoding.withChunks(maximumBytes: 4_096) { chunk in
                #expect(chunk.count <= 4_096)
                if consumed >= encoding.headerByteCount {
                    let payloadOffset = consumed - encoding.headerByteCount
                    #expect(chunk.baseAddress == source.baseAddress?.advanced(by: payloadOffset))
                }
                consumed += chunk.count
            }
        }
        #expect(consumed == encoding.byteCount)
    }

    @Test(arguments: [nil, 2] as [Int?])
    func layoutMatchesExistingContainer(suffix: Int?) throws {
        let (payload, descriptor) = fixture(Data([1, 2, 3, 4]), suffix: suffix)
        let name = suffix == nil ? "full.bin" : "suffix.bin"
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/PlaceholderContainer/" + name)
        let expected = try Data(contentsOf: url)
        let encoding = try PlaceholderContainerEncoding(payload: payload, descriptor: descriptor)
        var actual = Data()
        encoding.withChunks(maximumBytes: 7) { actual.append(contentsOf: $0) }
        #expect(actual == expected)
    }

    @Test func aWriteErrorStopsBeforeTheNextChunk() throws {
        struct WriteFailed: Error {}
        let (payload, descriptor) = fixture(Data(repeating: 0xAB, count: 9_009))
        let encoding = try PlaceholderContainerEncoding(payload: payload, descriptor: descriptor)
        var calls = 0
        #expect(throws: WriteFailed.self) {
            try encoding.withChunks(maximumBytes: 4_096) { _ in
                calls += 1
                throw WriteFailed()
            }
        }
        #expect(calls == 1)
    }
}
