import Foundation
import Testing

@testable import Tesseract_Agent

struct BoundedCacheParityTests {
    nonisolated enum Exit: CaseIterable, Sendable { case success, failure, cancellation }
    private struct InjectedFailure: Error {}

    @Test(arguments: Exit.allCases)
    func scratchStoreDrainsAndRemovesPayloadsOnEveryExit(exit: Exit) async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("parity-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        var observedStore: SSDSnapshotStore?
        do {
            try await BoundedCacheParity.withScratchStore(diskRoot: root) { store in
                observedStore = store
                store.registerPartition(
                    .init(
                        modelID: "fixture", modelFingerprint: String(repeating: "a", count: 64),
                        kvBits: nil, kvGroupSize: 64, createdAt: 100,
                        schemaVersion: SnapshotManifestSchema.currentVersion), digest: "fixture")
                let payload = SnapshotPayload(
                    tokenOffset: 1, checkpointType: .leaf,
                    layers: [
                        .init(
                            className: "KVCache",
                            state: [
                                .init(
                                    data: Data(repeating: 7, count: 256), dtype: "bfloat16",
                                    shape: [1, 128])
                            ],
                            metaState: [], offset: 1)
                    ])
                let descriptor = PersistedSnapshotDescriptor(
                    snapshotID: "base", partitionDigest: "fixture", pathFromRoot: [1],
                    tokenOffset: 1, checkpointType: "leaf", bytes: payload.totalBytes,
                    createdAt: 100, lastAccessAt: 100,
                    fileRelativePath: PersistedSnapshotDescriptor.relativeFilePath(
                        snapshotID: "base", partitionDigest: "fixture"),
                    schemaVersion: SnapshotManifestSchema.currentVersion)
                guard case .accepted = store.tryEnqueue(payload: payload, descriptor: descriptor)
                else {
                    Issue.record("fixture payload was rejected")
                    return
                }
                switch exit {
                case .success: break
                case .failure: throw InjectedFailure()
                case .cancellation: throw CancellationError()
                }
            }
            #expect(exit == .success)
        } catch is InjectedFailure {
            #expect(exit == .failure)
        } catch is CancellationError {
            #expect(exit == .cancellation)
        }
        let store = try #require(observedStore)
        #expect(store.pendingWriteCount() == 0)
        #expect(store.residency().descriptor(id: "base") != nil)
        #expect(!FileManager.default.fileExists(atPath: root.path))
    }
}
