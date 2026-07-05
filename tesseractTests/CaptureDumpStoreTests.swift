//
//  CaptureDumpStoreTests.swift
//  tesseractTests
//
//  Drives the **Capture Dump** store through its narrow seam (`save`/`deleteAll`)
//  against a temporary directory — the store's whole observable surface is the
//  files it leaves on disk. Ring bounds (count and total size, oldest evicted
//  first), condition tagging, and the non-fatality of write failures are the
//  contract (PRD #175); how the store names or writes files internally is not.
//  Disk work runs on a background chain, so each test `flush()`es before it
//  looks at the directory.
//

import AVFoundation
import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct CaptureDumpStoreTests {

    private func makeTempDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("capture-dump-tests-\(UUID().uuidString)")
    }

    private func makeCapture(
        seconds: Double = 0.1, sampleRate: Double = 48_000, voiceProcessed: Bool = false
    ) -> RawCapture {
        RawCapture(
            samples: [Float](repeating: 0.25, count: Int(sampleRate * seconds)),
            sampleRate: sampleRate,
            voiceProcessed: voiceProcessed
        )
    }

    private func wavFiles(in directory: URL) -> [URL] {
        let contents =
            (try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil)) ?? []
        return contents.filter { $0.pathExtension == "wav" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    // MARK: - Saving

    @Test func saveWritesAReadableWAVCarryingTheCaptureConditions() async throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = CaptureDumpStore(directory: directory)
        let capture = makeCapture(seconds: 0.1, sampleRate: 48_000, voiceProcessed: true)

        store.save(capture)
        await store.flush()

        let files = wavFiles(in: directory)
        #expect(files.count == 1)

        let name = try #require(files.first).lastPathComponent
        #expect(name.contains("48000Hz"))
        #expect(name.contains("vp-on"))

        let audioFile = try AVAudioFile(forReading: try #require(files.first))
        #expect(audioFile.fileFormat.sampleRate == 48_000)
        #expect(Int(audioFile.length) == capture.samples.count)
    }

    @Test func saveTagsAnUnprocessedCaptureAsVPOff() async throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = CaptureDumpStore(directory: directory)

        store.save(makeCapture(voiceProcessed: false))
        await store.flush()

        let name = try #require(wavFiles(in: directory).first).lastPathComponent
        #expect(name.contains("vp-off"))
    }

    // MARK: - Ring bounds

    @Test func exceedingTheRecordingCountEvictsOldestFirst() async {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = CaptureDumpStore(
            directory: directory,
            limits: .init(maxRecordings: 3, maxTotalBytes: .max)
        )

        var namesInSaveOrder: [String] = []
        for _ in 0..<4 {
            let before = Set(wavFiles(in: directory).map(\.lastPathComponent))
            store.save(makeCapture())
            await store.flush()
            let after = Set(wavFiles(in: directory).map(\.lastPathComponent))
            if let newName = after.subtracting(before).first {
                namesInSaveOrder.append(newName)
            }
        }

        let remaining = Set(wavFiles(in: directory).map(\.lastPathComponent))
        #expect(remaining.count == 3)
        #expect(remaining == Set(namesInSaveOrder.suffix(3)))
        #expect(!remaining.contains(namesInSaveOrder[0]))
    }

    @Test func exceedingTheTotalSizeBoundEvictsOldestFirst() async {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        // One 0.1 s 48 kHz float32 mono capture is ~19 KB of payload; a 50 KB
        // budget holds two recordings but not three.
        let store = CaptureDumpStore(
            directory: directory,
            limits: .init(maxRecordings: .max, maxTotalBytes: 50_000)
        )

        for _ in 0..<3 {
            store.save(makeCapture(seconds: 0.1))
        }
        await store.flush()

        let files = wavFiles(in: directory)
        #expect(files.count < 3)

        let totalBytes = files.reduce(0) { total, url in
            total + ((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        #expect(totalBytes <= 50_000)
    }

    // MARK: - Delete all

    @Test func deleteAllEmptiesTheStore() async {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = CaptureDumpStore(directory: directory)
        store.save(makeCapture())
        store.save(makeCapture())

        store.deleteAll()
        await store.flush()

        #expect(wavFiles(in: directory).isEmpty)
    }

    // MARK: - Failure is non-fatal

    @Test func saveIntoAnUncreatableDirectoryDoesNotCrashOrWrite() async throws {
        // A plain file where the store expects its directory: creation fails.
        let blocker = FileManager.default.temporaryDirectory
            .appendingPathComponent("capture-dump-blocker-\(UUID().uuidString)")
        try Data("not a directory".utf8).write(to: blocker)
        defer { try? FileManager.default.removeItem(at: blocker) }
        let store = CaptureDumpStore(directory: blocker)

        store.save(makeCapture())
        await store.flush()

        #expect(wavFiles(in: blocker).isEmpty)
    }
}
