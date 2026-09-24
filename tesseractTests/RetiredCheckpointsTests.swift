//
//  RetiredCheckpointsTests.swift
//  tesseractTests
//
//  Pins the one-time removal of checkpoints an earlier catalog entry
//  downloaded (today the Voice Engine's bf16, which the engine never loaded).
//  Hermetic: a temp storage root and a throwaway defaults suite per test, no
//  manager (the same pattern as `ModelDownloadLifecycleTests`' harness).
//

import Foundation
import TesseractSpeech
import Testing

@testable import Tesseract_Agent

@MainActor
struct RetiredCheckpointsTests {

    private static let bf16 = TTSModelSpec.voiceDesign17B(.bf16).repo
    private static let listedRepos = Set(ModelDefinition.all.compactMap(\.repoID))

    @Test func theShippedVoiceCheckpointIsNeverRetired() {
        #expect(RetiredCheckpoints.repos.contains(Self.bf16))
        #expect(!RetiredCheckpoints.repos.contains(ModelDefinition.textToSpeechModelSpec.repo))
    }

    @Test func removesTheBf16DirectoryAndKeepsEverythingElse() throws {
        let world = try World()
        defer { world.tearDown() }
        let shipped = ModelDefinition.textToSpeechModelSpec.repo
        try world.place(repo: Self.bf16, path: "model.safetensors")
        try world.place(repo: Self.bf16, path: "speech_tokenizer/model.safetensors")
        try world.place(repo: shipped, path: "model.safetensors")
        try world.place(repo: "fixture/unrelated", path: "model.safetensors")

        let removed = RetiredCheckpoints.removeIfNeeded(
            from: world.root, listedRepos: Self.listedRepos, defaults: world.defaults)

        #expect(removed.map(\.path) == [world.directory(Self.bf16).path])
        #expect(!world.exists(Self.bf16))
        #expect(world.exists(shipped))
        #expect(world.exists("fixture/unrelated"))
        #expect(world.defaults.bool(forKey: RetiredCheckpoints.completionDefaultsKey))
    }

    /// Runs once: a bf16 copy put back later (a dev tool's reference run) is
    /// left alone.
    @Test func laterLaunchesLeaveTheStoreAlone() throws {
        let world = try World()
        defer { world.tearDown() }

        RetiredCheckpoints.removeIfNeeded(
            from: world.root, listedRepos: Self.listedRepos, defaults: world.defaults)
        try world.place(repo: Self.bf16, path: "model.safetensors")
        let removed = RetiredCheckpoints.removeIfNeeded(
            from: world.root, listedRepos: Self.listedRepos, defaults: world.defaults)

        #expect(removed.isEmpty)
        #expect(world.exists(Self.bf16))
    }

    /// If the catalog lists bf16 again, its directory is a live download.
    @Test func aRepoTheCatalogListsIsKept() throws {
        let world = try World()
        defer { world.tearDown() }
        try world.place(repo: Self.bf16, path: "model.safetensors")

        let removed = RetiredCheckpoints.removeIfNeeded(
            from: world.root, listedRepos: [Self.bf16], defaults: world.defaults)

        #expect(removed.isEmpty)
        #expect(world.exists(Self.bf16))
    }

    /// A failed removal isn't fatal and leaves the flag unset, so the next
    /// launch tries again.
    @Test func aFailedRemovalRetriesNextLaunch() throws {
        let world = try World()
        defer { world.tearDown() }
        try world.place(repo: Self.bf16, path: "model.safetensors")

        // A read-only store root can't have entries removed from it.
        try world.setRootWritable(false)
        let first = RetiredCheckpoints.removeIfNeeded(
            from: world.root, listedRepos: Self.listedRepos, defaults: world.defaults)
        #expect(first.isEmpty)
        #expect(world.exists(Self.bf16))
        #expect(!world.defaults.bool(forKey: RetiredCheckpoints.completionDefaultsKey))

        try world.setRootWritable(true)
        let second = RetiredCheckpoints.removeIfNeeded(
            from: world.root, listedRepos: Self.listedRepos, defaults: world.defaults)
        #expect(second.map(\.path) == [world.directory(Self.bf16).path])
        #expect(world.defaults.bool(forKey: RetiredCheckpoints.completionDefaultsKey))
    }

    // MARK: - Fixtures

    /// A temp model store plus a throwaway defaults suite.
    private final class World {
        let root: URL
        let defaults: UserDefaults
        private let suiteName = "RetiredCheckpointsTests-\(UUID().uuidString)"

        init() throws {
            root = FileManager.default.temporaryDirectory
                .appendingPathComponent("retired-checkpoints-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            defaults = try #require(UserDefaults(suiteName: suiteName))
        }

        func tearDown() {
            try? setRootWritable(true)
            try? FileManager.default.removeItem(at: root)
            defaults.removePersistentDomain(forName: suiteName)
        }

        func directory(_ repo: String) -> URL {
            root.modelDirectory(forRepo: repo)
        }

        func place(repo: String, path: String) throws {
            let target = directory(repo).appendingPathComponent(path)
            try FileManager.default.createDirectory(
                at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(count: 8).write(to: target)
        }

        func exists(_ repo: String) -> Bool {
            FileManager.default.fileExists(atPath: directory(repo).path)
        }

        func setRootWritable(_ writable: Bool) throws {
            try FileManager.default.setAttributes(
                [.posixPermissions: writable ? 0o755 : 0o555], ofItemAtPath: root.path)
        }
    }
}
