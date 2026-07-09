//
//  OnboardingTourControllerTests.swift
//  tesseractTests
//
//  The Onboarding Tour's one seam (PRD #171, ADR-0021): chapter navigation,
//  close-=-skip completion semantics, the hardware-aware model pick, the
//  speech→voice→agent download queue, and ready-state derivation — asserted
//  through the controller's public surface plus the download manager's
//  observable statuses. Hermetic: `InMemorySettingsStore` for settings,
//  `InMemoryModelFetching` + temp storage root for downloads. Prior art:
//  `ModelDownloadLifecycleTests` (LifecycleHarness), `SettingsManagerTests`.
//

import Combine
import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct OnboardingTourControllerTests {

    // MARK: - Chapters

    @Test func chaptersAdvanceInOrderAndClampAtBothEnds() throws {
        let world = try TourWorld()
        defer { world.tearDown() }
        let controller = world.makeController()

        #expect(controller.chapter == .welcome)
        #expect(!controller.canGoBack)

        var visited: [OnboardingTourController.Chapter] = [controller.chapter]
        while !controller.isLastChapter {
            controller.advance()
            visited.append(controller.chapter)
        }
        #expect(visited == [.welcome, .agent, .dictation, .voice, .server, .ready])

        controller.advance()
        #expect(controller.chapter == .ready, "advance clamps at the last chapter")

        controller.go(to: .server)
        #expect(controller.chapter == .server)

        controller.goBack()
        #expect(controller.chapter == .voice)

        controller.go(to: .welcome)
        controller.goBack()
        #expect(controller.chapter == .welcome, "goBack clamps at the first chapter")
    }

    // MARK: - Model pick

    @Test func freshMachinePicksByPhysicalRAM() throws {
        let world = try TourWorld()
        defer { world.tearDown() }

        let small = world.makeController(physicalMemoryGiB: 16)
        #expect(small.chosenAgentModelID == "qwen3.5-4b-paro")

        let big = world.makeController(physicalMemoryGiB: 64)
        #expect(big.chosenAgentModelID == "qwen3.6-35b-a3b-paro")
    }

    @Test func downloadedSelectedAgentModelOutranksTheRAMPick() throws {
        let world = try TourWorld()
        defer { world.tearDown() }

        // The user already runs the 9B: selected in settings and on disk.
        world.settings.selectedAgentModelID = "qwen3.5-9b-paro"
        try world.placeModelOnDisk("qwen3.5-9b-paro")

        // Even on a machine whose RAM pick would say 27B.
        let controller = world.makeController(physicalMemoryGiB: 128)
        #expect(controller.chosenAgentModelID == "qwen3.5-9b-paro")
    }

    // MARK: - Setup queue

    @Test func setupDownloadsSpeechThenVoiceThenAgent() async throws {
        let world = try TourWorld()
        defer { world.tearDown() }
        let controller = world.makeController(physicalMemoryGiB: 16)

        controller.beginSetupIfNeeded()
        try await world.waitUntil { controller.isSetupComplete }

        #expect(
            world.fetching.listedRepos == [
                "fixture/stt", "fixture/voice", "fixture/agent-4b",
            ],
            "strictly sequential: speech first, voice second, agent last")
        #expect(controller.speechModelReady)
        #expect(controller.voiceModelReady)
        #expect(controller.agentModelReady)
    }

    @Test func beginSetupWritesTheChosenAgentModelToSettings() async throws {
        let world = try TourWorld()
        defer { world.tearDown() }
        let controller = world.makeController(physicalMemoryGiB: 64)

        controller.beginSetupIfNeeded()
        #expect(world.settings.selectedAgentModelID == "qwen3.6-35b-a3b-paro")
        try await world.waitUntil { controller.isSetupComplete }
    }

    @Test func beginSetupIsIdempotent() async throws {
        let world = try TourWorld()
        defer { world.tearDown() }
        let controller = world.makeController(physicalMemoryGiB: 16)

        controller.beginSetupIfNeeded()
        try await world.waitUntil { controller.isSetupComplete }
        let listingsAfterFirst = world.fetching.listedRepos.count

        controller.beginSetupIfNeeded()
        try await Task.sleep(for: .milliseconds(20))
        #expect(world.fetching.listedRepos.count == listingsAfterFirst)
    }

    @Test func modelsAlreadyOnDiskAreSkippedByTheQueue() async throws {
        let world = try TourWorld()
        defer { world.tearDown() }
        try world.placeModelOnDisk("stt-fixture")

        let controller = world.makeController(physicalMemoryGiB: 16)
        controller.beginSetupIfNeeded()
        try await world.waitUntil { controller.isSetupComplete }

        #expect(world.fetching.listedRepos == ["fixture/voice", "fixture/agent-4b"])
    }

    @Test func aFailedDownloadAdvancesTheQueueInsteadOfStallingIt() async throws {
        let world = try TourWorld()
        defer { world.tearDown() }
        world.fetching.errorsByRepo["fixture/stt"] =
            ModelFetchingError.invalidRepository("fixture/stt")

        let controller = world.makeController(physicalMemoryGiB: 16)
        controller.beginSetupIfNeeded()

        try await world.waitUntil { controller.voiceModelReady && controller.agentModelReady }
        #expect(!controller.speechModelReady)
        #expect(!controller.isSetupComplete)
    }

    @Test func readyStateDerivesFromAllThreeTargets() async throws {
        let world = try TourWorld()
        defer { world.tearDown() }
        let controller = world.makeController(physicalMemoryGiB: 16)

        #expect(!controller.isSetupComplete)
        #expect(controller.setupProgress == 0)
        #expect(!controller.speechModelReady)

        controller.beginSetupIfNeeded()
        try await world.waitUntil { controller.isSetupComplete }
        #expect(controller.setupProgress == 1)
    }

    // MARK: - Completion semantics

    @Test func finishingMarksOnboardingCompleted() throws {
        let world = try TourWorld()
        defer { world.tearDown() }
        let controller = world.makeController()

        #expect(!world.settings.hasCompletedOnboarding)
        controller.complete()
        #expect(world.settings.hasCompletedOnboarding)
        #expect(controller.didComplete)
    }

    @Test func skippingPreservesInFlightDownloads() async throws {
        let world = try TourWorld()
        defer { world.tearDown() }
        world.fetching.holdListings = true

        let controller = world.makeController(physicalMemoryGiB: 16)
        controller.beginSetupIfNeeded()
        try await world.waitUntil { world.fetching.listedRepos.count == 1 }
        #expect(world.manager.status(for: "stt-fixture") == .downloading(progress: 0))

        // Close-=-skip: completion never cancels what's in flight.
        controller.complete()
        #expect(world.settings.hasCompletedOnboarding)
        try await Task.sleep(for: .milliseconds(20))
        #expect(world.manager.status(for: "stt-fixture") == .downloading(progress: 0))

        world.manager.cancelDownload(modelID: "stt-fixture")
        _ = try await world.waitUntil {
            world.manager.status(for: "stt-fixture") == .notDownloaded
        }
    }

    // MARK: - Changing the pick

    @Test func selectingAnotherAgentModelBeforeSetupUsesTheNewChoice() async throws {
        let world = try TourWorld()
        defer { world.tearDown() }
        let controller = world.makeController(physicalMemoryGiB: 16)

        controller.selectAgentModel("qwen3.5-9b-paro")
        #expect(controller.chosenAgentModelID == "qwen3.5-9b-paro")

        controller.beginSetupIfNeeded()
        #expect(world.settings.selectedAgentModelID == "qwen3.5-9b-paro")

        try await world.waitUntil { controller.isSetupComplete }
        #expect(world.fetching.listedRepos.contains("fixture/agent-9b"))
        #expect(!world.fetching.listedRepos.contains("fixture/agent-4b"))
    }

    @Test func selectingAnotherAgentModelMidFlightCancelsTheOldDownload() async throws {
        let world = try TourWorld()
        defer { world.tearDown() }
        // Speech and voice already on disk, so the agent slot is the live one.
        try world.placeModelOnDisk("stt-fixture")
        try world.placeModelOnDisk("tts-fixture")

        world.fetching.holdListings = true
        let controller = world.makeController(physicalMemoryGiB: 16)
        controller.beginSetupIfNeeded()
        try await world.waitUntil {
            world.manager.status(for: "qwen3.5-4b-paro") == .downloading(progress: 0)
        }

        world.fetching.holdListings = false
        controller.selectAgentModel("qwen3.5-9b-paro")
        #expect(world.settings.selectedAgentModelID == "qwen3.5-9b-paro")

        try await world.waitUntil { controller.agentModelReady }
        #expect(world.manager.isDownloaded("qwen3.5-9b-paro"))
        #expect(world.manager.status(for: "qwen3.5-4b-paro") == .notDownloaded)
        #expect(controller.isSetupComplete)
    }
}

// MARK: - World

/// Per-test world for the Tour Controller: a temp storage root, the in-memory
/// fetching peer, a real `ModelDownloadManager` over fixture definitions, and
/// a `SettingsManager` on the in-memory store. Fixture agent ids reuse the
/// real catalogue ids so the RAM pick's answers resolve inside the fixtures.
@MainActor
private final class TourWorld {
    let storageRoot: URL
    let fetching: InMemoryModelFetching
    let manager: ModelDownloadManager
    let settings: SettingsManager

    private static let gib: UInt64 = 1 << 30

    static let definitions: [ModelDefinition] = [
        fixture("stt-fixture", repo: "fixture/stt", category: .speechToText),
        fixture("tts-fixture", repo: "fixture/voice", category: .textToSpeech),
        fixture("qwen3.5-4b-paro", repo: "fixture/agent-4b", category: .agent),
        fixture("qwen3.5-9b-paro", repo: "fixture/agent-9b", category: .agent),
        fixture("qwen3.6-35b-a3b-paro", repo: "fixture/agent-35b", category: .agent),
    ]

    init() throws {
        storageRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("onboarding-tour-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: storageRoot, withIntermediateDirectories: true)
        fetching = InMemoryModelFetching(
            storageRoot: storageRoot,
            repos: [
                "fixture/stt": [.init(path: "model.safetensors", size: 8)],
                "fixture/voice": [.init(path: "model.safetensors", size: 8)],
                "fixture/agent-4b": [.init(path: "model.safetensors", size: 8)],
                "fixture/agent-9b": [.init(path: "model.safetensors", size: 8)],
                "fixture/agent-35b": [.init(path: "model.safetensors", size: 8)],
            ]
        )
        manager = ModelDownloadManager(
            fetching: fetching, storageRoot: storageRoot, definitions: Self.definitions)
        settings = SettingsManager(store: InMemorySettingsStore())
    }

    func makeController(physicalMemoryGiB: UInt64 = 16) -> OnboardingTourController {
        OnboardingTourController(
            settings: settings,
            downloadManager: manager,
            speechToTextModelID: "stt-fixture",
            voiceModelID: "tts-fixture",
            physicalMemoryBytes: physicalMemoryGiB * Self.gib
        )
    }

    func placeModelOnDisk(_ id: String) throws {
        guard let definition = Self.definitions.first(where: { $0.id == id }),
            let repo = definition.repoID
        else { fatalError("unknown fixture id \(id)") }
        let target = storageRoot.modelDirectory(forRepo: repo)
            .appendingPathComponent("model.safetensors")
        try FileManager.default.createDirectory(
            at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(count: 8).write(to: target)
        manager.refreshAllStatuses()
    }

    func tearDown() {
        try? FileManager.default.removeItem(at: storageRoot)
    }

    @discardableResult
    func waitUntil(
        timeout: Duration = .seconds(10), _ condition: () -> Bool
    ) async throws -> Bool {
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: timeout)
        while clock.now < deadline {
            if condition() { return true }
            try await Task.sleep(for: .milliseconds(2))
        }
        Issue.record("condition not met within \(timeout)")
        return false
    }

    private static func fixture(
        _ id: String, repo: String, category: ModelCategory
    ) -> ModelDefinition {
        ModelDefinition(
            id: id, displayName: "Display \(id)", description: "",
            category: category,
            source: .huggingFace(repo: repo, requiredExtension: "safetensors"),
            sizeDescription: "~1 GB", dependencies: [])
    }
}
