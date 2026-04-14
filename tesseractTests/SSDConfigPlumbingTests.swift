//
//  SSDConfigPlumbingTests.swift
//  tesseractTests
//
//  Cover the MainActor config-snapshot flow. `SettingsManager` is
//  `@MainActor` but the prefix-cache hot path inside `container.perform`
//  on `LLMActor` cannot await MainActor; the bridge is a value-type
//  snapshot (`SSDPrefixCacheConfig`) produced once at model load via
//  `SettingsManager.makeSSDPrefixCacheConfig()` and held as an
//  actor-isolated stored property on `LLMActor`. These tests cover the
//  factory + `AgentEngine` / `CachePartitionKey` plumbing without
//  requiring a real model load — the end-to-end path through a real
//  load/unload cycle is validated by the restart benchmark
//  (`PrefixCacheE2ERunner`) gated by `--prefix-cache-e2e`.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct SSDConfigPlumbingTests {

    // MARK: - Helpers

    /// Remove any UserDefaults state our tests might have left behind.
    /// Called before and after every test so global state stays clean.
    private func clearSSDDefaults() {
        let keys = [
            "prefixCacheSSDEnabled",
            "prefixCacheSSDBudgetBytes",
            "prefixCacheSSDDirectoryOverride",
        ]
        for key in keys {
            UserDefaults.standard.removeObject(forKey: key)
        }
    }

    // MARK: - Factory method

    @Test
    func factoryReturnsNilWhenDisabled() {
        clearSSDDefaults()
        defer { clearSSDDefaults() }

        let settings = SettingsManager()
        settings.prefixCacheSSDEnabled = false

        #expect(settings.makeSSDPrefixCacheConfig() == nil)
    }

    @Test
    func factoryReturnsPopulatedConfigWhenEnabled() {
        clearSSDDefaults()
        defer { clearSSDDefaults() }

        let settings = SettingsManager()
        settings.prefixCacheSSDEnabled = true

        let config = settings.makeSSDPrefixCacheConfig()
        let unwrapped = try! #require(config)
        #expect(unwrapped.enabled == true)
        #expect(unwrapped.budgetBytes == settings.prefixCacheSSDBudgetBytes)
        #expect(unwrapped.rootURL.path.hasSuffix("prefix-cache"))
    }

    @Test
    func defaultBudgetIsTwentyGiB() {
        clearSSDDefaults()
        defer { clearSSDDefaults() }

        // A fresh SettingsManager with cleared UserDefaults must pick up the
        // registered defaults (20 GiB, enabled, no override).
        let settings = SettingsManager()
        #expect(settings.prefixCacheSSDBudgetBytes == 20 * 1024 * 1024 * 1024)
        #expect(settings.prefixCacheSSDEnabled == true)
        #expect(settings.prefixCacheSSDDirectoryOverride == nil)
    }

    @Test
    func maxPendingBytesIsBoundedBy4GiBAndPhysicalMemoryOver16() {
        clearSSDDefaults()
        defer { clearSSDDefaults() }

        let settings = SettingsManager()
        settings.prefixCacheSSDEnabled = true

        let config = try! #require(settings.makeSSDPrefixCacheConfig())
        let physicalMemory = Int(clamping: ProcessInfo.processInfo.physicalMemory)
        let expected = min(4 * 1024 * 1024 * 1024, physicalMemory / 16)
        #expect(config.maxPendingBytes == expected)
        // Hard ceiling — never exceeds 4 GiB regardless of RAM.
        #expect(config.maxPendingBytes <= 4 * 1024 * 1024 * 1024)
        // And it scales with RAM — on any realistic dev machine (≥16 GiB),
        // maxPendingBytes must be positive.
        #expect(config.maxPendingBytes > 0)
    }

    @Test
    func directoryOverrideReplacesDefaultRootURL() {
        clearSSDDefaults()
        defer { clearSSDDefaults() }

        let settings = SettingsManager()
        settings.prefixCacheSSDEnabled = true
        let customPath = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssd-override-\(UUID().uuidString)", isDirectory: true)
            .path
        settings.prefixCacheSSDDirectoryOverride = customPath

        let config = try! #require(settings.makeSSDPrefixCacheConfig())
        #expect(config.rootURL.path == customPath)
    }

    @Test
    func directoryOverrideAcceptsFileURLStringForm() {
        clearSSDDefaults()
        defer { clearSSDDefaults() }

        let settings = SettingsManager()
        settings.prefixCacheSSDEnabled = true
        let customURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssd-override-urlform-\(UUID().uuidString)", isDirectory: true)
        settings.prefixCacheSSDDirectoryOverride = customURL.absoluteString

        let config = try! #require(settings.makeSSDPrefixCacheConfig())
        #expect(config.rootURL.path == customURL.path)
    }

    @Test
    func settingsValuesRoundTripViaUserDefaults() {
        clearSSDDefaults()
        defer { clearSSDDefaults() }

        // First manager: write non-default values.
        let first = SettingsManager()
        first.prefixCacheSSDEnabled = true
        first.prefixCacheSSDBudgetBytes = 12 * 1024 * 1024 * 1024
        first.prefixCacheSSDDirectoryOverride = "/tmp/roundtrip-test"

        // Second manager reads from UserDefaults — values must match.
        let second = SettingsManager()
        #expect(second.prefixCacheSSDEnabled == true)
        #expect(second.prefixCacheSSDBudgetBytes == 12 * 1024 * 1024 * 1024)
        #expect(second.prefixCacheSSDDirectoryOverride == "/tmp/roundtrip-test")
    }

    @Test
    func disablingPersistsAcrossInstances() {
        clearSSDDefaults()
        defer { clearSSDDefaults() }

        let first = SettingsManager()
        first.prefixCacheSSDEnabled = false
        let second = SettingsManager()
        #expect(second.prefixCacheSSDEnabled == false)
        #expect(second.makeSSDPrefixCacheConfig() == nil)
    }

    // MARK: - AgentEngine config resolution

    @Test
    func resolveReturnsNilForZeroArgInit() {
        clearSSDDefaults()
        defer { clearSSDDefaults() }

        let engine = AgentEngine()
        #expect(engine.resolveSSDConfig() == nil)
    }

    @Test
    func resolveReturnsSettingsDerivedConfig() {
        clearSSDDefaults()
        defer { clearSSDDefaults() }

        let settings = SettingsManager()
        settings.prefixCacheSSDEnabled = true
        settings.prefixCacheSSDBudgetBytes = 8 * 1024 * 1024 * 1024

        let engine = AgentEngine(settingsManager: settings)
        let resolved = try! #require(engine.resolveSSDConfig())
        #expect(resolved.enabled == true)
        #expect(resolved.budgetBytes == 8 * 1024 * 1024 * 1024)
    }

    @Test
    func resolveReturnsNilWhenSettingsDisabled() {
        clearSSDDefaults()
        defer { clearSSDDefaults() }

        let settings = SettingsManager()
        settings.prefixCacheSSDEnabled = false

        let engine = AgentEngine(settingsManager: settings)
        #expect(engine.resolveSSDConfig() == nil)
    }

    @Test
    func resolveReturnsExplicitConfigVerbatim() {
        let customRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssd-explicit-\(UUID().uuidString)", isDirectory: true)
        let config = SSDPrefixCacheConfig(
            enabled: true,
            rootURL: customRoot,
            budgetBytes: 4 * 1024 * 1024 * 1024,
            maxPendingBytes: 1 * 1024 * 1024 * 1024
        )
        let engine = AgentEngine(ssdConfig: config)
        #expect(engine.resolveSSDConfig() == config)
    }

    @Test
    func resolvePrefersExplicitOverSettings() {
        // When both sources are provided, explicit wins — documented
        // precedence rule. Any future refactor that reverses this must
        // trip this test.
        clearSSDDefaults()
        defer { clearSSDDefaults() }

        let settings = SettingsManager()
        settings.prefixCacheSSDEnabled = true
        settings.prefixCacheSSDBudgetBytes = 8 * 1024 * 1024 * 1024

        let explicit = SSDPrefixCacheConfig(
            enabled: true,
            rootURL: FileManager.default.temporaryDirectory
                .appendingPathComponent("ssd-prec-\(UUID().uuidString)", isDirectory: true),
            budgetBytes: 2 * 1024 * 1024 * 1024,
            maxPendingBytes: 512 * 1024 * 1024
        )

        let engine = AgentEngine(settingsManager: settings, ssdConfig: explicit)
        let resolved = try! #require(engine.resolveSSDConfig())
        #expect(resolved.budgetBytes == explicit.budgetBytes)
        #expect(resolved == explicit)
    }

    @Test
    func resolveReflectsLiveSettingsChangesBetweenCalls() {
        // The "snapshot refresh across loads" property: the resolver
        // reaches back into SettingsManager on every call, so two
        // consecutive calls with a mutation between them produce two
        // different configs. This is the no-real-model proxy for the
        // restart benchmark's cross-load refresh check.
        clearSSDDefaults()
        defer { clearSSDDefaults() }

        let settings = SettingsManager()
        settings.prefixCacheSSDEnabled = true
        let engine = AgentEngine(settingsManager: settings)

        settings.prefixCacheSSDBudgetBytes = 5 * 1024 * 1024 * 1024
        let first = try! #require(engine.resolveSSDConfig())

        settings.prefixCacheSSDBudgetBytes = 15 * 1024 * 1024 * 1024
        let second = try! #require(engine.resolveSSDConfig())

        #expect(first.budgetBytes == 5 * 1024 * 1024 * 1024)
        #expect(second.budgetBytes == 15 * 1024 * 1024 * 1024)
        #expect(first != second)

        // Flipping the toggle off must make subsequent resolves return nil.
        settings.prefixCacheSSDEnabled = false
        #expect(engine.resolveSSDConfig() == nil)
    }

    // MARK: - LLMActor boundary plumbing

    @Test
    func llmActorStartsWithNilConfigAndFingerprint() async {
        // Fresh engine → the inner LLMActor holds no load-time state.
        clearSSDDefaults()
        defer { clearSSDDefaults() }

        let settings = SettingsManager()
        let engine = AgentEngine(settingsManager: settings)
        let config = await engine.llmActor.currentSSDConfigForTesting
        let fingerprint = await engine.llmActor.currentModelFingerprintForTesting
        #expect(config == nil)
        #expect(fingerprint == nil)
    }

    /// Create a scratch "model" directory that exists but does not
    /// contain a loadable MLX model. The real `loadModel` path runs
    /// `ModelFingerprint.computeFingerprint` (succeeds on any readable
    /// directory) and `installLoadTimeSSDState` (succeeds unconditionally)
    /// before the container load trips the first MLX-specific failure.
    /// Caller is responsible for cleanup.
    private func makeFakeModelDirectory() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssd-fake-model-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    @Test
    func agentEngineLoadInstallsSSDConfigAndFingerprintViaRealPath() async throws {
        // Drive the full production chain:
        //   AgentEngine.loadModel → resolveSSDConfig
        //     → LLMActor.loadModel(..., ssdConfig:)
        //       → installLoadTimeSSDState
        // The container load trips on the fake directory and throws,
        // but by then the install has already run. Any refactor that
        // forgets to call `installLoadTimeSSDState` or that stops
        // passing the resolved config down the chain is caught here.
        clearSSDDefaults()
        defer { clearSSDDefaults() }

        let settings = SettingsManager()
        settings.prefixCacheSSDEnabled = true
        settings.prefixCacheSSDBudgetBytes = 7 * 1024 * 1024 * 1024
        let engine = AgentEngine(settingsManager: settings)
        let fakeDir = try makeFakeModelDirectory()
        defer { try? FileManager.default.removeItem(at: fakeDir) }

        do {
            try await engine.loadModel(from: fakeDir, visionMode: false)
            Issue.record("expected loadModel to throw for a non-model directory")
        } catch {
            // Expected — container load fails on a directory without weights.
        }

        let installedConfig = await engine.llmActor.currentSSDConfigForTesting
        let installedFingerprint = await engine.llmActor.currentModelFingerprintForTesting
        let unwrappedConfig = try #require(installedConfig)
        #expect(unwrappedConfig.enabled == true)
        #expect(unwrappedConfig.budgetBytes == 7 * 1024 * 1024 * 1024)
        #expect(installedFingerprint != nil)
        #expect(installedFingerprint?.count == 64)
    }

    @Test
    func agentEngineLoadInstallsNilSSDConfigWhenSettingsDisabled() async throws {
        // Disabled-SSD branch: the chain still installs the fingerprint
        // (the partition key needs it to guard against weight swaps) but
        // `ssdConfig` stays nil on the actor.
        clearSSDDefaults()
        defer { clearSSDDefaults() }

        let settings = SettingsManager()
        settings.prefixCacheSSDEnabled = false
        let engine = AgentEngine(settingsManager: settings)
        let fakeDir = try makeFakeModelDirectory()
        defer { try? FileManager.default.removeItem(at: fakeDir) }

        do {
            try await engine.loadModel(from: fakeDir, visionMode: false)
            Issue.record("expected loadModel to throw for a non-model directory")
        } catch {
            // Expected.
        }

        #expect(await engine.llmActor.currentSSDConfigForTesting == nil)
        let fingerprint = await engine.llmActor.currentModelFingerprintForTesting
        #expect(fingerprint != nil)
    }

    @Test
    func agentEngineUnloadClearsActorStateViaProductionPath() async throws {
        // Drive the full production unload chain:
        //   AgentEngine.unloadModel (sync) → Task { await llmActor.unloadModel() }
        //     → LLMActor.unloadModel
        //       → modelFingerprint/ssdConfig = nil
        // Seeds state via the real `loadModel` path so both halves use
        // production code. Any refactor that stops reaching the actor
        // clear path is caught here.
        clearSSDDefaults()
        defer { clearSSDDefaults() }

        let settings = SettingsManager()
        settings.prefixCacheSSDEnabled = true
        let engine = AgentEngine(settingsManager: settings)
        let fakeDir = try makeFakeModelDirectory()
        defer { try? FileManager.default.removeItem(at: fakeDir) }

        do {
            try await engine.loadModel(from: fakeDir, visionMode: false)
            Issue.record("expected loadModel to throw for a non-model directory")
        } catch {
            // Expected.
        }

        // Sanity: state was installed.
        #expect(await engine.llmActor.currentSSDConfigForTesting != nil)
        #expect(await engine.llmActor.currentModelFingerprintForTesting != nil)

        engine.unloadModel()
        await engine.awaitPendingUnloadForTesting()

        #expect(await engine.llmActor.currentSSDConfigForTesting == nil)
        #expect(await engine.llmActor.currentModelFingerprintForTesting == nil)
    }

    // MARK: - CachePartitionKey fingerprint folding

    @Test
    func partitionKeysWithDifferentFingerprintsAreDistinct() {
        let keyA = CachePartitionKey(
            modelID: "model",
            kvBits: 8,
            kvGroupSize: 64,
            sessionAffinity: "session",
            modelFingerprint: "aaaaaaaaaaaaaaaa"
        )
        let keyB = CachePartitionKey(
            modelID: "model",
            kvBits: 8,
            kvGroupSize: 64,
            sessionAffinity: "session",
            modelFingerprint: "bbbbbbbbbbbbbbbb"
        )
        #expect(keyA != keyB)
        #expect(keyA.hashValue != keyB.hashValue || keyA != keyB)
    }

    @Test
    func partitionKeyWithoutFingerprintMatchesLegacyCallSites() {
        // Legacy call sites (existing unit tests) don't pass modelFingerprint.
        // Both the 4-arg and 5-arg forms must produce equal keys when the
        // fingerprint is nil, otherwise every existing test would start
        // mis-matching its own inputs.
        let legacy = CachePartitionKey(
            modelID: "m",
            kvBits: nil,
            kvGroupSize: 64
        )
        let explicit = CachePartitionKey(
            modelID: "m",
            kvBits: nil,
            kvGroupSize: 64,
            sessionAffinity: nil,
            modelFingerprint: nil
        )
        #expect(legacy == explicit)
    }

    @Test
    func partitionKeyComparableIsStableWithFingerprint() {
        // Deterministic < ordering so partition iteration stays reproducible
        // for eviction tie-break scenarios. With fingerprint folded in, the
        // tuple now has 5 elements — exercise a representative ordering.
        let a = CachePartitionKey(
            modelID: "m", kvBits: 8, kvGroupSize: 64,
            sessionAffinity: "s", modelFingerprint: "aaa"
        )
        let b = CachePartitionKey(
            modelID: "m", kvBits: 8, kvGroupSize: 64,
            sessionAffinity: "s", modelFingerprint: "bbb"
        )
        #expect(a < b)
        #expect(!(b < a))
    }
}
