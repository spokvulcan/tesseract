import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

struct DFlash2SupportTests {

    // MARK: - Server cold-path engagement policy

    /// `shouldEngage` with the engaging happy path as defaults, so each test
    /// names only the axis it varies.
    private func engages(
        hasDrafter: Bool = true,
        textOnlyIdentityKeySpace: Bool = true,
        predictedLeafStoreMode: HTTPLeafStoreMode = .directLeaf
    ) -> Bool {
        DFlash2Support.shouldEngage(
            hasDrafter: hasDrafter,
            textOnlyIdentityKeySpace: textOnlyIdentityKeySpace,
            predictedLeafStoreMode: predictedLeafStoreMode
        )
    }

    @Test func engagesOnTextOnlyColdPromptWithDirectLeaf() {
        #expect(engages())
    }

    @Test func refusesWithoutDrafter() {
        #expect(!engages(hasDrafter: false))
    }

    @Test func refusesImageBearingRequests() {
        #expect(!engages(textOnlyIdentityKeySpace: false))
    }

    @Test func refusesNonDirectLeafStoreModes() {
        // The DFlash2 iterator runs its own chunked capture prefill, so the
        // mid-prefill boundary snapshots other leaf modes synthesize from
        // never exist — same constraint as MTP (ADR-0056).
        #expect(!engages(predictedLeafStoreMode: .canonicalUserLeaf))
        #expect(!engages(predictedLeafStoreMode: .directToolLeaf))
    }

    // MARK: - Agent raw-arm engagement policy

    @Test func rawArmEngagesOnTextOnlyInput() {
        let input = LMInput(text: .init(tokens: MLXArray([Int32(1), 2, 3])))
        #expect(DFlash2Support.shouldEngageRawArm(hasDrafter: true, input: input))
    }

    @Test func rawArmRefusesWithoutDrafter() {
        let input = LMInput(text: .init(tokens: MLXArray([Int32(1), 2, 3])))
        #expect(!DFlash2Support.shouldEngageRawArm(hasDrafter: false, input: input))
    }

    // MARK: - Draft folder detection

    private func makeStorageRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("dflash2-detect-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    @Test func draftDirectoryRequiresConfigAndWeights() throws {
        let root = try makeStorageRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let draftDir = root.appendingPathComponent(DFlash2Support.draftCacheSubdirectory)

        // Missing entirely.
        #expect(
            DFlash2Support.draftDirectory(storageRoot: root) == nil,
            "empty storage root must yield nil, got \(String(describing: DFlash2Support.draftDirectory(storageRoot: root)))"
        )

        // Config only — no weights.
        try FileManager.default.createDirectory(at: draftDir, withIntermediateDirectories: true)
        try "{}".write(
            to: draftDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        #expect(
            DFlash2Support.draftDirectory(storageRoot: root) == nil,
            "config without safetensors must yield nil, got \(String(describing: DFlash2Support.draftDirectory(storageRoot: root)))"
        )

        // Weights too — detected. (Compare standardized paths: URL equality
        // is literal about trailing slashes.)
        try Data([0, 1]).write(to: draftDir.appendingPathComponent("model.safetensors"))
        let detected = DFlash2Support.draftDirectory(storageRoot: root)
        #expect(
            detected?.standardizedFileURL.path == draftDir.standardizedFileURL.path,
            "config + safetensors must detect: got \(String(describing: detected)) vs \(draftDir)"
        )
    }

    // MARK: - Model definition wiring

    @MainActor
    @Test func draftIsDownloadableDependencyOfQwen38() {
        let draft = ModelDefinition.withID(DFlash2Support.draftModelID)
        #expect(draft != nil)
        #expect(draft?.category == .draft)
        #expect(draft?.repoID == "incoai/Qwen3.8-27B-DFlash2")
        let target = ModelDefinition.withID("qwen3.8-27b")
        #expect(target?.dependencies.contains(DFlash2Support.draftModelID) == true)
    }

    @MainActor
    @Test func draftCacheSubdirectoryMatchesDefinitionRule() {
        // `DFlash2Support.draftCacheSubdirectory` duplicates the definition's
        // `cacheSubdirectory` because the latter is MainActor-isolated; this
        // pins them together.
        let draft = ModelDefinition.withID(DFlash2Support.draftModelID)
        #expect(draft?.cacheSubdirectory == DFlash2Support.draftCacheSubdirectory)
    }
}
