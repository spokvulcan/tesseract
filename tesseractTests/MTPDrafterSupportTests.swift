import Foundation
import Testing

@testable import Tesseract_Agent

struct MTPDrafterSupportTests {

    // MARK: - Engagement policy

    /// The 27B profile: 24 attention heads, bf16 scores.
    private let scratchProfile = ModelIdentity.FullAttentionScratchProfile(
        attentionHeads: 24, bytesPerElement: 2)

    @Test func engagesOnGreedyTextOnlyColdPromptWithinBudget() {
        #expect(
            MTPDrafterSupport.shouldEngage(
                hasDrafter: true,
                temperature: 0,
                textOnlyIdentityKeySpace: true,
                promptTokens: 2048,
                scratchProfile: scratchProfile
            ))
    }

    @Test func refusesWithoutDrafter() {
        #expect(
            !MTPDrafterSupport.shouldEngage(
                hasDrafter: false,
                temperature: 0,
                textOnlyIdentityKeySpace: true,
                promptTokens: 2048,
                scratchProfile: scratchProfile
            ))
    }

    @Test func refusesNonGreedySampling() {
        // The Qwen drafters are greedy-only; engaging at temp > 0 would make
        // the vendor iterator passthrough after paying the drafter prefill.
        #expect(
            !MTPDrafterSupport.shouldEngage(
                hasDrafter: true,
                temperature: 0.6,
                textOnlyIdentityKeySpace: true,
                promptTokens: 2048,
                scratchProfile: scratchProfile
            ))
    }

    @Test func refusesImageBearingRequests() {
        #expect(
            !MTPDrafterSupport.shouldEngage(
                hasDrafter: true,
                temperature: 0,
                textOnlyIdentityKeySpace: false,
                promptTokens: 2048,
                scratchProfile: scratchProfile
            ))
    }

    @Test func refusesPromptsPastTheScratchBudget() {
        // 24 heads × L² × 2 bytes ≤ 4 GiB ⇒ L ≤ ~9459. One past the
        // boundary must refuse — the MTP prompt prefill is unchunked.
        let boundary = Int(
            (Double(MTPDrafterSupport.singleShotScratchBudgetBytes) / (24 * 2))
                .squareRoot())
        #expect(
            MTPDrafterSupport.shouldEngage(
                hasDrafter: true,
                temperature: 0,
                textOnlyIdentityKeySpace: true,
                promptTokens: boundary,
                scratchProfile: scratchProfile
            ))
        #expect(
            !MTPDrafterSupport.shouldEngage(
                hasDrafter: true,
                temperature: 0,
                textOnlyIdentityKeySpace: true,
                promptTokens: boundary + 1,
                scratchProfile: scratchProfile
            ))
    }

    @Test func refusesWithoutAScratchProfile() {
        // No profile means the single-shot prepare cannot be priced — never
        // engage unpriced.
        #expect(
            !MTPDrafterSupport.shouldEngage(
                hasDrafter: true,
                temperature: 0,
                textOnlyIdentityKeySpace: true,
                promptTokens: 128,
                scratchProfile: nil
            ))
    }

    // MARK: - Drafter pairing

    @Test func drafterPairingRefusesUnknownModelFamilies() {
        // A target outside the Qwen3.5 family has no drafter — pairing keys
        // on the loaded instance's class, and anything unrecognized must
        // yield nil (speculation off) rather than a drafter that would
        // fatalError on its first draft call.
        #expect(MTPDrafterSupport.drafterPairing(for: ToyLanguageModel(script: [0])) == nil)
    }

    // MARK: - Head detection

    private func makeModelDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtp-detect-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func writeIndex(_ dir: URL, keys: [String]) throws {
        let index: [String: Any] = [
            "metadata": ["total_size": 1],
            "weight_map": Dictionary(uniqueKeysWithValues: keys.map { ($0, "model.safetensors") }),
        ]
        let data = try JSONSerialization.data(withJSONObject: index)
        try data.write(to: dir.appendingPathComponent("model.safetensors.index.json"))
    }

    /// Minimal valid safetensors file: 8-byte little-endian header length,
    /// then the JSON header. No tensor data needed for header scanning.
    private func writeSafetensors(_ dir: URL, name: String, keys: [String]) throws {
        var header: [String: Any] = [:]
        for key in keys {
            header[key] = ["dtype": "BF16", "shape": [1], "data_offsets": [0, 2]]
        }
        let json = try JSONSerialization.data(withJSONObject: header)
        var blob = Data()
        var length = UInt64(json.count).littleEndian
        blob.append(Data(bytes: &length, count: 8))
        blob.append(json)
        try blob.write(to: dir.appendingPathComponent(name))
    }

    @Test func detectsHeadViaIndex() throws {
        let dir = try makeModelDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeIndex(dir, keys: ["model.embed_tokens.weight", "mtp.fc.weight"])
        #expect(MTPDrafterSupport.checkpointShipsMTPHead(directory: dir))
    }

    @Test func rejectsHeadlessIndex() throws {
        let dir = try makeModelDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeIndex(dir, keys: ["model.embed_tokens.weight", "lm_head.weight"])
        #expect(!MTPDrafterSupport.checkpointShipsMTPHead(directory: dir))
    }

    @Test func detectsHeadViaShardHeaderWhenNoIndex() throws {
        let dir = try makeModelDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeSafetensors(
            dir, name: "model.safetensors",
            keys: ["model.embed_tokens.weight", "mtp.norm.weight"])
        #expect(MTPDrafterSupport.checkpointShipsMTPHead(directory: dir))
    }

    @Test func rejectsHeadlessShardAndEmptyDirectory() throws {
        let dir = try makeModelDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(!MTPDrafterSupport.checkpointShipsMTPHead(directory: dir))
        try writeSafetensors(
            dir, name: "model.safetensors", keys: ["model.embed_tokens.weight"])
        #expect(!MTPDrafterSupport.checkpointShipsMTPHead(directory: dir))
    }

    // MARK: - Greedy preset

    @MainActor
    @Test func greedySpeculativePresetIsTempZeroWithPresenceBackstop() {
        let params = SamplingPreset.greedySpeculative.apply(to: .qwen35)
        #expect(params.temperature == 0)
        #expect(params.presencePenalty == 1.5)
        #expect(params.repetitionPenalty == nil)
        // Non-sampling fields ride through from the model-derived base.
        #expect(params.prefillStepSize == AgentGenerateParameters.qwen35.prefillStepSize)
    }
}
