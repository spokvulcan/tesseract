import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import Tokenizers  // referenced by the #huggingFaceTokenizerLoader macro expansion

/// App-side surface for loading the MTP speculative-decoding drafter that
/// rides inside a Qwen3.5-family checkpoint (the `mtp.*` weight prefix).
///
/// Two jobs:
/// 1. **Detection** — does this checkpoint ship the MTP head at all? Most
///    quantized redistributions strip it, so absence is the common case and
///    must be cheap (headers only, no weight reads).
/// 2. **Loading** — instantiate the drafter *matching the target the app
///    actually loaded*. The vendor's global registrations pick the drafter by
///    config shape (`vision_config` present ⇒ VLM drafter), but config shape
///    does not determine the loaded class: the app's non-vision path
///    force-loads the MLXLLM text target even from VLM-shaped checkpoints
///    (see `LLMActor.loadModel`), and the generic loader itself falls back
///    VLM → LLM when the VLM factory throws (legacy-layout checkpoints like
///    the Qwen3.8-27B community quant, which ships `language_model.*` /
///    `vision_tower.*` tensor names, load as the text model even in vision
///    mode). Each drafter `fatalError`s on the other family's target, so
///    selection keys on ``drafterPairing(for:)`` — the class of the model
///    instance that actually loaded — never on config shape or intent.
nonisolated enum MTPDrafterSupport {

    /// Which drafter family pairs with a loaded target model.
    enum DrafterPairing: String, Sendable {
        case vlm
        case text
    }

    /// The drafter that pairs with the model instance the app actually
    /// loaded, or `nil` when the family has no MTP drafter (speculation
    /// stays off). The MoE variants subclass the dense classes in both
    /// modules, so two checks per family cover all four targets.
    static func drafterPairing(for model: any LanguageModel) -> DrafterPairing? {
        if model is MLXVLM.Qwen35 { return .vlm }
        if model is MLXLLM.Qwen35Model || model is Qwen35TextModel { return .text }
        return nil
    }

    /// Total tokens per speculation round (1 bonus + `blockSize - 1` drafted).
    /// The vendor default. Note the Qwen3.5 drafters clamp to their
    /// `maximumBlockSize = 2` (one draft per round — the hybrid target's
    /// recurrent state rewinds at most one position in place), so today's
    /// effective ceiling is a 2× cut in target calls; the 4 stays for any
    /// future drafter that can draft deeper.
    static let blockSize = 4

    /// Ceiling on the single-shot full-attention score matrix
    /// (`[heads, L, L]`) the MTP prepare may allocate. The vendor MTP prompt
    /// prefill is unchunked by design (the target must expose one hidden row
    /// per prompt token for the drafter's shifted-prompt cache), so prompt
    /// length is the engagement knob: past this bound the request stays on
    /// the ordinary chunked path. 4 GiB ≈ a 9K-token prompt on the 27B's
    /// 24-head bf16 profile.
    static let singleShotScratchBudgetBytes: UInt64 = 4 << 30

    // MARK: - Engagement policy

    /// The cold-path engagement decision (pure, unit-tested): speculate only
    /// when a drafter is loaded, sampling is greedy (the Qwen drafters are
    /// greedy-only — the vendor iterator would silently passthrough
    /// otherwise, wasting the drafter prefill), the request is text-only
    /// (identity key space), and the whole-prompt single-shot prepare fits
    /// the scratch budget.
    static func shouldEngage(
        hasDrafter: Bool,
        temperature: Float,
        textOnlyIdentityKeySpace: Bool,
        promptTokens: Int,
        scratchProfile: ModelIdentity.FullAttentionScratchProfile?
    ) -> Bool {
        guard hasDrafter, temperature == 0, textOnlyIdentityKeySpace else { return false }
        guard let scratchProfile,
            let scratchBytes = scratchProfile.scoreMatrixBytes(sequenceLength: promptTokens)
        else { return false }
        return scratchBytes <= singleShotScratchBudgetBytes
    }

    // MARK: - Detection

    /// `true` when the checkpoint directory ships MTP head weights (any
    /// tensor under the `mtp.` prefix). Reads the safetensors index when
    /// present, otherwise scans each shard's JSON header — never the weights.
    static func checkpointShipsMTPHead(directory: URL) -> Bool {
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        if let data = try? Data(contentsOf: indexURL),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let weightMap = json["weight_map"] as? [String: Any]
        {
            return weightMap.keys.contains { $0.hasPrefix("mtp.") }
        }

        guard
            let files = try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil)
        else { return false }
        for file in files where file.pathExtension == "safetensors" {
            if safetensorsHeaderKeys(of: file).contains(where: { $0.hasPrefix("mtp.") }) {
                return true
            }
        }
        return false
    }

    /// Tensor names from a safetensors file header (8-byte little-endian
    /// header length, then that many bytes of JSON). Empty on any read or
    /// parse failure — detection degrades to "no head", never throws.
    private static func safetensorsHeaderKeys(of file: URL) -> [String] {
        guard let handle = try? FileHandle(forReadingFrom: file) else { return [] }
        defer { try? handle.close() }
        guard let lengthData = try? handle.read(upToCount: 8), lengthData.count == 8
        else { return [] }
        let headerLength = lengthData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
        // Corrupt-length guard: headers are JSON, tens of MB at the extreme.
        guard headerLength > 0, headerLength < 256 * 1024 * 1024,
            let headerData = try? handle.read(upToCount: Int(headerLength)),
            let json = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any]
        else { return [] }
        return Array(json.keys)
    }

    // MARK: - Loading

    /// Load the drafter from the same checkpoint directory as the target.
    ///
    /// The registry is built per-call with exactly the creators that pair
    /// with the loaded target family, mirroring the vendor registration
    /// creators (`Qwen35TextMTPRegistration` / `Qwen35VLMMTPRegistration`)
    /// minus their config-shape predicates — the pairing decision was already
    /// made by ``drafterPairing(for:)`` on the loaded target instance.
    static func loadDrafter(
        directory: URL,
        pairing: DrafterPairing
    ) async throws -> MTPDrafterContext {
        let typeRegistry = ModelTypeRegistry<any MTPDrafterModel>()
        if pairing == .vlm {
            for modelType in ["qwen3_5", "qwen3_5_moe"] {
                await typeRegistry.registerModelType(modelType) { data in
                    let config = try JSONDecoder.json5().decode(
                        MLXVLM.Qwen35Configuration.self, from: data)
                    return Qwen35VLMNextNDraftModel(config)
                }
            }
        } else {
            await typeRegistry.registerModelType("qwen3_5_text") { data in
                let config = try JSONDecoder.json5().decode(
                    Qwen35TextConfiguration.self, from: data)
                return Qwen35MTPDraftModel(config)
            }
            for modelType in ["qwen3_5", "qwen3_5_moe"] {
                await typeRegistry.registerModelType(modelType) { data in
                    let config = try JSONDecoder.json5().decode(
                        MLXLLM.Qwen35Configuration.self, from: data)
                    return Qwen35MTPDraftModel(config)
                }
            }
        }
        let factory = MTPDrafterModelFactory(
            typeRegistry: typeRegistry,
            modelRegistry: MTPDrafterRegistry.shared
        )
        return try await factory.load(from: directory, using: #huggingFaceTokenizerLoader())
    }
}
