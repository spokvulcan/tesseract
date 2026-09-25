import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// Behaviour of the **Model Identity** value — the load-time facts derived from
/// a model directory. The interface is the test surface: tests construct an
/// identity and assert its facets, never reaching into how the files are parsed.
///
/// Two surfaces are exercised: `init(directory:)` against real fixture
/// directories (the established idiom — runs the real file-reading path), and
/// the internal `init(configJSON:chatTemplate:)` against dict/string literals
/// for fast, no-disk interpretation coverage.
struct ModelIdentityTests {

    // MARK: - Family & variant (directory surface)

    /// Tracer bullet: a Qwen3.5 model directory yields the family fact and the
    /// Qwen3.5 family detection, read through the directory-based interface.
    @Test func qwen35DirectoryYieldsFamily() throws {
        let dir = try makeModelDir(config: #"{ "model_type": "qwen3_5" }"#)
        defer { try? FileManager.default.removeItem(at: dir) }

        let identity = ModelIdentity(directory: dir)

        #expect(identity.isQwen35 == true)
    }

    /// The MoE variant sets both `isQwen35` and `isMoE`.
    @Test func qwen35MoEDirectoryIsFamilyAndMoE() throws {
        let dir = try makeModelDir(config: #"{ "model_type": "qwen3_5_moe" }"#)
        defer { try? FileManager.default.removeItem(at: dir) }

        let identity = ModelIdentity(directory: dir)

        #expect(identity.isQwen35 == true)
        #expect(identity.isMoE == true)
    }

    /// The dense variant is family but not MoE.
    @Test func qwen35DenseDirectoryIsNotMoE() throws {
        let dir = try makeModelDir(config: #"{ "model_type": "qwen3_5" }"#)
        defer { try? FileManager.default.removeItem(at: dir) }

        #expect(ModelIdentity(directory: dir).isMoE == false)
    }

    // MARK: - Base Architecture

    /// A Rotated Ternary Checkpoint declares its own `model_type` beside
    /// `base_model_type`; the family facts key on the Base Architecture.
    @Test func rotatedPackResolvesFamilyFromBaseModelType() throws {
        let dir = try makeModelDir(
            config: #"{ "model_type": "prism_hadamard_qwen35", "base_model_type": "qwen3_5" }"#)
        defer { try? FileManager.default.removeItem(at: dir) }

        let identity = ModelIdentity(directory: dir)

        #expect(identity.isQwen35 == true)
        #expect(identity.isMoE == false)
    }

    /// The pack is recognized by its module manifest, never by its name: the
    /// same `model_type` without a manifest is not a Rotated Ternary
    /// Checkpoint, and a manifest whose entries name no `path` does not count.
    @Test func rotatedTernaryCheckpointIsRecognizedByItsModuleManifest() {
        let rotated = ModelIdentity(
            configJSON: [
                "model_type": "prism_hadamard_qwen35",
                "base_model_type": "qwen3_5",
                "modules": [
                    ["path": "model.embed_tokens", "block": 1024, "embedding": true],
                    ["path": "lm_head", "block": 1024],
                ],
            ],
            chatTemplate: nil)
        #expect(rotated.isRotatedTernaryCheckpoint)

        let namedOnly = ModelIdentity(
            configJSON: ["model_type": "prism_hadamard_qwen35", "base_model_type": "qwen3_5"],
            chatTemplate: nil)
        #expect(!namedOnly.isRotatedTernaryCheckpoint)

        let pathless = ModelIdentity(
            configJSON: ["model_type": "qwen3_5", "modules": [["block": 1024]]],
            chatTemplate: nil)
        #expect(!pathless.isRotatedTernaryCheckpoint)

        #expect(
            !ModelIdentity(configJSON: ["model_type": "qwen3_5"], chatTemplate: nil)
                .isRotatedTernaryCheckpoint)
    }

    /// Without `base_model_type`, `model_type` is the Base Architecture.
    @Test func baseArchitectureFallsBackToModelType() {
        #expect(ModelIdentity.baseArchitecture(configJSON: ["model_type": "qwen3_5"]) == "qwen3_5")
        #expect(
            ModelIdentity.baseArchitecture(configJSON: [
                "model_type": "prism_hadamard_qwen35", "base_model_type": "qwen3_5_moe",
            ]) == "qwen3_5_moe")
        #expect(ModelIdentity.baseArchitecture(configJSON: nil) == nil)
    }

    /// Every Qwen3.5-gated interpreter reads the Base Architecture: a rotated
    /// pack of a VLM checkpoint prices like the checkpoint it runs.
    @Test func rotatedPackKeysEveryFamilyInterpreterOnBaseArchitecture() {
        let identity = ModelIdentity(
            configJSON: [
                "model_type": "prism_hadamard_qwen35",
                "base_model_type": "qwen3_5",
                "image_token_id": 248_056,
                "text_config": [
                    "num_hidden_layers": 64,
                    "hidden_size": 5120,
                    "linear_num_value_heads": 48,
                    "linear_value_head_dim": 128,
                    "linear_key_head_dim": 128,
                    "num_attention_heads": 24,
                    "full_attention_interval": 4,
                    "dtype": "float16",
                ] as [String: Any],
                "vision_config": ["num_heads": 16, "spatial_merge_size": 2] as [String: Any],
            ],
            chatTemplate: nil
        )

        #expect(identity.flopProfile != .fallback)
        #expect(identity.flopProfile.attentionLayers == 16)
        #expect(identity.fullAttentionScratchProfile?.attentionHeads == 24)
        #expect(identity.visionAttentionScratchProfile?.attentionHeads == 16)
        #expect(identity.imageKeying?.imagePadTokenId == 248_056)
    }

    // MARK: - flopProfile is Total (directory surface)

    /// VLM checkpoints nest architecture fields under `text_config`; a 9B-shaped
    /// config yields a profile distinct from the 4B fallback.
    @Test func flopProfileReadsNestedTextConfig() throws {
        let dir = try makeModelDir(
            config: #"""
                {
                  "model_type": "qwen3_5",
                  "text_config": {
                    "model_type": "qwen3_5_text",
                    "num_hidden_layers": 32,
                    "hidden_size": 4096,
                    "linear_num_value_heads": 32,
                    "linear_key_head_dim": 128,
                    "full_attention_interval": 4
                  }
                }
                """#)
        defer { try? FileManager.default.removeItem(at: dir) }

        let profile = ModelIdentity(directory: dir).flopProfile
        #expect(profile.hiddenSize == 4096)
        #expect(profile.attentionLayers == 8)
        #expect(profile.ssmLayers == 24)
        #expect(profile.mlpLayers == 32)
        #expect(profile.ssmStateDim == 32 * 128)
        #expect(profile != .qwen35_4B_PARO)
    }

    /// LLM-only checkpoints keep the fields at the top level; the same 9B shape
    /// as the nested case (now top-level) yields a profile distinct from the 4B
    /// fallback — proving the top-level fields are actually read, not silently
    /// falling through to `.qwen35_4B_PARO`. (A 4B-shaped fixture would parse
    /// byte-identical to the fallback and pass even if the read path broke.)
    @Test func flopProfileReadsTopLevelFields() throws {
        let dir = try makeModelDir(
            config: #"""
                {
                  "model_type": "qwen3_5",
                  "num_hidden_layers": 32,
                  "hidden_size": 4096,
                  "linear_num_value_heads": 32,
                  "linear_key_head_dim": 128,
                  "full_attention_interval": 4
                }
                """#)
        defer { try? FileManager.default.removeItem(at: dir) }

        let profile = ModelIdentity(directory: dir).flopProfile
        #expect(profile.hiddenSize == 4096)
        #expect(profile.attentionLayers == 8)
        #expect(profile.ssmLayers == 24)
        #expect(profile.mlpLayers == 32)
        #expect(profile.ssmStateDim == 32 * 128)
        #expect(profile != .qwen35_4B_PARO)
    }

    @Test func flopProfilePricesNanbeigeLoopsAsEffectiveDepth() throws {
        // The looped transformer runs every layer `num_loops` times per token
        // (issue #422) — eviction must price 44 dense attention blocks, not
        // fall back to the hybrid Qwen 4B profile.
        let dir = try makeModelDir(
            config: #"""
                {
                  "model_type": "nanbeige",
                  "num_hidden_layers": 22,
                  "hidden_size": 3072,
                  "num_loops": 2
                }
                """#)
        defer { try? FileManager.default.removeItem(at: dir) }

        let profile = ModelIdentity(directory: dir).flopProfile
        #expect(profile.attentionLayers == 44)
        #expect(profile.mlpLayers == 44)
        #expect(profile.ssmLayers == 0)
        #expect(profile.hiddenSize == 3072)
        #expect(profile != .qwen35_4B_PARO)
    }

    /// An unknown `model_type` falls back to `.qwen35_4B_PARO` — the field is
    /// Total, so eviction never handles an absent profile. (Was `nil` when the
    /// detector lived on `LLMActor`.)
    @Test func flopProfileFallsBackForUnknownModelType() throws {
        let dir = try makeModelDir(config: #"{ "model_type": "llama", "hidden_size": 4096 }"#)
        defer { try? FileManager.default.removeItem(at: dir) }

        let identity = ModelIdentity(directory: dir)
        #expect(identity.flopProfile == .qwen35_4B_PARO)
        #expect(identity.isQwen35 == false)
    }

    /// A directory with no `config.json` and no `chat_template.jinja` yields the
    /// fully-defaulted identity — construction is total and non-throwing, and
    /// the whole value is asserted via `Equatable`.
    @Test func missingFilesYieldDefaultIdentity() throws {
        let dir = try makeModelDir(config: nil)
        defer { try? FileManager.default.removeItem(at: dir) }

        #expect(ModelIdentity(directory: dir) == ModelIdentity(configJSON: nil, chatTemplate: nil))
    }

    // MARK: - No-disk interpretation seam

    /// The seam interprets config + template together with no disk access.
    @Test func seamInterpretsConfigAndTemplate() {
        let identity = ModelIdentity(
            configJSON: ["model_type": "qwen3_5_moe"],
            chatTemplate: "add_generation_prompt ... <think>"
        )
        #expect(identity.isQwen35 == true)
        #expect(identity.isMoE == true)
    }

    /// `nil` inputs through the seam yield the same Total defaults as a bare
    /// directory: the fallback flop profile and all-false facts.
    @Test func seamWithNilInputsYieldsDefaults() {
        let identity = ModelIdentity(configJSON: nil, chatTemplate: nil)
        #expect(identity.isQwen35 == false)
        #expect(identity.isMoE == false)
        #expect(identity.flopProfile == .qwen35_4B_PARO)
        #expect(identity.fullAttentionScratchProfile == nil)
    }

    /// Two identities built from equal inputs compare equal — one comparison
    /// pins a whole expected identity.
    @Test func equalInputsProduceEqualIdentities() {
        let a = ModelIdentity(configJSON: ["model_type": "qwen3_5"], chatTemplate: nil)
        let b = ModelIdentity(configJSON: ["model_type": "qwen3_5"], chatTemplate: nil)
        let other = ModelIdentity(configJSON: ["model_type": "llama"], chatTemplate: nil)
        #expect(a == b)
        #expect(a != other)
    }

    // MARK: - Full-attention scratch profile

    /// Qwen3.6 PARO exposes full-attention geometry under `text_config`; the
    /// memory guard needs exactly these fields to preflight Qwen35 VLM prepare.
    @Test func fullAttentionScratchProfileReadsNestedQwen36Fields() {
        let identity = ModelIdentity(
            configJSON: [
                "model_type": "qwen3_5",
                "text_config": [
                    "num_attention_heads": 24,
                    "full_attention_interval": 4,
                    "dtype": "bfloat16",
                ],
            ],
            chatTemplate: nil
        )

        #expect(
            identity.fullAttentionScratchProfile
                == ModelIdentity.FullAttentionScratchProfile(
                    attentionHeads: 24,
                    bytesPerElement: 2
                ))
    }

    /// LLM-only layouts keep the same fields at top level; fp32 activations double
    /// the preflight bytes.
    @Test func fullAttentionScratchProfileReadsTopLevelDType() {
        let identity = ModelIdentity(
            configJSON: [
                "model_type": "qwen3_5",
                "num_attention_heads": 16,
                "full_attention_interval": 4,
                "dtype": "float32",
            ],
            chatTemplate: nil
        )

        #expect(
            identity.fullAttentionScratchProfile
                == ModelIdentity.FullAttentionScratchProfile(
                    attentionHeads: 16,
                    bytesPerElement: 4
                ))
    }

    /// Non-Qwen3.5 families are not guarded by Qwen35-specific scratch math.
    @Test func fullAttentionScratchProfileIsNilOffQwen35Family() {
        let identity = ModelIdentity(
            configJSON: [
                "model_type": "qwen2_vl",
                "num_attention_heads": 16,
                "full_attention_interval": 4,
                "dtype": "bfloat16",
            ],
            chatTemplate: nil
        )

        #expect(identity.fullAttentionScratchProfile == nil)
    }

    // MARK: - Vision-tower scratch profile (ADR-0014)

    /// The Qwen3.5 vision variant (family prefix + `vision_config`) exposes the
    /// vision tower's head count so the patch guard can price its global
    /// O(patches²) attention matrix. PARO's ViT declares `num_heads: 16`.
    @Test func visionScratchProfileReadsVisionConfigHeads() {
        let identity = ModelIdentity(
            configJSON: [
                "model_type": "qwen3_5",
                "vision_config": ["num_heads": 16],
            ],
            chatTemplate: nil
        )

        #expect(
            identity.visionAttentionScratchProfile
                == ModelIdentity.FullAttentionScratchProfile(
                    attentionHeads: 16,
                    bytesPerElement: 2
                ))
    }

    /// Text-only Qwen3.5 (no `vision_config`) has no vision tower to guard.
    @Test func visionScratchProfileIsNilWithoutVisionConfig() {
        let identity = ModelIdentity(
            configJSON: ["model_type": "qwen3_5"],
            chatTemplate: nil
        )

        #expect(identity.visionAttentionScratchProfile == nil)
    }

    /// A `vision_config` that omits `num_heads` cannot be priced, so the guard
    /// stays inert rather than inventing a head count.
    @Test func visionScratchProfileIsNilWithoutHeadCount() {
        let identity = ModelIdentity(
            configJSON: [
                "model_type": "qwen3_5",
                "vision_config": ["spatial_merge_size": 2],
            ],
            chatTemplate: nil
        )

        #expect(identity.visionAttentionScratchProfile == nil)
    }

    /// Non-Qwen3.5 vision families are not priced by the Qwen3.5-specific guard.
    @Test func visionScratchProfileIsNilOffTheRecognizedFamily() {
        let identity = ModelIdentity(
            configJSON: [
                "model_type": "qwen2_vl",
                "vision_config": ["num_heads": 16],
            ],
            chatTemplate: nil
        )

        #expect(identity.visionAttentionScratchProfile == nil)
    }

    // MARK: - Image keying

    /// The Qwen3.5 vision variant (family prefix + `vision_config`) yields the
    /// image-keying facts, reading the explicit token id and merge size.
    @Test func imageKeyingReadsVisionConfigFields() {
        let identity = ModelIdentity(
            configJSON: [
                "model_type": "qwen3_5",
                "image_token_id": 248_056,
                "vision_config": ["spatial_merge_size": 2],
            ],
            chatTemplate: nil
        )
        #expect(
            identity.imageKeying
                == ModelIdentity.ImageKeying(
                    imagePadTokenId: 248_056, spatialMergeSize: 2
                ))
    }

    /// Missing optional fields fall back to the vendor decode defaults.
    @Test func imageKeyingDefaultsMirrorTheVendorDecode() {
        let identity = ModelIdentity(
            configJSON: ["model_type": "qwen3_5", "vision_config": [String: Any]()],
            chatTemplate: nil
        )
        #expect(
            identity.imageKeying
                == ModelIdentity.ImageKeying(
                    imagePadTokenId: 248_056, spatialMergeSize: 2
                ))
    }

    /// Text-only Qwen3.5 (no `vision_config`) and non-Qwen3.5 vision models
    /// are not recognized for image keying — their image requests degrade to
    /// Unkeyed Completions instead of being keyed with unverified geometry.
    @Test func imageKeyingIsNilOffTheRecognizedVisionFamily() {
        #expect(
            ModelIdentity(
                configJSON: ["model_type": "qwen3_5"], chatTemplate: nil
            ).imageKeying == nil)
        #expect(
            ModelIdentity(
                configJSON: ["model_type": "llava", "vision_config": [String: Any]()],
                chatTemplate: nil
            ).imageKeying == nil)
        #expect(ModelIdentity(configJSON: nil, chatTemplate: nil).imageKeying == nil)
    }

    // MARK: - Fixtures

    /// Write `config.json` (when non-nil) into a fresh temp directory.
    private func makeModelDir(config: String?) throws -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("modelidentity-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        if let config {
            try config.write(
                to: dir.appendingPathComponent("config.json"),
                atomically: true, encoding: .utf8
            )
        }
        return dir
    }
}
