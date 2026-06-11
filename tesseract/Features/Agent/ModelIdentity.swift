import Foundation
import MLXLMCommon

/// The load-time, directory-derived facts about a model: "what model is this,
/// and what does that imply downstream." Computed **once** from the model
/// directory at load and read by every gate in the load path, replacing the
/// loose `detect*`/`is*` statics that each re-parsed `config.json` at their own
/// call site (four parses per load).
///
/// Construction is **total and non-throwing**: a missing or malformed
/// `config.json` / `chat_template.jinja` degrades to safe defaults rather than
/// erroring at load. The directory `URL` is the seam — production model dirs and
/// test fixture dirs are its two real inhabitants, so there is no filesystem
/// port.
///
/// See `CONTEXT.md` → Language → Model loading (**Model Identity**). Quant-format
/// routing (`isParoQuantModel`) and the weight `ModelFingerprint` are
/// deliberately *not* here: the former is a container-load concern, the latter
/// throws and is identity-for-cache-invalidation, not a capability fact.
nonisolated struct ModelIdentity: Sendable, Equatable {

    /// The image-keying facts of a recognized vision family (PRD #72): the
    /// placeholder pad token whose prepared runs are one image each, and the
    /// spatial merge size the **Position Anchor** reconstruction needs to turn
    /// an image grid into its M-RoPE position span. `nil` means the loaded
    /// family is not recognized for image keying — an image-bearing request
    /// then degrades to an **Unkeyed Completion**.
    struct ImageKeying: Sendable, Equatable {
        let imagePadTokenId: Int
        let spatialMergeSize: Int
    }

    /// Chat-template tool-call format. `nil` means "no override — use the
    /// vendor JSON default." Qwen3.5 uses XML function syntax
    /// (`<function=name>…</function>` inside `<tool_call>`).
    let toolCallFormat: ToolCallFormat?

    /// `true` when the top-level `model_type` has the `qwen3_5` prefix — the
    /// Qwen3.5 family (dense, MoE, text, and VLM variants all share it).
    let isQwen35: Bool

    /// `true` for the Qwen3.5-family MoE variant (`model_type == qwen3_5_moe`).
    /// A discriminator for MoE-specific specialization downstream.
    let isMoE: Bool

    /// `true` when the chat template opens a `<think>` block in its
    /// generation-prompt section.
    let promptStartsThinking: Bool

    /// FLOP/state-size profile the eviction policy scores against. **Total**:
    /// a non-Qwen3.5 or unparseable config yields `ModelFlopProfile.fallback`,
    /// never `nil`, so the single consumer (`EvictionPolicy`) never handles an
    /// absent profile.
    let flopProfile: ModelFlopProfile

    /// Image-keying facts for the Qwen3.5 vision variant; `nil` for text-only
    /// models and unrecognized families.
    let imageKeying: ImageKeying?

    /// Build the identity from a model directory, reading `config.json` and
    /// `chat_template.jinja` **exactly once each**. The directory-based
    /// constructor is the module-facing interface; total and non-throwing.
    init(directory: URL) {
        let configJSON = Self.loadConfigJSON(directory: directory)
        let chatTemplate = try? String(
            contentsOf: directory.appendingPathComponent("chat_template.jinja"),
            encoding: .utf8
        )
        self.init(configJSON: configJSON, chatTemplate: chatTemplate)
    }

    /// Internal interpretation seam: derive the facts from already-loaded
    /// inputs, with no disk access. Pure — `init(directory:)` delegates here
    /// after its two reads. Reachable by `@testable` tests for no-disk
    /// interpretation coverage; the directory-based init is the interface this
    /// is not part of. See `CONTEXT.md` (Model Identity).
    init(configJSON: [String: Any]?, chatTemplate: String?) {
        let modelType = configJSON?["model_type"] as? String
        self.isQwen35 = modelType?.hasPrefix("qwen3_5") ?? false
        self.isMoE = modelType == "qwen3_5_moe"
        self.toolCallFormat = Self.interpretToolCallFormat(modelType: modelType)
        self.promptStartsThinking = Self.interpretPromptStartsThinking(chatTemplate: chatTemplate)
        self.flopProfile = Self.interpretFlopProfile(configJSON: configJSON)
        self.imageKeying = Self.interpretImageKeying(configJSON: configJSON)
    }

    // MARK: - Interpretation (pure)

    /// `config.json` parsed once into a top-level dict; `nil` if missing or
    /// unparseable.
    private static func loadConfigJSON(directory: URL) -> [String: Any]? {
        let configURL = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return json
    }

    /// Defers to the vendor's `model_type` inference, which already maps the
    /// Qwen3.5 family to `.xmlFunction`. A `nil` `model_type` ⇒ `nil` (no
    /// override — use the vendor JSON default).
    private static func interpretToolCallFormat(modelType: String?) -> ToolCallFormat? {
        guard let modelType else { return nil }
        return ToolCallFormat.infer(from: modelType)
    }

    /// All known thinking templates put `<think>` right after
    /// `<|im_start|>assistant` in the `add_generation_prompt` block at the end
    /// of the template.
    private static func interpretPromptStartsThinking(chatTemplate: String?) -> Bool {
        guard let chatTemplate,
              let genPromptRange = chatTemplate.range(of: "add_generation_prompt")
        else { return false }
        return chatTemplate[genPromptRange.upperBound...].contains("<think>")
    }

    /// Qwen3.5 hybrid profile from `config.json` (the VLM variant nests
    /// architecture fields under `text_config`; LLM-only puts them at the top
    /// level). Non-Qwen3.5, missing fields, or a malformed config fall back to
    /// `ModelFlopProfile.fallback` — the one shared default, so the
    /// parse-failure path and `LLMActor`'s pre-load path agree in one place.
    private static func interpretFlopProfile(configJSON: [String: Any]?) -> ModelFlopProfile {
        guard let root = configJSON,
              let topModelType = root["model_type"] as? String,
              topModelType.hasPrefix("qwen3_5")
        else { return .fallback }

        let textConfig = (root["text_config"] as? [String: Any]) ?? root
        guard let hiddenLayers = textConfig["num_hidden_layers"] as? Int,
              let hiddenSize = textConfig["hidden_size"] as? Int,
              let linearNumValueHeads = textConfig["linear_num_value_heads"] as? Int,
              let linearKeyHeadDim = textConfig["linear_key_head_dim"] as? Int,
              let fullAttentionInterval = textConfig["full_attention_interval"] as? Int
        else { return .fallback }

        return .qwen35(
            hiddenLayers: hiddenLayers,
            hiddenSize: hiddenSize,
            linearNumValueHeads: linearNumValueHeads,
            linearKeyHeadDim: linearKeyHeadDim,
            fullAttentionInterval: fullAttentionInterval
        )
    }

    /// Image keying exists only for the Qwen3.5 vision variant — the family the
    /// pseudo-token keying and Position Anchor seeding are spike-verified
    /// against (ADR-0007). Recognized by the `qwen3_5` prefix plus a
    /// `vision_config` block; the defaults mirror the vendor's `Qwen35`
    /// config decode (`image_token_id` 248056, `spatial_merge_size` 2).
    private static func interpretImageKeying(configJSON: [String: Any]?) -> ImageKeying? {
        guard let root = configJSON,
              let modelType = root["model_type"] as? String,
              modelType.hasPrefix("qwen3_5"),
              let visionConfig = root["vision_config"] as? [String: Any]
        else { return nil }

        return ImageKeying(
            imagePadTokenId: root["image_token_id"] as? Int ?? 248_056,
            spatialMergeSize: visionConfig["spatial_merge_size"] as? Int ?? 2
        )
    }
}
