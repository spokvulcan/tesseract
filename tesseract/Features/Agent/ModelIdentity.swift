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
    /// placeholder pad token whose prepared runs are one image each, plus the
    /// family's position-span rule. `nil` means the loaded family is not
    /// recognized for image keying — an image-bearing request then degrades
    /// to an **Unkeyed Completion**.
    /// Framing tokens the *processor* wraps around a media placeholder run
    /// (Gemma 4's `boi…eoi` / `boa…eoa`). They exist in the prepared/live
    /// sequence but not in a chat-template render, which emits only the bare
    /// placeholder — so render→key translation must splice them back in.
    /// `nil` when the family's framing (if any) is already part of the
    /// template render (Qwen-VL's `vision_start`/`vision_end`).
    struct MediaFraming: Sendable, Equatable, Hashable {
        let startTokenId: Int
        let endTokenId: Int
    }

    struct ImageKeying: Sendable, Equatable {
        /// How an image's placeholder run maps onto model positions.
        enum PositionSpanRule: Sendable, Equatable {
            /// Qwen-VL M-RoPE: span = max(t, h/m, w/m) from the processed
            /// grid — the **Position Anchor** reconstruction's geometry.
            case mropeGrid(spatialMergeSize: Int)
            /// Standard-RoPE families (Gemma 4 unified): every soft token
            /// occupies one sequential position, so span == run length and
            /// warm restores need no anchor.
            case sequential
        }

        let imagePadTokenId: Int
        let spanRule: PositionSpanRule
        /// Processor-added framing around the run; see ``MediaFraming``.
        var framing: MediaFraming? = nil

        /// Compatibility surface for the M-RoPE consumers (grid logging, the
        /// Position Anchor geometry). `nil` under the sequential rule.
        var spatialMergeSize: Int? {
            if case .mropeGrid(let merge) = spanRule { return merge }
            return nil
        }

        /// Whether warm continuations must seed the **Position Anchor** —
        /// true only for the M-RoPE family; sequential families restore with
        /// nil-state semantics like text models.
        var anchorsWarmContinuations: Bool {
            if case .mropeGrid = spanRule { return true }
            return false
        }

        /// The Qwen-VL shape, preserved for its existing construction sites.
        init(imagePadTokenId: Int, spatialMergeSize: Int) {
            self.imagePadTokenId = imagePadTokenId
            self.spanRule = .mropeGrid(spatialMergeSize: spatialMergeSize)
        }

        init(imagePadTokenId: Int, spanRule: PositionSpanRule, framing: MediaFraming? = nil) {
            self.imagePadTokenId = imagePadTokenId
            self.spanRule = spanRule
            self.framing = framing
        }
    }

    /// The audio-keying facts of a recognized audio family: the placeholder
    /// pad token whose prepared runs are one clip each. Audio soft tokens are
    /// always sequential positions (no grid geometry exists), so the pad
    /// token is the whole identity. `nil` means the loaded family is not
    /// recognized for audio keying — an audio-bearing request then degrades
    /// to an **Unkeyed Completion**.
    struct AudioKeying: Sendable, Equatable {
        let audioPadTokenId: Int
        /// Processor-added framing around the run; see ``MediaFraming``.
        var framing: MediaFraming? = nil

        init(audioPadTokenId: Int, framing: MediaFraming? = nil) {
            self.audioPadTokenId = audioPadTokenId
            self.framing = framing
        }
    }

    /// Scratch-buffer geometry for Qwen3.5/Qwen3.6 full-attention layers.
    /// Used as a preflight guard before the VLM `prepare` path asks Metal for
    /// the `[batch, heads, L, L]` attention score matrix.
    struct FullAttentionScratchProfile: Sendable, Equatable {
        let attentionHeads: Int
        let bytesPerElement: Int

        /// Single-shot full-attention scratch: `[heads, L, L]` — the bound a
        /// vendor `prepare` over `L` tokens allocates in one pass (the OOM
        /// source ADR-0007 phase 2 replaces).
        func scoreMatrixBytes(sequenceLength: Int) -> UInt64? {
            scoreMatrixBytes(queryLength: sequenceLength, contextLength: sequenceLength)
        }

        /// Chunked full-attention scratch: `[heads, query, context]` — `query`
        /// tokens (one continuation window) attending over `context` KV slots.
        /// The windowed continuation (ADR-0007 phase 2) honors this bound, so a
        /// long image span estimates `heads·window·L`, linear in `L`, instead of
        /// the single-shot `heads·L²`.
        func scoreMatrixBytes(queryLength: Int, contextLength: Int) -> UInt64? {
            guard queryLength >= 0, contextLength >= 0, attentionHeads > 0, bytesPerElement > 0
            else {
                return nil
            }
            let query = UInt64(queryLength)
            let context = UInt64(contextLength)
            let heads = UInt64(attentionHeads)
            let bytes = UInt64(bytesPerElement)
            let area = query.multipliedReportingOverflow(by: context)
            guard !area.overflow else { return nil }
            let withHeads = area.partialValue.multipliedReportingOverflow(by: heads)
            guard !withHeads.overflow else { return nil }
            let total = withHeads.partialValue.multipliedReportingOverflow(by: bytes)
            guard !total.overflow else { return nil }
            return total.partialValue
        }
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

    /// `true` for the Gemma 4 unified family (`model_type == gemma4_unified`)
    /// — the encoder-free multimodal 12B. Discriminates the generation-prompt
    /// literal, which is family-shaped (`<|turn>model`, not `<|im_start|>`).
    let isGemma4Unified: Bool

    /// `true` when the chat template opens a `<think>` block in its
    /// generation-prompt section.
    let promptStartsThinking: Bool

    /// `true` when the template's generation-prompt section emits channel
    /// markup (`<|channel>`) after the assistant header — Gemma 4's empty
    /// pre-closed thought channel (`<|channel>thought\n<channel|>` under
    /// `enable_thinking=false`). Those tokens sit in the live KV of every
    /// request but vanish when the finished assistant turn is re-rendered, so
    /// the canonical sequence diverges from the live cache exactly like a
    /// `<think>` strip does. Leaf-store mode selection treats it as such.
    let promptEndsWithClosedChannel: Bool

    /// The literal string the chat template appends to open the model's turn
    /// (assistant header plus any pre-filled think/channel markup). The
    /// Prefill Planner subtracts this suffix from the full token path to find
    /// the last-message boundary; a family-wrong literal silently costs the
    /// boundary (no misbehavior, just a lost checkpoint site).
    var generationPromptSuffix: String {
        if isGemma4Unified {
            return promptEndsWithClosedChannel
                ? "<|turn>model\n<|channel>thought\n<channel|>"
                : "<|turn>model\n"
        }
        return promptStartsThinking
            ? "<|im_start|>assistant\n<think>\n"
            : "<|im_start|>assistant\n"
    }

    /// The opt-in render flags the chat template natively declares — the
    /// subset of `TemplateRenderContext`'s known flags the template text
    /// actually references (issue #98). The capability gate for the
    /// **Preserve-Thinking Render**: introspection, never model name, so a
    /// future template that adds `preserve_thinking` needs no code change
    /// and Qwen3.5-PARO (which lacks it) is naturally excluded.
    let declaredTemplateFlags: Set<TemplateRenderFlag>

    /// FLOP/state-size profile the eviction policy scores against. **Total**:
    /// a non-Qwen3.5 or unparseable config yields `ModelFlopProfile.fallback`,
    /// never `nil`, so the single consumer (`EvictionPolicy`) never handles an
    /// absent profile.
    let flopProfile: ModelFlopProfile

    /// Per-layer full-attention scratch geometry, when the loaded architecture
    /// exposes it. `nil` for unrecognized/non-full-attention models.
    let fullAttentionScratchProfile: FullAttentionScratchProfile?

    /// Vision-tower full-attention scratch geometry, when the loaded family
    /// ships a `vision_config`. The `qwen3_5` ViT attends *globally* over every
    /// patch of an image (no `window_size`/`fullatt_block_indexes`), so a single
    /// forward materializes one `[vision_heads, ΣpatchesΣpatches]` bf16 score
    /// matrix — O(patches²), uncapped, and invisible to the language-model
    /// `fullAttentionScratchProfile`. This profile lets the patch guard price
    /// that matrix before the tower runs. `nil` for text-only or unrecognized
    /// families (ADR-0014).
    let visionAttentionScratchProfile: FullAttentionScratchProfile?

    /// Image-keying facts for the recognized vision families (Qwen3.5 VL,
    /// Gemma 4 unified); `nil` for text-only models and unrecognized families.
    let imageKeying: ImageKeying?

    /// Audio-keying facts for the recognized audio family (Gemma 4 unified —
    /// the encoder-free 12B, whose `config.json` ships an `audio_config`);
    /// `nil` for audio-less models and unrecognized families. **Audio-capable**
    /// at the catalog level is exactly `audioKeying != nil`.
    let audioKeying: AudioKeying?

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
        self.isGemma4Unified = modelType == "gemma4_unified"
        self.toolCallFormat = Self.interpretToolCallFormat(modelType: modelType)
        self.promptStartsThinking = Self.interpretPromptStartsThinking(chatTemplate: chatTemplate)
        self.promptEndsWithClosedChannel = Self.interpretPromptEndsWithClosedChannel(
            chatTemplate: chatTemplate
        )
        self.declaredTemplateFlags = Self.interpretDeclaredTemplateFlags(chatTemplate: chatTemplate)
        self.flopProfile = Self.interpretFlopProfile(configJSON: configJSON)
        self.fullAttentionScratchProfile = Self.interpretFullAttentionScratchProfile(
            configJSON: configJSON
        )
        self.visionAttentionScratchProfile = Self.interpretVisionAttentionScratchProfile(
            configJSON: configJSON
        )
        self.imageKeying = Self.interpretImageKeying(configJSON: configJSON)
        self.audioKeying = Self.interpretAudioKeying(configJSON: configJSON)
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

    /// Gemma 4's template ends its generation prompt with channel markup —
    /// the empty pre-closed thought channel under `enable_thinking=false`.
    /// Probed the same way as `<think>`: channel markup appearing after the
    /// (final) `add_generation_prompt` reference is generation-prompt markup,
    /// not turn-body rendering.
    private static func interpretPromptEndsWithClosedChannel(chatTemplate: String?) -> Bool {
        guard let chatTemplate,
            let genPromptRange = chatTemplate.range(
                of: "add_generation_prompt", options: .backwards)
        else { return false }
        return chatTemplate[genPromptRange.upperBound...].contains("<|channel>")
    }

    /// A flag is "declared" when the template text references it — Jinja
    /// templates read their kwargs by name, so a flag the text never mentions
    /// cannot change the render. Only known flags are probed; arbitrary
    /// kwargs never become capabilities.
    private static func interpretDeclaredTemplateFlags(
        chatTemplate: String?
    ) -> Set<TemplateRenderFlag> {
        guard let chatTemplate else { return [] }
        // A flag is declared only when the template *references the variable*,
        // not merely mentions the string. Strip Jinja comments so a flag named
        // only in `{# … #}` doesn't count, then require an identifier-boundary
        // match so a longer name (`preserve_thinking_default`) isn't a false
        // positive. A bare `contains` over-declares the capability and forks a
        // zero-reuse cache partition for a render the template never branches on.
        let scannable = stripJinjaComments(chatTemplate)
        return Set(
            TemplateRenderFlag.allCases.filter {
                referencesIdentifier($0.rawValue, in: scannable)
            })
    }

    /// Remove `{# … #}` Jinja comment blocks (non-greedy, spanning newlines)
    /// so a flag mentioned only in a comment is not read as a declaration.
    private static func stripJinjaComments(_ template: String) -> String {
        guard
            let regex = try? NSRegularExpression(
                pattern: "\\{#.*?#\\}", options: [.dotMatchesLineSeparators]
            )
        else { return template }
        let range = NSRange(template.startIndex..., in: template)
        return regex.stringByReplacingMatches(
            in: template, range: range, withTemplate: " "
        )
    }

    /// Whole-identifier match: `name` not flanked by another identifier
    /// character, so it matches `{% if preserve_thinking %}` but not the
    /// longer identifier `preserve_thinking_default`.
    private static func referencesIdentifier(_ name: String, in text: String) -> Bool {
        let pattern =
            "(?<![A-Za-z0-9_])"
            + NSRegularExpression.escapedPattern(for: name)
            + "(?![A-Za-z0-9_])"
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return text.contains(name)
        }
        return regex.firstMatch(
            in: text, range: NSRange(text.startIndex..., in: text)
        ) != nil
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

    private static func interpretFullAttentionScratchProfile(
        configJSON: [String: Any]?
    ) -> FullAttentionScratchProfile? {
        guard let root = configJSON,
            let topModelType = root["model_type"] as? String,
            topModelType.hasPrefix("qwen3_5")
        else { return nil }

        let textConfig = (root["text_config"] as? [String: Any]) ?? root
        guard let attentionHeads = textConfig["num_attention_heads"] as? Int,
            attentionHeads > 0,
            textConfig["full_attention_interval"] is Int
        else { return nil }

        return FullAttentionScratchProfile(
            attentionHeads: attentionHeads,
            bytesPerElement: bytesPerElement(forActivationDType: textConfig["dtype"] as? String)
        )
    }

    /// Vision-tower scratch geometry for the `qwen3_5` vision variant: the ViT's
    /// `vision_config.num_heads` (16 for PARO), bf16 activations. Gated on the
    /// same `qwen3_5` family prefix as its language-model sibling, plus a
    /// `vision_config` block carrying a positive head count — a config without
    /// one cannot be priced, so the guard stays inert rather than guessing
    /// (ADR-0014).
    private static func interpretVisionAttentionScratchProfile(
        configJSON: [String: Any]?
    ) -> FullAttentionScratchProfile? {
        guard let root = configJSON,
            let modelType = root["model_type"] as? String,
            modelType.hasPrefix("qwen3_5"),
            let visionConfig = root["vision_config"] as? [String: Any],
            let numHeads = visionConfig["num_heads"] as? Int,
            numHeads > 0
        else { return nil }

        return FullAttentionScratchProfile(
            attentionHeads: numHeads,
            bytesPerElement: bytesPerElement(forActivationDType: visionConfig["dtype"] as? String)
        )
    }

    private static func bytesPerElement(forActivationDType dtype: String?) -> Int {
        switch dtype?.lowercased() {
        case "float32", "fp32":
            return 4
        case "float64", "fp64":
            return 8
        case "float16", "fp16", "bfloat16", "bf16":
            return 2
        default:
            return 2
        }
    }

    /// Image keying exists for the two recognized vision families. Qwen3.5:
    /// the `qwen3_5` prefix plus a `vision_config` block, with the M-RoPE
    /// grid rule the pseudo-token keying and Position Anchor seeding are
    /// spike-verified against (ADR-0007); defaults mirror the vendor's
    /// `Qwen35` config decode (`image_token_id` 248056, `spatial_merge_size`
    /// 2). Gemma 4 unified (`gemma4_unified` + `vision_config`): the
    /// sequential rule — soft tokens occupy one position each; default
    /// `image_token_id` 258880 mirrors the published config.
    private static func interpretImageKeying(configJSON: [String: Any]?) -> ImageKeying? {
        guard let root = configJSON,
            let modelType = root["model_type"] as? String
        else { return nil }

        if modelType.hasPrefix("qwen3_5"),
            let visionConfig = root["vision_config"] as? [String: Any]
        {
            return ImageKeying(
                imagePadTokenId: root["image_token_id"] as? Int ?? 248_056,
                spatialMergeSize: visionConfig["spatial_merge_size"] as? Int ?? 2
            )
        }

        if modelType == "gemma4_unified", root["vision_config"] is [String: Any] {
            return ImageKeying(
                imagePadTokenId: root["image_token_id"] as? Int ?? 258_880,
                spanRule: .sequential,
                framing: MediaFraming(
                    startTokenId: root["boi_token_id"] as? Int ?? 255_999,
                    endTokenId: root["eoi_token_id"] as? Int ?? 258_882
                )
            )
        }

        return nil
    }

    /// Audio keying exists only for the encoder-free Gemma 4 unified family:
    /// `gemma4_unified` plus an `audio_config` block — a config without one
    /// is an audio-less export (upstream main strips the audio weights), so
    /// the capability must never be inferred from the family alone. The
    /// default `audio_token_id` 258881 mirrors the published config.
    private static func interpretAudioKeying(configJSON: [String: Any]?) -> AudioKeying? {
        guard let root = configJSON,
            let modelType = root["model_type"] as? String,
            modelType == "gemma4_unified",
            root["audio_config"] is [String: Any]
        else { return nil }

        return AudioKeying(
            audioPadTokenId: root["audio_token_id"] as? Int ?? 258_881,
            framing: MediaFraming(
                // `eoa_token_id` is absent from the shipped config root; the
                // fallbacks are the family constants the vendor pipeline uses.
                startTokenId: root["boa_token_id"] as? Int ?? 256_000,
                endTokenId: root["eoa_token_id"] as? Int ?? 258_883
            )
        )
    }
}

extension ModelIdentity {
    /// Whether the model at `directory` declares `flag` in its chat template.
    /// `init(directory:)` reads `chat_template.jinja` + `config.json` from disk,
    /// so the probe runs off the MainActor (ADR-0001) — a view can `await` it
    /// without stuttering while opening or switching a settings pane. The single
    /// home for the template-flag capability probe shared by the agent-preferences
    /// and server-configuration **Preserve-Thinking Render** toggles (issue #98).
    static func declares(
        _ flag: TemplateRenderFlag,
        atDirectory directory: URL
    ) async -> Bool {
        await Task.detached {
            ModelIdentity(directory: directory).declaredTemplateFlags.contains(flag)
        }.value
    }
}
