import Foundation
import MLXLMCommon

nonisolated enum TemplateRenderFlag: String, CaseIterable, Sendable, Hashable, Codable {
    case preserveThinking = "preserve_thinking"
    /// Whether the model thinks at all this turn. Every current thinking
    /// template defaults it **on** (the `enable_thinking is defined and … is
    /// false` shape only fires on an explicit `false`), so the kwarg is
    /// emitted only when a request turns thinking off — which also swaps the
    /// generation prompt to a closed, empty think block, see
    /// ``TemplateRenderContext/disablesThinking``.
    case enableThinking = "enable_thinking"
}

/// The native reasoning-effort levels of an effort-declaring chat template
/// (ADR-0060; Qwen3.8 is the first): prose the template injects into the
/// request's *first system block*, so a different level is a different
/// prefix from token 0. `xhigh` and `low` each inject an instruction
/// sentence; `medium` injects nothing; the template's own default (`xhigh`
/// for Qwen3.8) applies when the kwarg is absent.
nonisolated enum ReasoningEffort: String, CaseIterable, Sendable, Hashable, Codable {
    case low
    case medium
    case xhigh
}

/// The resolved chat-template render context for one completion (PRD #94,
/// issue #98): which template-declared render kwargs this request renders
/// with — the boolean flags (`preserve_thinking`, `enable_thinking`) plus the
/// string-valued **Reasoning Effort** level (ADR-0060).
///
/// One value is resolved per request — request-level `chat_template_kwargs`
/// wins per flag, the per-model app setting is the fallback, and flags the
/// loaded template does not declare are ignored entirely (capability gate by
/// template introspection, never model name; see
/// `ModelIdentity.declaredTemplateFlags`). The value then rides the whole
/// completion: every template render in the pipeline merges `kwargs` into its
/// `additionalContext`, and `digest` folds the kwargs into the conversation's
/// template-context digest and the cache partition — so toggling a flag lands
/// in a fresh partition and mixed renders can never share one.
///
/// Templates disagree on the flag's **default polarity**: Qwen3.6 strips
/// prior think blocks unless `preserve_thinking is true`; Nanbeige4.2
/// preserves them unless `preserve_thinking is false`. The desired state is
/// therefore separated from the wire form — `resolve` emits a kwarg only
/// where the desired state differs from the template's own default
/// (`ModelIdentity.templateFlagDefaults`), so a render the template would
/// produce anyway never fragments the partition, in either polarity.
nonisolated struct TemplateRenderContext: Sendable, Hashable {
    /// The template flag: render every assistant turn's think block instead
    /// of stripping turns at-or-before the last user query. Makes the render
    /// append-stable across new user messages — the **Think-Strip Rewind**
    /// cannot occur.
    static let preserveThinkingFlag = TemplateRenderFlag.preserveThinking

    /// The kwargs this render passes to the template — only entries whose
    /// value differs from the template's default render. An empty dictionary
    /// is the canonical render.
    let kwargs: [TemplateRenderFlag: Bool]

    /// The state the render actually produces for prior-turn thinking,
    /// regardless of which polarity's kwarg (if any) had to be emitted to get
    /// there. Consumers gating on render semantics (speculative seeding
    /// guards against the Think-Strip Rewind) read this, never `kwargs`.
    let preservesThinking: Bool

    /// The **Reasoning Effort** kwarg this render passes to the template
    /// (ADR-0060) — non-`nil` only when the desired level differs from the
    /// template's own default, mirroring the boolean-flag emission rule, so
    /// an explicit request for the default level keeps the canonical render
    /// and its cache partition.
    let reasoningEffort: ReasoningEffort?

    /// The default render — no kwargs, prior thinking stripped (every
    /// declared-flag template before Nanbeige4.2 strips by default). Its
    /// `digest` equals `HTTPPrefixCacheConversation.defaultTemplateContextDigest`.
    static let canonical = TemplateRenderContext(kwargs: [:], preservesThinking: false)

    init(
        kwargs: [TemplateRenderFlag: Bool],
        preservesThinking: Bool,
        reasoningEffort: ReasoningEffort? = nil
    ) {
        self.kwargs = kwargs
        self.preservesThinking = preservesThinking
        self.reasoningEffort = reasoningEffort
    }

    /// Whether this render turns the model's thinking off entirely: an emitted
    /// `enable_thinking: false` also swaps the template's generation prompt to
    /// a closed, empty think block, so the stream parser must NOT start inside
    /// `<think>` — the stream-loop's `startsInsideThinkBlock` derives from
    /// `promptStartsThinking && !disablesThinking`.
    var disablesThinking: Bool {
        kwargs[.enableThinking] == false
    }

    /// Strip-by-default convenience (the Qwen3.6 polarity): the historical
    /// shape where a present flag means `true` and preservation tracks
    /// membership. Kept for call sites and tests predating polarity.
    init(flags: Set<TemplateRenderFlag>) {
        self.init(
            kwargs: Dictionary(uniqueKeysWithValues: flags.map { ($0, true) }),
            preservesThinking: flags.contains(.preserveThinking)
        )
    }

    /// Digest over the canonical JSON form of the kwargs (sorted keys, real
    /// boolean values), matching the conversation's default digest
    /// (`digest of "{}"`) when no kwarg is emitted — so canonical requests
    /// keep their existing conversation identity and on-disk partitions. A
    /// `false` kwarg digests too: on a preserve-by-default template it is
    /// precisely the value that changes the render.
    var digest: String {
        // The canonical (no-kwargs) case is the majority of traffic and is hit
        // at least twice per request (partition key + the `PrefixCacheInput`
        // precondition, which runs in release). Its digest is the compile-time
        // constant `digest of "{}"`, so skip the dict-build + JSON-encode +
        // SHA256 entirely.
        guard !kwargs.isEmpty || reasoningEffort != nil else {
            return HTTPPrefixCacheConversation.defaultTemplateContextDigest
        }
        var object = Dictionary(
            uniqueKeysWithValues: kwargs.map { flag, value in
                (flag.rawValue, JSONValue.bool(value))
            })
        if let reasoningEffort {
            object[Self.reasoningEffortKwargName] = JSONValue.string(reasoningEffort.rawValue)
        }
        return httpPrefixCacheDigest(
            for: Data(encodeCanonicalHTTPPrefixCacheJSONObject(object).utf8)
        )
    }

    /// The template kwarg name **Reasoning Effort** rides under — the same
    /// identifier on the OpenAI wire (top-level and inside
    /// `chat_template_kwargs`) and in the Jinja template.
    static let reasoningEffortKwargName = "reasoning_effort"

    /// The kwargs merged over a render's `additionalContext`. Identity for the
    /// canonical context — callers keep passing exactly what they pass today.
    func additionalContext(
        merging base: [String: any Sendable]? = nil
    ) -> [String: any Sendable]? {
        guard !kwargs.isEmpty || reasoningEffort != nil else { return base }
        var merged = base ?? [:]
        for (flag, value) in kwargs {
            merged[flag.rawValue] = value
        }
        if let reasoningEffort {
            merged[Self.reasoningEffortKwargName] = reasoningEffort.rawValue
        }
        return merged
    }

    /// Resolve one request's render context. Per flag the precedence is:
    /// request `chat_template_kwargs` value if present, else the app's desired
    /// state (`appDesired`), else the template's own default — and only flags
    /// in `declaredFlags` participate at all, so an unsupported kwarg can
    /// neither change the render nor fragment the partition. A kwarg is
    /// emitted only where the desired state differs from the template's
    /// default (`templateDefaults`, strip-by-default when absent), so both
    /// polarities resolve to the minimal wire form. A flag with no app stance
    /// (absent from `appDesired` — `enable_thinking` has no app setting)
    /// follows the template default and is emitted only on an explicit
    /// request value.
    ///
    /// **Reasoning Effort** resolves alongside (ADR-0060): the requested
    /// level participates only when the template declares the kwarg
    /// (`declaresReasoningEffort` — introspection, never model name), and is
    /// emitted only when it differs from the template's own default level, so
    /// an explicit request for the default keeps the canonical render.
    static func resolve(
        requestKwargs: [String: Bool]?,
        appDesired: [TemplateRenderFlag: Bool],
        declaredFlags: Set<TemplateRenderFlag>,
        templateDefaults: [TemplateRenderFlag: Bool] = [:],
        requestedReasoningEffort: ReasoningEffort? = nil,
        declaresReasoningEffort: Bool = false,
        reasoningEffortTemplateDefault: ReasoningEffort? = nil
    ) -> TemplateRenderContext {
        var kwargs: [TemplateRenderFlag: Bool] = [:]
        var preservesThinking = false
        for flag in declaredFlags {
            let templateDefault = templateDefaults[flag] ?? false
            let desired = requestKwargs?[flag.rawValue] ?? appDesired[flag] ?? templateDefault
            if desired != templateDefault {
                kwargs[flag] = desired
            }
            if flag == .preserveThinking {
                preservesThinking = desired
            }
        }
        let reasoningEffort: ReasoningEffort? = {
            guard declaresReasoningEffort, let requestedReasoningEffort else { return nil }
            guard requestedReasoningEffort != reasoningEffortTemplateDefault else { return nil }
            return requestedReasoningEffort
        }()
        return TemplateRenderContext(
            kwargs: kwargs,
            preservesThinking: preservesThinking,
            reasoningEffort: reasoningEffort
        )
    }
}
