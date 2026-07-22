import Foundation
import MLXLMCommon

nonisolated enum TemplateRenderFlag: String, CaseIterable, Sendable, Hashable, Codable {
    case preserveThinking = "preserve_thinking"
}

/// The resolved chat-template render context for one completion (PRD #94,
/// issue #98): which template-declared render kwargs this request renders
/// with — today the single known flag is `preserve_thinking`.
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

    /// The default render — no kwargs, prior thinking stripped (every
    /// declared-flag template before Nanbeige4.2 strips by default). Its
    /// `digest` equals `HTTPPrefixCacheConversation.defaultTemplateContextDigest`.
    static let canonical = TemplateRenderContext(kwargs: [:], preservesThinking: false)

    init(kwargs: [TemplateRenderFlag: Bool], preservesThinking: Bool) {
        self.kwargs = kwargs
        self.preservesThinking = preservesThinking
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
        guard !kwargs.isEmpty else {
            return HTTPPrefixCacheConversation.defaultTemplateContextDigest
        }
        let object = Dictionary(
            uniqueKeysWithValues: kwargs.map { flag, value in
                (flag.rawValue, JSONValue.bool(value))
            })
        return httpPrefixCacheDigest(
            for: Data(encodeCanonicalHTTPPrefixCacheJSONObject(object).utf8)
        )
    }

    /// The kwargs merged over a render's `additionalContext`. Identity for the
    /// canonical context — callers keep passing exactly what they pass today.
    func additionalContext(
        merging base: [String: any Sendable]? = nil
    ) -> [String: any Sendable]? {
        guard !kwargs.isEmpty else { return base }
        var merged = base ?? [:]
        for (flag, value) in kwargs {
            merged[flag.rawValue] = value
        }
        return merged
    }

    /// Resolve one request's render context. Per flag the precedence is:
    /// request `chat_template_kwargs` value if present, else whether an app
    /// setting enables it — and only flags in `declaredFlags` participate at
    /// all, so an unsupported kwarg can neither change the render nor
    /// fragment the partition. A kwarg is emitted only where the desired
    /// state differs from the template's default (`templateDefaults`,
    /// strip-by-default when absent), so both polarities resolve to the
    /// minimal wire form.
    static func resolve(
        requestKwargs: [String: Bool]?,
        appEnabledFlags: Set<TemplateRenderFlag>,
        declaredFlags: Set<TemplateRenderFlag>,
        templateDefaults: [TemplateRenderFlag: Bool] = [:]
    ) -> TemplateRenderContext {
        var kwargs: [TemplateRenderFlag: Bool] = [:]
        var preservesThinking = false
        for flag in declaredFlags {
            let desired = requestKwargs?[flag.rawValue] ?? appEnabledFlags.contains(flag)
            if desired != (templateDefaults[flag] ?? false) {
                kwargs[flag] = desired
            }
            if flag == .preserveThinking {
                preservesThinking = desired
            }
        }
        return TemplateRenderContext(kwargs: kwargs, preservesThinking: preservesThinking)
    }
}
