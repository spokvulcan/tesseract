import Foundation
import MLXLMCommon

nonisolated enum TemplateRenderFlag: String, CaseIterable, Sendable, Hashable, Codable {
    case preserveThinking = "preserve_thinking"
}

/// The resolved chat-template render context for one server completion
/// (PRD #94, issue #98): which template-declared, opt-in render flags this
/// request renders with — today the single known flag is Qwen3.6's
/// `preserve_thinking` (the **Preserve-Thinking Render**).
///
/// One value is resolved per request by the completion handler — request-level
/// `chat_template_kwargs` wins per flag, the per-model app setting is the
/// fallback, and flags the loaded template does not declare are ignored
/// entirely (capability gate by template introspection, never model name;
/// see `ModelIdentity.declaredTemplateFlags`). The value then rides the whole
/// completion: every template render in the pipeline merges `kwargs` into its
/// `additionalContext`, and `digest` folds the flags into the conversation's
/// template-context digest and the cache partition — so toggling a flag lands
/// in a fresh partition and mixed renders can never share one.
nonisolated struct TemplateRenderContext: Sendable, Hashable {
    /// Qwen3.6's template flag: render every assistant turn's think block
    /// instead of stripping turns at-or-before the last user query. Makes the
    /// render append-stable across new user messages — the **Think-Strip
    /// Rewind** cannot occur.
    static let preserveThinkingFlag = TemplateRenderFlag.preserveThinking

    /// The effective-true template flags, already allowlisted against the
    /// template's declared flags. Off flags are not represented: a flag set
    /// to `false` equals the template's default render, so folding it would
    /// fragment the partition without changing a single token.
    let flags: Set<TemplateRenderFlag>

    /// The default render — no opt-in flags, the template's own semantics.
    /// Its `digest` equals `HTTPPrefixCacheConversation.defaultTemplateContextDigest`.
    static let canonical = TemplateRenderContext(flags: [])

    var preservesThinking: Bool { flags.contains(.preserveThinking) }

    /// Digest over the canonical JSON form of the flags (sorted keys,
    /// `true` values only), matching the conversation's default digest
    /// (`digest of "{}"`) when no flag is set — so canonical requests keep
    /// their existing conversation identity and on-disk partitions.
    var digest: String {
        let object = Dictionary(uniqueKeysWithValues: flags.map { flag in
            (flag.rawValue, JSONValue.bool(true))
        })
        return httpPrefixCacheDigest(
            for: Data(encodeCanonicalHTTPPrefixCacheJSONObject(object).utf8)
        )
    }

    /// The flags merged over a render's `additionalContext`. Identity for the
    /// canonical context — callers keep passing exactly what they pass today.
    func additionalContext(
        merging base: [String: any Sendable]? = nil
    ) -> [String: any Sendable]? {
        guard !flags.isEmpty else { return base }
        var merged = base ?? [:]
        for flag in flags {
            merged[flag.rawValue] = true
        }
        return merged
    }

    /// Resolve one request's render context. Per flag the precedence is:
    /// request `chat_template_kwargs` value if present, else whether an app
    /// setting enables it — and only flags in `declaredFlags` participate at
    /// all, so an unsupported kwarg can neither change the render nor
    /// fragment the partition.
    static func resolve(
        requestKwargs: [String: Bool]?,
        appEnabledFlags: Set<TemplateRenderFlag>,
        declaredFlags: Set<TemplateRenderFlag>
    ) -> TemplateRenderContext {
        var effective: Set<TemplateRenderFlag> = []
        for flag in declaredFlags {
            let enabled = requestKwargs?[flag.rawValue] ?? appEnabledFlags.contains(flag)
            if enabled {
                effective.insert(flag)
            }
        }
        return TemplateRenderContext(flags: effective)
    }
}
