//
//  PreserveThinkingRenderTests.swift
//  tesseractTests
//
//  The **Preserve-Thinking Render** (PRD #94, issue #98): an opt-in,
//  template-declared render mode under which every assistant turn keeps its
//  think block, making the render append-stable — the **Think-Strip Rewind**
//  cannot occur. Covered at the pure seams: template introspection on
//  `ModelIdentity`, request/setting resolution on `TemplateRenderContext`,
//  the digest → partition fold on `CachePartitionKey`/`PartitionMeta`, the
//  lenient `chat_template_kwargs` decode, the speculative-seed self-skip,
//  and the append-stability property itself on the fake PARO template.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct PreserveThinkingRenderTests {

    private let preserveFlag = TemplateRenderContext.preserveThinkingFlag

    // MARK: - Template introspection (ModelIdentity)

    @Test func templateDeclaringTheFlagSurfacesIt() {
        let identity = ModelIdentity(
            configJSON: nil,
            chatTemplate: "{%- if preserve_thinking %}...{%- endif %}"
        )
        #expect(identity.declaredTemplateFlags == [preserveFlag])
    }

    @Test func templateWithoutTheFlagDeclaresNothing() {
        // The Qwen3.5-PARO shape: a thinking template with no opt-in flag.
        let identity = ModelIdentity(
            configJSON: nil,
            chatTemplate: "{%- if add_generation_prompt %}<think>{%- endif %}"
        )
        #expect(identity.declaredTemplateFlags.isEmpty)
        #expect(
            ModelIdentity(configJSON: nil, chatTemplate: nil)
                .declaredTemplateFlags.isEmpty)
    }

    // MARK: - Resolution (request wins, setting falls back, allowlist gates)

    @Test func settingEnablesTheFlagAndRequestOverridesIt() {
        let declared: Set<TemplateRenderFlag> = [preserveFlag]

        let fromSetting = TemplateRenderContext.resolve(
            requestKwargs: nil,
            appEnabledFlags: [preserveFlag],
            declaredFlags: declared
        )
        #expect(fromSetting.preservesThinking)

        let requestOff = TemplateRenderContext.resolve(
            requestKwargs: [preserveFlag.rawValue: false],
            appEnabledFlags: [preserveFlag],
            declaredFlags: declared
        )
        #expect(requestOff == .canonical)

        let requestOn = TemplateRenderContext.resolve(
            requestKwargs: [preserveFlag.rawValue: true],
            appEnabledFlags: [],
            declaredFlags: declared
        )
        #expect(requestOn.preservesThinking)
    }

    @Test func preserveThinkingDefaultsOnAndGatesOnDeclaration() {
        // #237: the per-model setting now defaults on. The blanket `true`
        // must be safe — it renders preserve only where the template declares
        // the flag, and canonical everywhere else.
        #expect(SettingsCatalogue.preserveThinkingRender(modelID: "anything").default)

        let appEnabled: Set<TemplateRenderFlag> =
            SettingsCatalogue.preserveThinkingRender(modelID: "m").default ? [preserveFlag] : []

        // Declaring model under the default → preserve.
        #expect(
            TemplateRenderContext.resolve(
                requestKwargs: nil,
                appEnabledFlags: appEnabled,
                declaredFlags: [preserveFlag]
            ).preservesThinking)

        // Non-declaring model under the same default → still canonical.
        #expect(
            TemplateRenderContext.resolve(
                requestKwargs: nil,
                appEnabledFlags: appEnabled,
                declaredFlags: []
            ) == .canonical)
    }

    @Test func undeclaredFlagsAreIgnoredWithoutFragmentingThePartition() {
        // Qwen3.5-PARO receiving preserve_thinking, plus an arbitrary kwarg:
        // neither may change the render nor the digest.
        let resolved = TemplateRenderContext.resolve(
            requestKwargs: [preserveFlag.rawValue: true, "bogus_flag": true],
            appEnabledFlags: [preserveFlag],
            declaredFlags: []
        )
        #expect(resolved == .canonical)
        #expect(resolved.digest == HTTPPrefixCacheConversation.defaultTemplateContextDigest)
    }

    // MARK: - Digest and render-context plumbing

    @Test func canonicalDigestMatchesTheConversationDefault() {
        #expect(
            TemplateRenderContext.canonical.digest
                == HTTPPrefixCacheConversation.defaultTemplateContextDigest)
        let preserve = TemplateRenderContext(flags: [preserveFlag])
        #expect(
            preserve.digest
                != HTTPPrefixCacheConversation.defaultTemplateContextDigest)
    }

    @Test func additionalContextMergesFlagsOverTheBase() {
        #expect(TemplateRenderContext.canonical.additionalContext() == nil)
        let base = TemplateRenderContext.canonical
            .additionalContext(merging: ["add_generation_prompt": false])
        #expect(base?.count == 1)

        let preserve = TemplateRenderContext(flags: [preserveFlag])
        let merged = preserve.additionalContext(
            merging: ["add_generation_prompt": false]
        )
        #expect(merged?[preserveFlag.rawValue] as? Bool == true)
        #expect(merged?["add_generation_prompt"] as? Bool == false)
    }

    // MARK: - Partition fold (CachePartitionKey + PartitionMeta)

    @Test func canonicalKeyKeepsItsPreFieldPartitionDigest() {
        // Back-compat contract: partitions persisted before the field
        // existed must stay reachable, so the default digest adds nothing
        // to the canonical form.
        let bare = CachePartitionKey(modelID: "m", kvBits: 8, kvGroupSize: 64)
        let explicit = CachePartitionKey(
            modelID: "m", kvBits: 8, kvGroupSize: 64,
            templateContextDigest: HTTPPrefixCacheConversation.defaultTemplateContextDigest
        )
        #expect(bare.partitionDigest == explicit.partitionDigest)
        #expect(bare == explicit)
    }

    @Test func preserveModeLandsInAFreshPartition() {
        let canonical = CachePartitionKey(modelID: "m", kvBits: 8, kvGroupSize: 64)
        let preserve = CachePartitionKey(
            modelID: "m", kvBits: 8, kvGroupSize: 64,
            templateContextDigest: TemplateRenderContext(flags: [preserveFlag]).digest
        )
        #expect(canonical != preserve)
        #expect(canonical.partitionDigest != preserve.partitionDigest)
    }

    @Test func partitionMetaRoundTripsTheDigestAndOldSidecarsStillDecode() throws {
        let preserveDigest = TemplateRenderContext(flags: [preserveFlag]).digest
        let meta = PartitionMeta(
            modelID: "m", modelFingerprint: "f", kvBits: nil, kvGroupSize: 64,
            createdAt: 0, schemaVersion: SnapshotManifestSchema.currentVersion,
            templateContextDigest: preserveDigest
        )
        let decoded = try JSONDecoder().decode(
            PartitionMeta.self, from: JSONEncoder().encode(meta)
        )
        #expect(decoded.templateContextDigest == preserveDigest)

        // A pre-field sidecar (no key) decodes to nil — the canonical render.
        let legacyJSON = """
            {"modelID":"m","modelFingerprint":"f","kvGroupSize":64,
             "createdAt":0,"schemaVersion":\(SnapshotManifestSchema.currentVersion)}
            """
        let legacy = try JSONDecoder().decode(
            PartitionMeta.self, from: Data(legacyJSON.utf8)
        )
        #expect(legacy.templateContextDigest == nil)
    }

    // MARK: - chat_template_kwargs lenient decode

    @Test func kwargsDecodeKeepsBoolsAndDropsEverythingElse() throws {
        let json = """
            {"preserve_thinking": true, "effort": "high", "depth": 3,
             "nested": {"a": 1}, "off": false}
            """
        let kwargs = try JSONDecoder().decode(
            OpenAI.ChatTemplateKwargs.self, from: Data(json.utf8)
        )
        #expect(kwargs.booleanFlags == ["preserve_thinking": true, "off": false])
    }

    // MARK: - Speculative seed self-skip

    @Test func speculativeSeedingSelfSkipsUnderPreserveThinking() {
        let preserve = TemplateRenderContext(flags: [preserveFlag])
        #expect(
            LeafStorePhase.speculativeSeedPlan(
                boundaryMode: .canonical, renderContext: .canonical
            ) != nil)
        #expect(
            LeafStorePhase.speculativeSeedPlan(
                boundaryMode: .canonical, renderContext: preserve
            ) == nil)
        #expect(
            LeafStorePhase.speculativeSeedPlan(
                boundaryMode: .directTool, renderContext: preserve
            ) == nil)
    }

    // MARK: - Append stability (the no-rewind property, on the fake template)

    private let tokenizer = FakeParoThinkingTokenizer()

    /// A finished **Tool Stretch**: question → tool-calling turn → tool
    /// result → stop answer. The shape whose next user message triggers the
    /// Think-Strip Rewind under the canonical render.
    private var stretchMessages: [[String: any Sendable]] {
        [
            ["role": "user", "content": "What is in the file?"],
            [
                "role": "assistant", "content": "",
                "reasoning_content": "Let me read it first.",
                "tool_calls": [
                    [
                        "function": [
                            "name": "read",
                            "arguments": ["filePath": "/tmp/answer.txt"],
                        ] as [String: any Sendable]
                    ]
                ] as [[String: any Sendable]],
            ],
            ["role": "tool", "content": "42"],
            [
                "role": "assistant", "content": "The file contains 42.",
                "reasoning_content": "The tool returned 42.",
            ],
        ]
    }

    @Test func preserveRenderIsAppendStableWhereCanonicalRewinds() throws {
        let steering: [String: any Sendable] = [
            "role": "user", "content": "Now compare it with the other file.",
        ]
        func render(_ messages: [[String: any Sendable]], preserve: Bool) throws -> [Int] {
            var context: [String: any Sendable] = ["add_generation_prompt": false]
            if preserve { context[preserveFlag.rawValue] = true }
            return try tokenizer.applyChatTemplate(
                messages: messages, tools: nil, additionalContext: context
            )
        }

        // Canonical render: appending the user message re-renders the whole
        // stretch think-stripped — the old render forks early (the rewind).
        let strippedBase = try render(stretchMessages, preserve: false)
        let strippedNext = try render(stretchMessages + [steering], preserve: false)
        let strippedShared = zip(strippedBase, strippedNext).prefix(while: ==).count
        #expect(strippedShared < strippedBase.count)

        // Preserve-Thinking Render: the old render is a byte-identical
        // prefix of the new one — a near-full prefix hit, no rewind.
        let preservedBase = try render(stretchMessages, preserve: true)
        let preservedNext = try render(stretchMessages + [steering], preserve: true)
        #expect(preservedNext.count > preservedBase.count)
        #expect(Array(preservedNext.prefix(preservedBase.count)) == preservedBase)
        // And the preserved next-request render genuinely keeps the
        // stretch's thinking the canonical render strips. (The two BASE
        // renders agree — the current turn keeps its thinking either way.)
        #expect(preservedNext.count > strippedNext.count)
    }
}
