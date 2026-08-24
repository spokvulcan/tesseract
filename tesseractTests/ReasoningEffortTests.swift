//
//  ReasoningEffortTests.swift
//  tesseractTests
//
//  **Reasoning Effort** (ADR-0060): native template-kwarg effort control for
//  effort-declaring models (Qwen3.8), the wire vocabulary mapping, the
//  render-context emission and digest rules, the `enable_thinking` flag it
//  ships alongside, and the thinking-safeguard budget split — the legacy
//  cutoff for non-native models, the fixed anti-runaway ceiling for native
//  ones, repetition triggers untouched for all.
//

import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

@MainActor
struct ReasoningEffortTests {

    /// The load-bearing fragments of the real Qwen3.8 chat template: the
    /// effort gate with its default, the preserve-by-default shape, and the
    /// thinking-off generation prompt.
    private let qwen38Template = """
        {%- set reasoning_instructions = '' %}
        {%- if enable_thinking is undefined or enable_thinking is true %}
            {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
            {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
                {{- raise_exception('Unexpected reasoning effort') }}
            {%- endif %}
        {%- endif %}
        {%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}
        {%- endif %}
        {%- if add_generation_prompt %}
            {{- '<|im_start|>assistant\\n' }}
            {%- if enable_thinking is defined and enable_thinking is false %}
                {{- '<think>\\n\\n</think>\\n\\n' }}
            {%- else %}
                {{- '<think>\\n' }}
            {%- endif %}
        {%- endif %}
        """

    // MARK: - Template introspection (ModelIdentity)

    @Test func qwen38TemplateDeclaresEffortWithXhighDefault() {
        let identity = ModelIdentity(configJSON: nil, chatTemplate: qwen38Template)
        #expect(identity.declaresReasoningEffort)
        #expect(identity.reasoningEffortTemplateDefault == .xhigh)
    }

    @Test func earlierTemplatesDoNotDeclareEffort() {
        let identity = ModelIdentity(
            configJSON: nil,
            chatTemplate: "{%- if enable_thinking is defined and enable_thinking is false %}"
        )
        #expect(!identity.declaresReasoningEffort)
        #expect(identity.reasoningEffortTemplateDefault == nil)
        #expect(!ModelIdentity(configJSON: nil, chatTemplate: nil).declaresReasoningEffort)
    }

    @Test func effortMentionedOnlyInACommentDoesNotDeclare() {
        let identity = ModelIdentity(
            configJSON: nil,
            chatTemplate: "{# reasoning_effort would go here #}{%- if x %}...{%- endif %}"
        )
        #expect(!identity.declaresReasoningEffort)
    }

    @Test func qwen38ShapePreserveThinkingDefaultsToPreserve() {
        // The `is undefined or … is true` shape: absent kwarg means preserve.
        // Before ADR-0060 the heuristic read this as strip-by-default and the
        // server emitted a spurious `preserve_thinking: true`, fragmenting
        // the cache partition off the canonical digest.
        let identity = ModelIdentity(configJSON: nil, chatTemplate: qwen38Template)
        #expect(identity.templateFlagDefaults[.preserveThinking] == true)
    }

    @Test func qwen38TemplateDeclaresEnableThinkingDefaultOn() {
        let identity = ModelIdentity(configJSON: nil, chatTemplate: qwen38Template)
        #expect(identity.declaredTemplateFlags.contains(.enableThinking))
        #expect(identity.templateFlagDefaults[.enableThinking] == true)
    }

    @Test func fleetEnableThinkingShapeDefaultsOn() {
        // Every current thinking template gates on an explicit `false`.
        let identity = ModelIdentity(
            configJSON: nil,
            chatTemplate: "{%- if enable_thinking is defined and enable_thinking is false %}"
        )
        #expect(identity.templateFlagDefaults[.enableThinking] == true)
    }

    // MARK: - Wire vocabulary (union of OpenAI and Qwen levels)

    @Test func wireVocabularyMapsToNativeLevels() {
        #expect(OpenAI.nativeReasoningEffort(fromWire: "minimal") == .low)
        #expect(OpenAI.nativeReasoningEffort(fromWire: "low") == .low)
        #expect(OpenAI.nativeReasoningEffort(fromWire: "medium") == .medium)
        #expect(OpenAI.nativeReasoningEffort(fromWire: "high") == .xhigh)
        #expect(OpenAI.nativeReasoningEffort(fromWire: "xhigh") == .xhigh)
        #expect(OpenAI.nativeReasoningEffort(fromWire: "none") == nil)
        #expect(OpenAI.nativeReasoningEffort(fromWire: "turbo") == nil)
        #expect(OpenAI.nativeReasoningEffort(fromWire: "") == nil)
    }

    @Test func chatTemplateKwargsCarryStringValues() throws {
        let json = """
            {"messages": [{"role": "user", "content": "hi"}],
             "chat_template_kwargs": {
                "enable_thinking": true, "reasoning_effort": "low",
                "depth": 3, "nested": {"a": 1}}}
            """
        let request = try JSONDecoder().decode(
            OpenAI.ChatCompletionRequest.self, from: Data(json.utf8))
        #expect(request.chat_template_kwargs?.booleanFlags == ["enable_thinking": true])
        #expect(request.chat_template_kwargs?.stringValues == ["reasoning_effort": "low"])
    }

    @Test func kwargsChannelWinsOverTopLevelField() throws {
        let json = """
            {"messages": [{"role": "user", "content": "hi"}],
             "reasoning_effort": "high",
             "chat_template_kwargs": {"reasoning_effort": "low"}}
            """
        let request = try JSONDecoder().decode(
            OpenAI.ChatCompletionRequest.self, from: Data(json.utf8))
        #expect(CompletionHandler.requestedReasoningEffortRaw(request) == "low")
    }

    @Test func topLevelFieldAloneIsRead() throws {
        let json = """
            {"messages": [{"role": "user", "content": "hi"}],
             "reasoning_effort": "medium"}
            """
        let request = try JSONDecoder().decode(
            OpenAI.ChatCompletionRequest.self, from: Data(json.utf8))
        #expect(CompletionHandler.requestedReasoningEffortRaw(request) == "medium")
    }

    @Test func validationSeesBothChannelsEvenWhenPrecedenceIgnoresOne() throws {
        // An unknown value on the losing channel must still be visible to the
        // pre-lease vocabulary check — precedence picks the first value, but
        // validation sweeps every value present on the wire.
        let json = """
            {"messages": [{"role": "user", "content": "hi"}],
             "reasoning_effort": "banana",
             "chat_template_kwargs": {"reasoning_effort": "low"}}
            """
        let request = try JSONDecoder().decode(
            OpenAI.ChatCompletionRequest.self, from: Data(json.utf8))
        let values = CompletionHandler.requestedReasoningEffortRawValues(request)
        #expect(values == ["low", "banana"])
        #expect(CompletionHandler.requestedReasoningEffortRaw(request) == "low")
        #expect(
            values.first(where: { OpenAI.nativeReasoningEffort(fromWire: $0) == nil })
                == "banana")
    }

    // MARK: - Render-context resolution and digest

    @Test func effortEqualToTemplateDefaultResolvesCanonical() {
        // Explicit xhigh on a template defaulting to xhigh: no kwarg, the
        // canonical digest, existing partitions.
        let resolved = TemplateRenderContext.resolve(
            requestKwargs: nil,
            appDesired: [:],
            declaredFlags: [],
            requestedReasoningEffort: .xhigh,
            declaresReasoningEffort: true,
            reasoningEffortTemplateDefault: .xhigh
        )
        #expect(resolved.reasoningEffort == nil)
        #expect(resolved.digest == HTTPPrefixCacheConversation.defaultTemplateContextDigest)
    }

    @Test func effortDifferingFromDefaultIsEmittedAndFragments() {
        let resolved = TemplateRenderContext.resolve(
            requestKwargs: nil,
            appDesired: [:],
            declaredFlags: [],
            requestedReasoningEffort: .medium,
            declaresReasoningEffort: true,
            reasoningEffortTemplateDefault: .xhigh
        )
        #expect(resolved.reasoningEffort == .medium)
        #expect(resolved.digest != HTTPPrefixCacheConversation.defaultTemplateContextDigest)
        #expect(
            resolved.additionalContext()?[TemplateRenderContext.reasoningEffortKwargName]
                as? String == "medium")
    }

    @Test func effortOnANonDeclaringModelIsIgnored() {
        let resolved = TemplateRenderContext.resolve(
            requestKwargs: nil,
            appDesired: [:],
            declaredFlags: [],
            requestedReasoningEffort: .low,
            declaresReasoningEffort: false,
            reasoningEffortTemplateDefault: nil
        )
        #expect(resolved == .canonical)
        #expect(resolved.digest == HTTPPrefixCacheConversation.defaultTemplateContextDigest)
    }

    @Test func unknownTemplateDefaultEmitsTheRequestedLevel() {
        // An unparseable default means we cannot prove the request matches
        // the template's own render — emit explicitly.
        let resolved = TemplateRenderContext.resolve(
            requestKwargs: nil,
            appDesired: [:],
            declaredFlags: [],
            requestedReasoningEffort: .xhigh,
            declaresReasoningEffort: true,
            reasoningEffortTemplateDefault: nil
        )
        #expect(resolved.reasoningEffort == .xhigh)
    }

    @Test func distinctLevelsLandInDistinctPartitions() {
        let low = TemplateRenderContext(
            kwargs: [:], preservesThinking: false, reasoningEffort: .low)
        let medium = TemplateRenderContext(
            kwargs: [:], preservesThinking: false, reasoningEffort: .medium)
        #expect(low.digest != medium.digest)
        #expect(low.digest != HTTPPrefixCacheConversation.defaultTemplateContextDigest)
    }

    @Test func effortCombinesWithFlagKwargsInTheDigest() {
        let effortOnly = TemplateRenderContext(
            kwargs: [:], preservesThinking: false, reasoningEffort: .low)
        let both = TemplateRenderContext(
            kwargs: [.preserveThinking: false], preservesThinking: false, reasoningEffort: .low)
        #expect(effortOnly.digest != both.digest)
        #expect(
            both.additionalContext()?["preserve_thinking"] as? Bool == false)
        #expect(
            both.additionalContext()?[TemplateRenderContext.reasoningEffortKwargName]
                as? String == "low")
    }

    // MARK: - enable_thinking (the sanctioned thinking-off switch)

    @Test func enableThinkingFollowsTheTemplateDefaultWhenUnrequested() {
        // No app setting exists for enable_thinking: with no request value it
        // must never be emitted, on either polarity.
        for templateDefault in [true, false] {
            let resolved = TemplateRenderContext.resolve(
                requestKwargs: nil,
                appDesired: [:],
                declaredFlags: [.enableThinking],
                templateDefaults: [.enableThinking: templateDefault]
            )
            #expect(resolved.kwargs.isEmpty)
            #expect(!resolved.disablesThinking)
        }
    }

    @Test func requestCanDisableThinking() {
        let resolved = TemplateRenderContext.resolve(
            requestKwargs: ["enable_thinking": false],
            appDesired: [:],
            declaredFlags: [.enableThinking],
            templateDefaults: [.enableThinking: true]
        )
        #expect(resolved.kwargs == [.enableThinking: false])
        #expect(resolved.disablesThinking)
        #expect(resolved.digest != HTTPPrefixCacheConversation.defaultTemplateContextDigest)
    }

    // MARK: - Safeguard budget split (ADR-0060)

    @Test func budgetPolicySplitsNativeFromLegacy() {
        var config = ThinkingRepetitionDetector.Config()

        config.applyThinkingBudgetPolicy(
            nativeReasoningEffort: true, cutoffEnabled: true, cutoffChars: 4_096)
        #expect(
            config.maxThinkingChars
                == ThinkingRepetitionDetector.Config.nativeReasoningEffortBudgetChars)

        config.applyThinkingBudgetPolicy(
            nativeReasoningEffort: false, cutoffEnabled: true, cutoffChars: 4_096)
        #expect(config.maxThinkingChars == 4_096)

        config.applyThinkingBudgetPolicy(
            nativeReasoningEffort: false, cutoffEnabled: false, cutoffChars: 4_096)
        #expect(config.maxThinkingChars == nil)
        // The repetition triggers are not the budget's business.
        #expect(config.enabled)
        #expect(config.maxLineRepeats == 6)
    }

    @Test func serverParametersApplyThePolicyBeforeVendorOverrides() throws {
        let base = """
            {"messages": [{"role": "user", "content": "hi"}]}
            """
        let request = try JSONDecoder().decode(
            OpenAI.ChatCompletionRequest.self, from: Data(base.utf8))

        let nativeState = ServerInferenceModelState(
            modelID: "qwen3.8-27b", visionMode: false,
            declaresReasoningEffort: true, reasoningEffortTemplateDefault: .xhigh)
        let native = CompletionHandler.makeGenerateParameters(
            from: request, modelState: nativeState,
            thinkingCutoffEnabled: true, thinkingCutoffChars: 4_096)
        #expect(
            native.thinkingSafeguard.maxThinkingChars
                == ThinkingRepetitionDetector.Config.nativeReasoningEffortBudgetChars)

        let legacyState = ServerInferenceModelState(modelID: "qwen3.6-27b", visionMode: false)
        let legacy = CompletionHandler.makeGenerateParameters(
            from: request, modelState: legacyState,
            thinkingCutoffEnabled: true, thinkingCutoffChars: 4_096)
        #expect(legacy.thinkingSafeguard.maxThinkingChars == 4_096)

        let off = CompletionHandler.makeGenerateParameters(
            from: request, modelState: legacyState,
            thinkingCutoffEnabled: false, thinkingCutoffChars: 4_096)
        #expect(off.thinkingSafeguard.maxThinkingChars == nil)

        // The per-request vendor extension stays authoritative over the policy.
        let withOverride = """
            {"messages": [{"role": "user", "content": "hi"}],
             "thinking_safeguard": {"max_thinking_chars": 300}}
            """
        let overrideRequest = try JSONDecoder().decode(
            OpenAI.ChatCompletionRequest.self, from: Data(withOverride.utf8))
        let overridden = CompletionHandler.makeGenerateParameters(
            from: overrideRequest, modelState: nativeState,
            thinkingCutoffEnabled: true, thinkingCutoffChars: 4_096)
        #expect(overridden.thinkingSafeguard.maxThinkingChars == 300)
    }

    @Test func budgetTriggerIgnoresTheRepetitionGrace() {
        // A cutoff configured below the grace period still cuts at the
        // configured length — the budget is its own absolute threshold.
        let detector = ThinkingRepetitionDetector(
            config: .init(maxThinkingChars: 300, minCharsBeforeIntervention: 8_192))
        let filler = String(repeating: "reasoning step by step here\n", count: 20)
        let decision = detector.ingest(chunk: filler)
        guard case .intervene(let reason, _) = decision else {
            Issue.record("expected budget intervention, got \(decision)")
            return
        }
        #expect(reason == .budgetExceeded)
    }

    @Test func repetitionRewindWinsWhenBudgetCrossesInTheSameChunk() {
        // Past the grace, the budget stays the *last* trigger checked: a loop
        // that crosses both the n-gram threshold and the budget in one chunk
        // must intervene as repetition (rewinding past the looped content),
        // not as a budget cut that keeps it.
        let config = ThinkingRepetitionDetector.Config(
            enabled: true,
            minLineLength: 9999,  // line signal off
            maxLineRepeats: 999,
            ngramSize: 20,
            maxNgramRepeats: 5,
            windowChars: 2_000,
            maxThinkingChars: 100,
            minCharsBeforeIntervention: 0
        )
        let detector = ThinkingRepetitionDetector(config: config)
        // 6 × 20 chars = 120: over the 100-char budget AND 6 identical
        // 20-char ngrams, above the 5-repeat threshold.
        let chunk = String(repeating: "abcdefghijklmnopqrst", count: 6)
        let decision = detector.ingest(chunk: chunk)
        guard case .intervene(let reason, let safe) = decision else {
            Issue.record("expected an intervention, got \(decision)")
            return
        }
        #expect(reason == .duplicateNgram)
        #expect(safe.isEmpty)  // rewound to before the loop's first occurrence
    }
}
