//
//  ReasoningEffortTests.swift
//  tesseractTests
//
//  **Reasoning Effort** (ADR-0060): native template-kwarg effort control for
//  effort-declaring models (Qwen3.8), the wire vocabulary mapping, the
//  render-context emission and digest rules, the `enable_thinking` flag it
//  ships alongside. The retired thinking-safeguard extension is ignored.
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

    @Test func retiredSafeguardRequestDoesNotChangeGenerationParameters() throws {
        let request = try JSONDecoder().decode(
            OpenAI.ChatCompletionRequest.self,
            from: Data(
                #"{"messages":[{"role":"user","content":"hi"}],"max_tokens":100000,"reasoning_effort":"xhigh","thinking_safeguard":{"enabled":true,"max_thinking_chars":1,"max_line_repeats":1,"injection_message":"Forced answer"}}"#
                    .utf8))
        for nativeEffort in [false, true] {
            let parameters = CompletionHandler.makeGenerateParameters(
                from: request,
                modelState: ServerInferenceModelState(
                    modelID: nativeEffort ? "qwen3.8-27b" : "qwen3.6-27b",
                    visionMode: false, declaresReasoningEffort: nativeEffort))
            #expect(parameters.maxTokens == 100_000)
            let encoded =
                try JSONSerialization.jsonObject(with: JSONEncoder().encode(parameters))
                as? [String: Any]
            #expect(encoded?["thinkingSafeguard"] == nil)
        }
        #expect(request.reasoning_effort == "xhigh")
    }
}
