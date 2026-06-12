import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import Tesseract_Agent

/// Behavior of the **Leaf Admission Builder**: the GPU-free reusable-prefix
/// probe and the `plan()` routing decision (`.fromBoundary` / `.skip`) that
/// finds the shared token path a future continuation can hydrate. Driven by a
/// byte-level fake tokenizer and a pure `resolveBoundary` closure — no model.
@Suite struct LeafAdmissionBuilderTests {

    private let tokenizer = FakeChatMLTokenizer()

    private func conversation(
        systemPrompt: String? = "You are helpful.",
        messages: [HTTPPrefixCacheMessage]
    ) -> HTTPPrefixCacheConversation {
        HTTPPrefixCacheConversation(systemPrompt: systemPrompt, messages: messages)
    }

    private func render(_ conversation: HTTPPrefixCacheConversation) throws -> [Int] {
        try tokenizer.applyChatTemplate(
            messages: conversation.promptMessages,
            tools: nil,
            additionalContext: ["add_generation_prompt": false]
        )
    }

    @Test func userTurnProbeReturnsTheStoredRenderWithoutTheProbeContinuation() throws {
        let stored = conversation(messages: [
            HTTPPrefixCacheMessage(role: .user, content: "question"),
            HTTPPrefixCacheMessage(role: .assistant, content: "answer"),
        ])
        let prefix = try LeafAdmissionBuilder.reusablePrefix(
            continuation: .userTurn,
            storedConversation: stored,
            toolSpecs: nil,
            tokenizer: tokenizer,
            keySpace: .identity()
        )
        // The reusable prefix is exactly the stored turn's own render — the
        // synthetic user continuation is excluded.
        #expect(try prefix?.get() == (try render(stored)))
    }

    @Test func toolResultProbeReturnsTheStoredRenderWithoutTheProbeContinuation() throws {
        let stored = conversation(messages: [
            HTTPPrefixCacheMessage(role: .user, content: "call the tool"),
            HTTPPrefixCacheMessage(role: .assistant, content: "calling"),
        ])
        let prefix = try LeafAdmissionBuilder.reusablePrefix(
            continuation: .toolResult,
            storedConversation: stored,
            toolSpecs: nil,
            tokenizer: tokenizer,
            keySpace: .identity()
        )
        #expect(try prefix?.get() == (try render(stored)))
    }

    @Test func userTurnProbeTranslatesTheImageRenderIntoKeySpace() throws {
        let stored = conversation(messages: [
            HTTPPrefixCacheMessage(
                role: .user, content: "describe", images: [HTTPPrefixCacheImage(data: Data([0x0A]))]
            ),
            HTTPPrefixCacheMessage(role: .assistant, content: "a diagram"),
        ])
        let keySpace = try FakeChatMLTokenizer.keySpace(for: stored, runLengths: [4])
        let prefix = try LeafAdmissionBuilder.reusablePrefix(
            continuation: .userTurn,
            storedConversation: stored,
            toolSpecs: nil,
            tokenizer: tokenizer,
            keySpace: keySpace
        )
        // The probe runs in render space (the stored turn's own render, as in
        // the text-only probes) and the shared prefix comes back translated:
        // the single pad becomes the image's 4-token pseudo-expansion.
        let renderTokens = try render(stored)
        let translated = try #require(try prefix?.get())
        #expect(translated == (try keySpace.translate(renderTokens: renderTokens).get()))
        #expect(translated.count == renderTokens.count + 3)
        #expect(translated.contains { $0 < 0 })
    }

    @Test func probeDivergesToNilWhenThereIsNoCommonPrefix() throws {
        // An empty conversation renders to nothing, so the stored render and the
        // probe-extended render share no prefix.
        let empty = HTTPPrefixCacheConversation(systemPrompt: nil, messages: [])
        let prefix = try LeafAdmissionBuilder.reusablePrefix(
            continuation: .userTurn,
            storedConversation: empty,
            toolSpecs: nil,
            tokenizer: tokenizer,
            keySpace: .identity()
        )
        #expect(prefix == nil)
    }

    // MARK: - futureSharedPrefix — the speculative canonical target path

    @Test func futureSharedPrefixEndsAtTheNextUserTurnHeader() throws {
        let stored = conversation(messages: [
            HTTPPrefixCacheMessage(role: .user, content: "question"),
            HTTPPrefixCacheMessage(role: .assistant, content: "answer"),
        ])
        let future = try LeafAdmissionBuilder.futureSharedPrefix(
            storedConversation: stored,
            toolSpecs: nil,
            tokenizer: tokenizer,
            keySpace: .identity()
        )
        // The two probe contents diverge at their first character, so the
        // shared path is the stored render plus exactly the next user turn's
        // header — no probe content can leak into it.
        let header = tokenizer.encode(text: "<|im_start|>user\n", addSpecialTokens: false)
        #expect(try future?.get() == (try render(stored)) + header)
    }

    @Test func futureSharedPrefixCoversTheThinkStripRewindSpan() throws {
        // A thinking template keeps the assistant's <think> in the stored
        // render (it is after the last real user message) and strips it the
        // moment any new user message lands — the Think-Strip Rewind.
        let strippingTokenizer = FakeChatMLTokenizer(stripsThinkBeforeLastUser: true)
        let stored = conversation(messages: [
            HTTPPrefixCacheMessage(role: .user, content: "question"),
            HTTPPrefixCacheMessage(
                role: .assistant, content: "<think>long reasoning</think>\nanswer"
            ),
        ])

        let canonical = try #require(try LeafAdmissionBuilder.reusablePrefix(
            continuation: .userTurn,
            storedConversation: stored,
            toolSpecs: nil,
            tokenizer: strippingTokenizer,
            keySpace: .identity()
        )?.get())
        let future = try #require(try LeafAdmissionBuilder.futureSharedPrefix(
            storedConversation: stored,
            toolSpecs: nil,
            tokenizer: strippingTokenizer,
            keySpace: .identity()
        )?.get())

        // The canonical leaf path stops at the strip divergence (the start of
        // the assistant's think block); the future shared path runs through
        // the whole think-stripped render plus the next user turn's header —
        // the rewind span between them is what the speculative pass prefills.
        #expect(canonical.count < future.count)
        #expect(Array(future[0..<canonical.count]) == canonical)

        let strippedRender = try strippingTokenizer.applyChatTemplate(
            messages: conversation(messages: [
                HTTPPrefixCacheMessage(role: .user, content: "question"),
                HTTPPrefixCacheMessage(role: .assistant, content: "answer"),
            ]).promptMessages,
            tools: nil,
            additionalContext: ["add_generation_prompt": false]
        )
        let header = strippingTokenizer.encode(
            text: "<|im_start|>user\n", addSpecialTokens: false
        )
        #expect(future == strippedRender + header)
    }

    @Test func futureSharedPrefixStopsAtItsCooperativeCheckWhenCancelled() async {
        // The probe body is synchronous render work — a preemption reaches it
        // only through its cooperative checks (the speculative pass cancels
        // its probe task on cancellation; `discard()`ed seeds do the same).
        // Without them, the preempting request would wait out the full probe.
        let stored = conversation(messages: [
            HTTPPrefixCacheMessage(role: .user, content: "question"),
            HTTPPrefixCacheMessage(role: .assistant, content: "answer"),
        ])
        let tokenizer = self.tokenizer

        let observedCancellation = await Task.detached { () -> Bool in
            withUnsafeCurrentTask { $0?.cancel() }
            do {
                _ = try LeafAdmissionBuilder.futureSharedPrefix(
                    storedConversation: stored,
                    toolSpecs: nil,
                    tokenizer: tokenizer,
                    keySpace: .identity()
                )
                return false
            } catch is CancellationError {
                return true
            } catch {
                return false
            }
        }.value

        #expect(observedCancellation)
    }

    // MARK: - plan() — the GPU-free leaf-capture routing decision

    /// A boundary snapshot with a controllable `tokenOffset` for routing tests —
    /// `plan` reads only the offset, never the KV body.
    private func boundary(offset: Int) -> HybridCacheSnapshot {
        let kv = KVCacheSimple()
        kv.state = [
            MLXArray.zeros([1, 1, max(offset, 1), 64]),
            MLXArray.zeros([1, 1, max(offset, 1), 64]),
        ]
        return HybridCacheSnapshot.capture(cache: [kv], offset: offset, type: .leaf)!
    }

    /// A resolver that never yields a boundary — the canonical fallback misses.
    private let noResolvedBoundary: @Sendable ([Int]) async -> HybridCacheSnapshot? = { _ in nil }

    // MARK: directTool routing

    private func toolTurn() -> HTTPPrefixCacheConversation {
        conversation(messages: [
            HTTPPrefixCacheMessage(role: .user, content: "call the tool"),
            HTTPPrefixCacheMessage(role: .assistant, content: "calling"),
        ])
    }

    @Test func directToolWithUsableBoundaryAndConvergingProbePlansFromBoundary() async throws {
        let stored = toolTurn()
        let toolTokens = try LeafAdmissionBuilder.reusablePrefix(
            continuation: .toolResult, storedConversation: stored, toolSpecs: nil,
            tokenizer: tokenizer, keySpace: .identity()
        )!.get()
        let plan = await LeafAdmissionBuilder.plan(
            mode: .directTool,
            storedConversation: stored,
            storedTokens: toolTokens,
            toolSpecs: nil,
            transientBoundary: boundary(offset: 5),
            tokenizer: tokenizer,
            keySpace: .identity(),
            resolveBoundary: noResolvedBoundary
        )
        guard case .fromBoundary(let b, let tokens) = plan else {
            Issue.record("expected .fromBoundary, got \(plan)")
            return
        }
        #expect(b.tokenOffset == 5)
        #expect(tokens == toolTokens)
    }

    @Test func directToolWithoutTransientBoundarySkips() async {
        let plan = await LeafAdmissionBuilder.plan(
            mode: .directTool,
            storedConversation: toolTurn(),
            storedTokens: [1, 2, 3],
            toolSpecs: nil,
            transientBoundary: nil,
            tokenizer: tokenizer,
            keySpace: .identity(),
            resolveBoundary: noResolvedBoundary
        )
        guard case .skip(let reason) = plan else {
            Issue.record("expected .skip, got \(plan)")
            return
        }
        #expect(reason == .noTransientBoundary)
    }

    @Test func directToolWithDivergingProbeSkips() async {
        // An empty conversation diverges; the transient boundary is present, so
        // the probe — not the boundary — is what rules out the capture.
        let plan = await LeafAdmissionBuilder.plan(
            mode: .directTool,
            storedConversation: HTTPPrefixCacheConversation(systemPrompt: nil, messages: []),
            storedTokens: [1, 2, 3],
            toolSpecs: nil,
            transientBoundary: boundary(offset: 5),
            tokenizer: tokenizer,
            keySpace: .identity(),
            resolveBoundary: noResolvedBoundary
        )
        guard case .skip(let reason) = plan else {
            Issue.record("expected .skip, got \(plan)")
            return
        }
        #expect(reason == .probeDivergence)
    }

    @Test func directToolWithEmptyResidualSkipsAtOrBeforeBoundary() async throws {
        let stored = toolTurn()
        let toolTokens = try LeafAdmissionBuilder.reusablePrefix(
            continuation: .toolResult, storedConversation: stored, toolSpecs: nil,
            tokenizer: tokenizer, keySpace: .identity()
        )!.get()
        // Boundary sits at the end of the tool render: the residual is empty.
        let plan = await LeafAdmissionBuilder.plan(
            mode: .directTool,
            storedConversation: stored,
            storedTokens: toolTokens,
            toolSpecs: nil,
            transientBoundary: boundary(offset: toolTokens.count),
            tokenizer: tokenizer,
            keySpace: .identity(),
            resolveBoundary: noResolvedBoundary
        )
        guard case .skip(let reason) = plan else {
            Issue.record("expected .skip, got \(plan)")
            return
        }
        #expect(reason == .storedAtOrBeforeBoundary(
            storedLen: toolTokens.count, boundaryOffset: toolTokens.count
        ))
    }

    @Test func directToolBoundaryInsideTheImagePrefixSkipsTyped() async throws {
        // One image, run length 4: the fake key space's prepared shape puts
        // the run end (minimum warm offset) at 5. A transient boundary at 3
        // would leave the image run in the residual — typed skip.
        let stored = conversation(messages: [
            HTTPPrefixCacheMessage(
                role: .user, content: "look", images: [HTTPPrefixCacheImage(data: Data([0x0C]))]
            ),
            HTTPPrefixCacheMessage(role: .assistant, content: "calling"),
        ])
        let keySpace = try FakeChatMLTokenizer.keySpace(for: stored, runLengths: [4])
        let plan = await LeafAdmissionBuilder.plan(
            mode: .directTool,
            storedConversation: stored,
            storedTokens: [1, 2, 3],
            toolSpecs: nil,
            transientBoundary: boundary(offset: 3),
            tokenizer: tokenizer,
            keySpace: keySpace,
            resolveBoundary: noResolvedBoundary
        )
        guard case .skip(let reason) = plan else {
            Issue.record("expected .skip, got \(plan)")
            return
        }
        #expect(reason == .boundaryInsideImagePrefix(
            boundaryOffset: 3, minimumWarmOffset: keySpace.minimumWarmOffset
        ))
        #expect(keySpace.minimumWarmOffset == 5)
    }

    @Test func canonicalBoundaryInsideTheImagePrefixFallsBackToTheResolver() async throws {
        let stored = conversation(messages: [
            HTTPPrefixCacheMessage(
                role: .user, content: "explain this", images: [HTTPPrefixCacheImage(data: Data([0x0D]))]
            ),
            HTTPPrefixCacheMessage(role: .assistant, content: "because reasons"),
        ])
        let keySpace = try FakeChatMLTokenizer.keySpace(for: stored, runLengths: [4])
        let canonical = try LeafAdmissionBuilder.reusablePrefix(
            continuation: .userTurn, storedConversation: stored, toolSpecs: nil,
            tokenizer: tokenizer, keySpace: keySpace
        )!.get()
        // The transient boundary sits inside the image prefix; the resolver's
        // snapshot past the warm offset is chosen instead.
        let resolvedOffset = keySpace.minimumWarmOffset + 2
        let plan = await LeafAdmissionBuilder.plan(
            mode: .canonical,
            storedConversation: stored,
            storedTokens: canonical,
            toolSpecs: nil,
            transientBoundary: boundary(offset: 3),
            tokenizer: tokenizer,
            keySpace: keySpace,
            resolveBoundary: { _ in self.boundary(offset: resolvedOffset) }
        )
        guard case .fromBoundary(let b, let tokens) = plan else {
            Issue.record("expected .fromBoundary, got \(plan)")
            return
        }
        #expect(b.tokenOffset == resolvedOffset)
        #expect(tokens == canonical)
    }

    @Test func canonicalSkipsWhenNoBoundaryClearsTheImagePrefix() async throws {
        let stored = conversation(messages: [
            HTTPPrefixCacheMessage(
                role: .user, content: "explain this", images: [HTTPPrefixCacheImage(data: Data([0x0E]))]
            ),
            HTTPPrefixCacheMessage(role: .assistant, content: "because reasons"),
        ])
        let keySpace = try FakeChatMLTokenizer.keySpace(for: stored, runLengths: [4])
        let canonical = try LeafAdmissionBuilder.reusablePrefix(
            continuation: .userTurn, storedConversation: stored, toolSpecs: nil,
            tokenizer: tokenizer, keySpace: keySpace
        )!.get()
        // Both the transient boundary and the resolved snapshot sit inside the
        // image prefix — there is no usable restore boundary.
        let plan = await LeafAdmissionBuilder.plan(
            mode: .canonical,
            storedConversation: stored,
            storedTokens: canonical,
            toolSpecs: nil,
            transientBoundary: boundary(offset: 3),
            tokenizer: tokenizer,
            keySpace: keySpace,
            resolveBoundary: { _ in self.boundary(offset: 2) }
        )
        guard case .skip(let reason) = plan else {
            Issue.record("expected .skip, got \(plan)")
            return
        }
        #expect(reason == .noResolvedBoundary(canonicalLen: canonical.count))
    }

    @Test func planSkipsTypedWhenTheProbeRenderCannotBeTranslated() async throws {
        let first = HTTPPrefixCacheMessage(
            role: .user, content: "one", images: [HTTPPrefixCacheImage(data: Data([0x0A]))]
        )
        let stored = conversation(messages: [
            first,
            HTTPPrefixCacheMessage(
                role: .user, content: "two", images: [HTTPPrefixCacheImage(data: Data([0x0B]))]
            ),
            HTTPPrefixCacheMessage(role: .assistant, content: "calling"),
        ])
        // A one-image key space against a two-image probe render: the second
        // pad has no image-table entry, so translation fails typed and only
        // this leaf capture skips.
        let keySpace = try FakeChatMLTokenizer.keySpace(
            for: conversation(messages: [first]), runLengths: [4]
        )
        let plan = await LeafAdmissionBuilder.plan(
            mode: .directTool,
            storedConversation: stored,
            storedTokens: [1, 2, 3],
            toolSpecs: nil,
            transientBoundary: boundary(offset: 1),
            tokenizer: tokenizer,
            keySpace: keySpace,
            resolveBoundary: noResolvedBoundary
        )
        guard case .skip(let reason) = plan else {
            Issue.record("expected .skip, got \(plan)")
            return
        }
        #expect(reason == .renderTranslationFailed(
            failure: .placeholderOccurrencesExceedImages(occurrences: 2, images: 1)
        ))
    }

    // MARK: canonical routing

    private func canonicalTurn() -> HTTPPrefixCacheConversation {
        conversation(messages: [
            HTTPPrefixCacheMessage(role: .user, content: "explain"),
            HTTPPrefixCacheMessage(role: .assistant, content: "because reasons"),
        ])
    }

    private func canonicalPrefix(_ stored: HTTPPrefixCacheConversation) throws -> [Int] {
        try LeafAdmissionBuilder.reusablePrefix(
            continuation: .userTurn, storedConversation: stored, toolSpecs: nil,
            tokenizer: tokenizer, keySpace: .identity()
        )!.get()
    }

    @Test func canonicalWithUsableTransientBoundaryPlansFromIt() async throws {
        let stored = canonicalTurn()
        let canonical = try canonicalPrefix(stored)
        let plan = await LeafAdmissionBuilder.plan(
            mode: .canonical,
            storedConversation: stored,
            storedTokens: canonical,
            toolSpecs: nil,
            transientBoundary: boundary(offset: 4),
            tokenizer: tokenizer,
            keySpace: .identity(),
            resolveBoundary: noResolvedBoundary
        )
        guard case .fromBoundary(let b, let tokens) = plan else {
            Issue.record("expected .fromBoundary, got \(plan)")
            return
        }
        #expect(b.tokenOffset == 4)
        #expect(tokens == canonical)
    }

    @Test func canonicalFallsBackToResolverWhenNoTransientBoundary() async throws {
        let stored = canonicalTurn()
        let canonical = try canonicalPrefix(stored)
        let plan = await LeafAdmissionBuilder.plan(
            mode: .canonical,
            storedConversation: stored,
            storedTokens: canonical,
            toolSpecs: nil,
            transientBoundary: nil,
            tokenizer: tokenizer,
            keySpace: .identity(),
            resolveBoundary: { _ in self.boundary(offset: 7) }
        )
        guard case .fromBoundary(let b, let tokens) = plan else {
            Issue.record("expected .fromBoundary, got \(plan)")
            return
        }
        #expect(b.tokenOffset == 7)
        #expect(tokens == canonical)
    }

    @Test func canonicalSkipsWhenResolverMisses() async throws {
        let stored = canonicalTurn()
        let canonical = try canonicalPrefix(stored)
        let plan = await LeafAdmissionBuilder.plan(
            mode: .canonical,
            storedConversation: stored,
            storedTokens: canonical,
            toolSpecs: nil,
            transientBoundary: nil,
            tokenizer: tokenizer,
            keySpace: .identity(),
            resolveBoundary: noResolvedBoundary
        )
        guard case .skip(let reason) = plan else {
            Issue.record("expected .skip, got \(plan)")
            return
        }
        #expect(reason == .noResolvedBoundary(canonicalLen: canonical.count))
    }

    @Test func canonicalSkipsWhenResolvedBoundaryIsNotStrictlyBeforeTheCanonicalPath() async throws {
        let stored = canonicalTurn()
        let canonical = try canonicalPrefix(stored)
        // A resolved snapshot at the canonical length is not a usable restore
        // boundary — there is no residual to reprefill.
        let plan = await LeafAdmissionBuilder.plan(
            mode: .canonical,
            storedConversation: stored,
            storedTokens: canonical,
            toolSpecs: nil,
            transientBoundary: nil,
            tokenizer: tokenizer,
            keySpace: .identity(),
            resolveBoundary: { _ in self.boundary(offset: canonical.count) }
        )
        guard case .skip(let reason) = plan else {
            Issue.record("expected .skip, got \(plan)")
            return
        }
        #expect(reason == .noResolvedBoundary(canonicalLen: canonical.count))
    }

    @Test func canonicalSkipsWhenCanonicalPrefixIsLongerThanTheStoredPath() async throws {
        let stored = canonicalTurn()
        let canonical = try canonicalPrefix(stored)
        // A usable boundary, but the stored path is shorter than the canonical
        // prefix — the render disagreement the canonical leaf exists to avoid.
        let plan = await LeafAdmissionBuilder.plan(
            mode: .canonical,
            storedConversation: stored,
            storedTokens: [1, 2],
            toolSpecs: nil,
            transientBoundary: boundary(offset: 4),
            tokenizer: tokenizer,
            keySpace: .identity(),
            resolveBoundary: noResolvedBoundary
        )
        guard case .skip(let reason) = plan else {
            Issue.record("expected .skip, got \(plan)")
            return
        }
        #expect(reason == .canonicalLongerThanStored(canonicalLen: canonical.count, storedLen: 2))
    }
}
