import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import Tesseract_Agent

/// Behavior of the **Leaf Admission Builder**'s reusable-prefix probe: the
/// GPU-free routing core that finds the shared token path a future continuation
/// can hydrate. Driven by a byte-level fake tokenizer — no model.
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
            tokenizer: tokenizer
        )
        // The reusable prefix is exactly the stored turn's own render — the
        // synthetic user continuation is excluded.
        #expect(prefix == (try render(stored)))
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
            tokenizer: tokenizer
        )
        #expect(prefix == (try render(stored)))
    }

    @Test func probeDivergesToNilWhenThereIsNoCommonPrefix() throws {
        // An empty conversation renders to nothing, so the stored render and the
        // probe-extended render share no prefix.
        let empty = HTTPPrefixCacheConversation(systemPrompt: nil, messages: [])
        let prefix = try LeafAdmissionBuilder.reusablePrefix(
            continuation: .userTurn,
            storedConversation: empty,
            toolSpecs: nil,
            tokenizer: tokenizer
        )
        #expect(prefix == nil)
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

    @Test func directLeafModePlansToLiveCacheAtTheStoredPath() async {
        let stored = conversation(messages: [
            HTTPPrefixCacheMessage(role: .user, content: "hi"),
            HTTPPrefixCacheMessage(role: .assistant, content: "hello"),
        ])
        let plan = await LeafAdmissionBuilder.plan(
            mode: .directLeaf,
            storedConversation: stored,
            storedTokens: [1, 2, 3, 4],
            toolSpecs: nil,
            transientBoundary: nil,
            tokenizer: tokenizer,
            resolveBoundary: noResolvedBoundary
        )
        guard case .liveCache(let tokens) = plan else {
            Issue.record("expected .liveCache, got \(plan)")
            return
        }
        #expect(tokens == [1, 2, 3, 4])
    }

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
            continuation: .toolResult, storedConversation: stored, toolSpecs: nil, tokenizer: tokenizer
        )!
        let plan = await LeafAdmissionBuilder.plan(
            mode: .directToolLeaf,
            storedConversation: stored,
            storedTokens: toolTokens,
            toolSpecs: nil,
            transientBoundary: boundary(offset: 5),
            tokenizer: tokenizer,
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
            mode: .directToolLeaf,
            storedConversation: toolTurn(),
            storedTokens: [1, 2, 3],
            toolSpecs: nil,
            transientBoundary: nil,
            tokenizer: tokenizer,
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
            mode: .directToolLeaf,
            storedConversation: HTTPPrefixCacheConversation(systemPrompt: nil, messages: []),
            storedTokens: [1, 2, 3],
            toolSpecs: nil,
            transientBoundary: boundary(offset: 5),
            tokenizer: tokenizer,
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
            continuation: .toolResult, storedConversation: stored, toolSpecs: nil, tokenizer: tokenizer
        )!
        // Boundary sits at the end of the tool render: the residual is empty.
        let plan = await LeafAdmissionBuilder.plan(
            mode: .directToolLeaf,
            storedConversation: stored,
            storedTokens: toolTokens,
            toolSpecs: nil,
            transientBoundary: boundary(offset: toolTokens.count),
            tokenizer: tokenizer,
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

    // MARK: canonical routing

    private func canonicalTurn() -> HTTPPrefixCacheConversation {
        conversation(messages: [
            HTTPPrefixCacheMessage(role: .user, content: "explain"),
            HTTPPrefixCacheMessage(role: .assistant, content: "because reasons"),
        ])
    }

    private func canonicalPrefix(_ stored: HTTPPrefixCacheConversation) throws -> [Int] {
        try LeafAdmissionBuilder.reusablePrefix(
            continuation: .userTurn, storedConversation: stored, toolSpecs: nil, tokenizer: tokenizer
        )!
    }

    @Test func canonicalWithUsableTransientBoundaryPlansFromIt() async throws {
        let stored = canonicalTurn()
        let canonical = try canonicalPrefix(stored)
        let plan = await LeafAdmissionBuilder.plan(
            mode: .canonicalUserLeaf,
            storedConversation: stored,
            storedTokens: canonical,
            toolSpecs: nil,
            transientBoundary: boundary(offset: 4),
            tokenizer: tokenizer,
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
            mode: .canonicalUserLeaf,
            storedConversation: stored,
            storedTokens: canonical,
            toolSpecs: nil,
            transientBoundary: nil,
            tokenizer: tokenizer,
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
            mode: .canonicalUserLeaf,
            storedConversation: stored,
            storedTokens: canonical,
            toolSpecs: nil,
            transientBoundary: nil,
            tokenizer: tokenizer,
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
            mode: .canonicalUserLeaf,
            storedConversation: stored,
            storedTokens: canonical,
            toolSpecs: nil,
            transientBoundary: nil,
            tokenizer: tokenizer,
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
            mode: .canonicalUserLeaf,
            storedConversation: stored,
            storedTokens: [1, 2],
            toolSpecs: nil,
            transientBoundary: boundary(offset: 4),
            tokenizer: tokenizer,
            resolveBoundary: noResolvedBoundary
        )
        guard case .skip(let reason) = plan else {
            Issue.record("expected .skip, got \(plan)")
            return
        }
        #expect(reason == .canonicalLongerThanStored(canonicalLen: canonical.count, storedLen: 2))
    }
}
