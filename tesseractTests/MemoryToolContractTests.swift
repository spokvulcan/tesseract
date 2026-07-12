//
//  MemoryToolContractTests.swift
//  tesseractTests
//
//  The remember tool's voice contract (#333).
//
//  Found on the owner's live store, 2026-07-12: the tool description said
//  "in the first person" — a phrase transplanted from ADR-0035, where it means
//  the ASSISTANT's first person — and the model resolved "I" to the owner
//  instead, echoing his words verbatim. 17 of 27 remember-born beliefs were in
//  his voice ("I love cats"); sleep, whose prompt says "write about him in the
//  third person", never slipped once in 117.
//
//  The rule is about referents, not pronouns: in a stored memory, "he" is the
//  owner and "I" is the agent. These tests pin that contract into the tool
//  description so the inverted phrase can never quietly return.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
@Suite("Memory tool contracts — the voice convention")
struct MemoryToolContractTests {

    /// A minimal engine: the contract lives in the tool's static text, so the
    /// store is a throwaway and the embedder is never loaded.
    private func makeEngine(root: URL) throws -> MemoryEngine {
        let store = try MemoryStore(directory: root)
        return MemoryEngine(
            store: store,
            embedder: MemoryEmbedder(),
            isEnabled: { true },
            isDictationCaptureEnabled: { false },
            embedderDirectory: { nil }
        )
    }

    @Test("remember teaches the agent's voice, not its opposite")
    func rememberVoiceContract() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("memory-tool-contract-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }

        let tool = createRememberTool(memory: try makeEngine(root: root))

        // The convention, stated: about him, in the third person.
        #expect(tool.description.contains("third person"))
        // The worked example is what a small model actually imitates.
        #expect(tool.description.contains("He likes to eat apples in the morning."))
        // A directive stays anchored on the ordainer — sleep's canon, one voice
        // at both write doors.
        #expect(tool.description.contains("He wants me to answer briefly"))
        // The phrase that caused #333: "first person" with the referent
        // inverted. It must not reappear in any form.
        #expect(!tool.description.localizedCaseInsensitiveContains("first person"))

        let textSchema = try #require(tool.parameterSchema.properties["text"])
        #expect(textSchema.description.contains("third person"))
        #expect(!textSchema.description.localizedCaseInsensitiveContains("first person"))
    }
}
