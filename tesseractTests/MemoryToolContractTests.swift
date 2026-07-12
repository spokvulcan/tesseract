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
import MLXLMCommon
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

    @Test("remember routes corrections to contest, not to a negation beside the lie")
    func rememberRedirectsCorrections() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("memory-tool-contract-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }

        let tool = createRememberTool(memory: try makeEngine(root: root))
        // The pre-#333 system prompt taught the opposite ("remember the
        // corrected version") and the live store ended up holding "He gave me
        // the nickname Pelican" and its negation, both live.
        #expect(tool.description.contains("contest"))
    }

    @Test("contest teaches the veto: dispute by handle, never rewrite")
    func contestVoiceContract() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("memory-tool-contract-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }

        let tool = createContestTool(memory: try makeEngine(root: root))

        // The gesture: he says a memory is wrong, the agent relays the veto.
        #expect(tool.description.contains("WRONG"))
        // The mechanism the model must know: nothing is deleted or rewritten
        // inline; sleep re-examines against what was said.
        #expect(tool.description.contains("sleep"))
        #expect(tool.description.localizedCaseInsensitiveContains("nothing is deleted"))
        // And where handles come from.
        #expect(tool.description.contains("recall"))

        let memory = try #require(tool.parameterSchema.properties["memory"])
        #expect(memory.description.contains("handle"))
        let reason = try #require(tool.parameterSchema.properties["reason"])
        #expect(reason.description.contains("he said"))
        #expect(tool.parameterSchema.required.contains("memory"))
        #expect(tool.parameterSchema.required.contains("reason"))
    }

    @Test("The veto travels: contest by handle flips the belief and keeps his words")
    func contestFlowThroughTheRealTool() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("memory-tool-contract-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let engine = try makeEngine(root: root)

        let wrong = try #require(await engine.remember("He gave me the nickname \"Pelican.\""))
        let handle = String(wrong.id.uuidString.prefix(8)).lowercased()
        let tool = createContestTool(memory: engine)

        // A handle that matches nothing teaches the model to recall again.
        await #expect(throws: MemoryToolError.self) {
            _ = try await tool.execute(
                "contest-0",
                ["memory": .string("00000000"), "reason": .string("wrong")], nil, nil)
        }

        let result = try await tool.execute(
            "contest-1",
            [
                "memory": .string(handle),
                "reason": .string("It referred to the SVG pelican image, not a name."),
            ], nil, nil)
        #expect(result.content.textContent.contains("Contested"))

        let after = try #require(await engine.allMemories().first { $0.id == wrong.id })
        #expect(after.status == .contested)

        // Contesting what is already disputed reports that, rather than failing.
        let again = try await tool.execute(
            "contest-2",
            ["memory": .string(handle), "reason": .string("still wrong")], nil, nil)
        #expect(again.content.textContent.contains("Already disputed"))
    }

    @Test("recall lines lead with the handle contest addresses them by")
    func recallLinesCarryHandles() {
        let record = MemoryRecord(
            text: "He gave me the nickname \"Pelican.\"", kind: .belief,
            provenance: .inferred, bornAt: Date())
        let line = MemoryToolFormatter.line(
            ScoredMemory(memory: record, score: 1, relevance: 1, isExploration: false))

        let handle = record.id.uuidString.prefix(8).lowercased()
        #expect(line.hasPrefix("- [\(handle)] He gave me"))
        #expect(line.contains("[inferred]"))
    }
}
