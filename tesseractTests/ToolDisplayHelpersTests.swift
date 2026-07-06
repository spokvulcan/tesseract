//
//  ToolDisplayHelpersTests.swift
//  tesseractTests
//
//  The Tool Row Title grammar: imperative verb + Workspace-relative target,
//  matched against the actual tool registry — unknown tools fall back to the
//  raw name.
//

import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

struct ToolDisplayHelpersTests {

    @Test func fileToolTitlesUseVerbPlusRelativePath() {
        #expect(
            ToolDisplayHelpers.titleForTool("read", arguments: ["path": .string("notes/todo.md")])
                == "Read notes/todo.md")
        #expect(
            ToolDisplayHelpers.titleForTool("write", arguments: ["path": .string("./a.txt")])
                == "Write a.txt")
        // Bare verb while arguments are still streaming.
        #expect(ToolDisplayHelpers.titleForTool("edit", arguments: nil) == "Edit")
    }

    @Test func absolutePathsInsideTheWorkspaceRenderRelative() {
        let absolute = PathSandbox.defaultRoot.appendingPathComponent("docs/plan.md").path
        #expect(
            ToolDisplayHelpers.titleForTool("read", arguments: ["path": .string(absolute)])
                == "Read docs/plan.md")
    }

    @Test func listTitlesNameTheWorkspaceRoot() {
        #expect(ToolDisplayHelpers.titleForTool("ls", arguments: nil) == "List workspace")
        #expect(
            ToolDisplayHelpers.titleForTool("ls", arguments: ["path": .string(".")])
                == "List workspace")
        #expect(
            ToolDisplayHelpers.titleForTool("ls", arguments: ["path": .string("docs")])
                == "List docs")
    }

    @Test func skillSearchFetchAndUnknownTitles() {
        #expect(
            ToolDisplayHelpers.titleForTool("use_skill", arguments: ["name": .string("proofread")])
                == "Load skill proofread")
        #expect(
            ToolDisplayHelpers.titleForTool("web_search", arguments: ["query": .string("swift 6")])
                == "Search \u{201C}swift 6\u{201D}")
        #expect(
            ToolDisplayHelpers.titleForTool(
                "web_fetch", arguments: ["url": .string("https://example.com/docs/")])
                == "Fetch example.com/docs")
        #expect(ToolDisplayHelpers.titleForTool("mystery_tool", arguments: nil) == "mystery_tool")
    }
}
