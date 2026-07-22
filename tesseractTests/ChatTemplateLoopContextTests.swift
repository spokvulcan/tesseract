//
//  ChatTemplateLoopContextTests.swift
//  tesseractTests
//
//  Pins the swift-jinja capability the Nanbeige4.2 chat template depends on.
//
//  Found live 2026-07-22: the template's tool-response branch calls
//  `loop.previtem.get('role', '')` / `loop.nextitem.get('role', '')` to decide
//  where the wrapping user turn opens and closes. swift-jinja < 2.3.6 has no
//  previtem/nextitem on the loop object, so the opener was silently dropped
//  (malformed prompt) and any tool message that was not the last message threw
//  "Runtime error: Cannot call non-function value", killing the agent run.
//  Package.resolved pins ≥ 2.3.6; this test fails if a re-resolve ever slides
//  back.
//

import Foundation
import Jinja
import Testing

@Suite("Chat template loop context — previtem/nextitem")
struct ChatTemplateLoopContextTests {

    /// The exact construct from the Nanbeige tool-response branch, reduced:
    /// consecutive tool messages merge into one user turn, and the closer only
    /// lands after the last of them.
    private let template = """
        {%- for message in messages -%}
        {%- if message.get('role', '') == "tool" -%}
        {%- if loop.previtem and loop.previtem.get('role', '') != "tool" -%}
        {{- '<|im_start|>user' -}}
        {%- endif -%}
        {{- '<r>' + message.content + '</r>' -}}
        {%- if loop.last or loop.nextitem.get('role', '') != "tool" -%}
        {{- '<|im_end|>' -}}
        {%- endif -%}
        {%- endif -%}
        {%- endfor -%}
        """

    @Test("A tool message mid-transcript renders instead of throwing")
    func toolMessageMidTranscriptRenders() throws {
        let messages: Value = .array([
            .object(["role": .string("assistant"), "content": .string("calling")]),
            .object(["role": .string("tool"), "content": .string("first")]),
            .object(["role": .string("tool"), "content": .string("second")]),
            .object(["role": .string("assistant"), "content": .string("done")]),
        ])
        let rendered = try Template(template).render(["messages": messages])
        #expect(rendered == "<|im_start|>user<r>first</r><r>second</r><|im_end|>")
    }
}
