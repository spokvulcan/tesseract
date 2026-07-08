//
//  FFIRoundTripTests.swift
//  TesseractHighlight
//
//  The FFI-contract seam (PRD #200): the generated bindings round-trip real
//  calls into the Rust crate — spans reassemble losslessly, roles are known,
//  diff rows carry coherent line numbers. The heavy per-behavior coverage
//  lives in the crate's own `cargo test`; this proves the bridge.
//

import Testing

@testable import TesseractHighlight

@Suite struct FFIRoundTripTests {

    @Test func highlightReassemblesSourceLosslessly() {
        let code = "func greet(name: String) -> String {\n    \"hi \\(name)\" // wave\n}"
        let lines = highlight(code: code, languageHint: "swift")
        let reassembled = lines.map { line in
            line.spans.map(\.text).joined()
        }.joined(separator: "\n")
        #expect(reassembled == code)
        #expect(lines.contains { $0.spans.contains { $0.role == .keyword } })
        #expect(lines.contains { $0.spans.contains { $0.role == .comment } })
    }

    @Test func unknownLanguageFallsBackToPlain() {
        let lines = highlight(code: "plain words", languageHint: "no-such-language")
        #expect(lines.count == 1)
        #expect(lines[0].spans.allSatisfy { $0.role == .plain })
    }

    @Test func diffCarriesKindsLineNumbersAndEmphasis() {
        let rows = diffHighlighted(
            old: "let a = 1\nlet b = 2\n",
            new: "let a = 1\nlet b = 3\n",
            languageHint: "swift"
        )
        #expect(rows.map(\.kind) == [.context, .removed, .added])
        #expect(rows[1].oldLine == 2 && rows[1].newLine == nil)
        #expect(rows[2].newLine == 2 && rows[2].oldLine == nil)
        let addedText = rows[2].segments.map(\.text).joined()
        #expect(addedText == "let b = 3")
        #expect(rows[2].segments.contains { $0.emphasized && $0.text.contains("3") })
    }

    @Test func unicodeSurvivesTheBridge() {
        let lines = highlight(code: "let café = \"🌊 ünïcode\"", languageHint: "swift")
        let reassembled = lines[0].spans.map(\.text).joined()
        #expect(reassembled == "let café = \"🌊 ünïcode\"")
    }

    @Test func emptyInputsAreSafe() {
        #expect(highlight(code: "", languageHint: "swift").count == 1)
        #expect(diffHighlighted(old: "", new: "", languageHint: "swift").isEmpty)
    }
}
