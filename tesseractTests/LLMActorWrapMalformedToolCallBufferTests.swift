import Testing

@testable import Tesseract_Agent

struct LLMActorWrapMalformedToolCallBufferTests {

    @Test func appendsCloseTagWhenMissing() {
        let input = "<tool_call>\n<read>\n<file_path>/tmp/x</file_path>\n</read>"
        let wrapped = LLMActor.wrapMalformedToolCallBuffer(input)
        #expect(wrapped.hasPrefix("<tool_call>"))
        #expect(wrapped.hasSuffix("</tool_call>"))
        #expect(wrapped == input + "\n</tool_call>")
    }

    @Test func prependsOpenTagWhenMissing() {
        let input = "<read>\n<file_path>/tmp/x</file_path>\n</read>\n</tool_call>"
        let wrapped = LLMActor.wrapMalformedToolCallBuffer(input)
        #expect(wrapped.hasPrefix("<tool_call>"))
        #expect(wrapped.hasSuffix("</tool_call>"))
        #expect(wrapped == "<tool_call>\n" + input)
    }

    @Test func addsBothTagsWhenNeitherPresent() {
        let input = "<read>\n<file_path>/tmp/x</file_path>\n</read>"
        let wrapped = LLMActor.wrapMalformedToolCallBuffer(input)
        #expect(wrapped == "<tool_call>\n" + input + "\n</tool_call>")
    }

    @Test func leavesCompleteBufferUnchanged() {
        let input = "<tool_call>\n<read>\n<file_path>/tmp/x</file_path>\n</read>\n</tool_call>"
        let wrapped = LLMActor.wrapMalformedToolCallBuffer(input)
        #expect(wrapped == input)
    }

    @Test func handlesBufferEndingWithoutNewline() {
        let input = "<tool_call><function=read><parameter=path>/tmp/x</parameter></function>"
        let wrapped = LLMActor.wrapMalformedToolCallBuffer(input)
        #expect(wrapped.hasPrefix("<tool_call>"))
        #expect(wrapped.hasSuffix("</tool_call>"))
        #expect(wrapped == input + "\n</tool_call>")
    }

    @Test func preservesExistingTrailingNewlineBeforeCloseTag() {
        let input = "<tool_call>\n<read>\n</read>\n"
        let wrapped = LLMActor.wrapMalformedToolCallBuffer(input)
        #expect(wrapped == input + "</tool_call>")
    }

    @Test func idempotentWhenCalledTwice() {
        let input = "<tool_call>\n<read>\n</read>"
        let once = LLMActor.wrapMalformedToolCallBuffer(input)
        let twice = LLMActor.wrapMalformedToolCallBuffer(once)
        #expect(once == twice)
    }
}
