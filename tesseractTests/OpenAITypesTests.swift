//
//  OpenAITypesTests.swift
//  tesseractTests
//

import Foundation
import Testing
@testable import Tesseract_Agent

@MainActor
struct OpenAITypesTests {

    // MARK: - Request Decoding

    @Test func decodesFullChatCompletionRequest() throws {
        let json = """
        {
          "model": "qwen3.5-9b-paro",
          "messages": [
            { "role": "system", "content": "You are a coding assistant." },
            { "role": "user", "content": "Read main.swift" },
            {
              "role": "assistant",
              "content": "I'll read that file for you.",
              "tool_calls": [
                {
                  "id": "call_abc123",
                  "type": "function",
                  "function": {
                    "name": "read",
                    "arguments": "{\\"path\\": \\"main.swift\\"}"
                  }
                }
              ]
            },
            { "role": "tool", "tool_call_id": "call_abc123", "content": "file contents..." },
            { "role": "user", "content": [
                { "type": "text", "text": "What about this image?" },
                { "type": "image_url", "image_url": { "url": "data:image/png;base64,abc" } }
              ]
            }
          ],
          "tools": [
            {
              "type": "function",
              "function": {
                "name": "bash",
                "description": "Execute shell commands",
                "parameters": {
                  "type": "object",
                  "properties": {
                    "command": { "type": "string" }
                  },
                  "required": ["command"]
                }
              }
            }
          ],
          "stream": true,
          "max_tokens": 4096,
          "max_completion_tokens": 8192,
          "temperature": 0.6,
          "top_p": 0.95,
          "reasoning_effort": "medium",
          "stream_options": { "include_usage": true }
        }
        """

        let request = try JSONDecoder().decode(OpenAI.ChatCompletionRequest.self, from: Data(json.utf8))

        #expect(request.model == "qwen3.5-9b-paro")
        #expect(request.messages.count == 5)
        #expect(request.stream == true)
        #expect(request.max_tokens == 4096)
        #expect(request.max_completion_tokens == 8192)
        #expect(request.effectiveMaxTokens == 8192)
        #expect(request.temperature == 0.6)
        #expect(request.top_p == 0.95)
        #expect(request.reasoning_effort == "medium")
        #expect(request.stream_options?.include_usage == true)

        // System message
        #expect(request.messages[0].role == .system)
        #expect(request.messages[0].content?.textValue == "You are a coding assistant.")

        // User text message
        #expect(request.messages[1].role == .user)
        #expect(request.messages[1].content?.textValue == "Read main.swift")

        // Assistant with tool calls
        #expect(request.messages[2].role == .assistant)
        #expect(request.messages[2].tool_calls?.count == 1)
        #expect(request.messages[2].tool_calls?[0].id == "call_abc123")
        #expect(request.messages[2].tool_calls?[0].function?.name == "read")

        // Tool result
        #expect(request.messages[3].role == .tool)
        #expect(request.messages[3].tool_call_id == "call_abc123")
        #expect(request.messages[3].content?.textValue == "file contents...")

        // Multipart user message
        #expect(request.messages[4].role == .user)
        if case .parts(let parts) = request.messages[4].content {
            #expect(parts.count == 2)
            #expect(parts[0].type == .text)
            #expect(parts[0].text == "What about this image?")
            #expect(parts[1].type == .image_url)
            #expect(parts[1].image_url?.url == "data:image/png;base64,abc")
        } else {
            Issue.record("Expected multipart content")
        }

        // Tools
        #expect(request.tools?.count == 1)
        #expect(request.tools?[0].type == "function")
        #expect(request.tools?[0].function.name == "bash")
        #expect(request.tools?[0].function.description == "Execute shell commands")
    }

    @Test func effectiveMaxTokensFallsBackToMaxTokens() throws {
        let json = """
        { "messages": [{ "role": "user", "content": "hi" }], "max_tokens": 512 }
        """
        let request = try JSONDecoder().decode(OpenAI.ChatCompletionRequest.self, from: Data(json.utf8))
        #expect(request.effectiveMaxTokens == 512)
    }

    @Test func decodesAssistantReasoningContentInRequestMessages() throws {
        let json = """
        {
          "messages": [
            {
              "role": "assistant",
              "content": "Final answer",
              "reasoning_content": "Hidden chain of thought"
            }
          ]
        }
        """

        let request = try JSONDecoder().decode(OpenAI.ChatCompletionRequest.self, from: Data(json.utf8))

        #expect(request.messages.count == 1)
        #expect(request.messages[0].reasoning_content == "Hidden chain of thought")
        #expect(request.messages[0].resolvedReasoningContent == "Hidden chain of thought")
    }

    @Test func decodesAssistantReasoningAliasInRequestMessages() throws {
        let json = """
        {
          "messages": [
            {
              "role": "assistant",
              "content": "Final answer",
              "reasoning": "Alias reasoning"
            }
          ]
        }
        """

        let request = try JSONDecoder().decode(OpenAI.ChatCompletionRequest.self, from: Data(json.utf8))

        #expect(request.messages.count == 1)
        #expect(request.messages[0].reasoning == "Alias reasoning")
        #expect(request.messages[0].resolvedReasoningContent == "Alias reasoning")
    }

    @Test func resolvedReasoningContentTrimsSurroundingWhitespace() {
        let trailing = OpenAI.ChatMessage(
            role: .assistant,
            content: .text("Hi"),
            reasoning_content: "Thinking about it.\n"
        )
        #expect(trailing.resolvedReasoningContent == "Thinking about it.")

        let surrounding = OpenAI.ChatMessage(
            role: .assistant,
            content: .text("Hi"),
            reasoning_content: "  Thinking about it.  "
        )
        #expect(surrounding.resolvedReasoningContent == "Thinking about it.")

        let whitespaceOnly = OpenAI.ChatMessage(
            role: .assistant,
            content: .text("Hi"),
            reasoning_content: "  \n\n  "
        )
        #expect(whitespaceOnly.resolvedReasoningContent == nil)

        let empty = OpenAI.ChatMessage(
            role: .assistant,
            content: .text("Hi"),
            reasoning_content: ""
        )
        #expect(empty.resolvedReasoningContent == nil)

        let nilReasoning = OpenAI.ChatMessage(role: .assistant, content: .text("Hi"))
        #expect(nilReasoning.resolvedReasoningContent == nil)
    }

    @Test func stopSequenceDecodesStringAndArray() throws {
        let singleJSON = """
        { "messages": [{ "role": "user", "content": "hi" }], "stop": "\\n" }
        """
        let single = try JSONDecoder().decode(OpenAI.ChatCompletionRequest.self, from: Data(singleJSON.utf8))
        #expect(single.stop?.sequences == ["\n"])

        let arrayJSON = """
        { "messages": [{ "role": "user", "content": "hi" }], "stop": ["\\n", "END"] }
        """
        let array = try JSONDecoder().decode(OpenAI.ChatCompletionRequest.self, from: Data(arrayJSON.utf8))
        #expect(array.stop?.sequences == ["\n", "END"])
    }

    // MARK: - Wire-Format Response Decoding (spec section 4.2 samples)

    @Test func decodesSpecNonStreamingResponse() throws {
        let json = """
        {
          "id": "chatcmpl-abc123",
          "object": "chat.completion",
          "model": "qwen3.5-9b-paro",
          "created": 1712345678,
          "system_fingerprint": "tesseract-1.0-mlx",
          "choices": [
            {
              "index": 0,
              "finish_reason": "stop",
              "message": {
                "role": "assistant",
                "content": "Here is the file content...",
                "tool_calls": [
                  {
                    "id": "call_xyz789",
                    "type": "function",
                    "function": {
                      "name": "bash",
                      "arguments": "{\\"command\\": \\"ls\\"}"
                    }
                  }
                ]
              }
            }
          ],
          "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 42,
            "total_tokens": 192,
            "prompt_tokens_details": {
              "cached_tokens": 120
            }
          }
        }
        """

        let response = try JSONDecoder().decode(OpenAI.ChatCompletionResponse.self, from: Data(json.utf8))

        #expect(response.id == "chatcmpl-abc123")
        #expect(response.object == "chat.completion")
        #expect(response.model == "qwen3.5-9b-paro")
        #expect(response.created == 1712345678)
        #expect(response.system_fingerprint == "tesseract-1.0-mlx")

        #expect(response.choices.count == 1)
        let choice = response.choices[0]
        #expect(choice.index == 0)
        #expect(choice.finish_reason == .stop)
        #expect(choice.message.role == .assistant)
        #expect(choice.message.content == "Here is the file content...")

        // Message carries both content AND tool_calls
        #expect(choice.message.tool_calls?.count == 1)
        let tc = choice.message.tool_calls![0]
        #expect(tc.id == "call_xyz789")
        #expect(tc.type == "function")
        #expect(tc.function?.name == "bash")
        #expect(tc.function?.arguments == #"{"command": "ls"}"#)

        #expect(response.usage?.prompt_tokens == 150)
        #expect(response.usage?.completion_tokens == 42)
        #expect(response.usage?.total_tokens == 192)
        #expect(response.usage?.prompt_tokens_details?.cached_tokens == 120)
    }

    @Test func decodesSpecStreamingChunks() throws {
        // Content chunk
        let contentJSON = """
        {"id":"chatcmpl-s1","object":"chat.completion.chunk","model":"qwen3.5-9b-paro","created":1712345678,"choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}
        """
        let content = try JSONDecoder().decode(OpenAI.ChatCompletionChunk.self, from: Data(contentJSON.utf8))
        #expect(content.choices[0].delta.role == .assistant)
        #expect(content.choices[0].delta.content == "Hello")
        #expect(content.choices[0].finish_reason == nil)

        // Tool call name chunk
        let toolNameJSON = """
        {"id":"chatcmpl-s1","object":"chat.completion.chunk","model":"qwen3.5-9b-paro","created":1712345678,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"bash","arguments":""}}]},"finish_reason":null}]}
        """
        let toolName = try JSONDecoder().decode(OpenAI.ChatCompletionChunk.self, from: Data(toolNameJSON.utf8))
        #expect(toolName.choices[0].delta.tool_calls?.count == 1)
        #expect(toolName.choices[0].delta.tool_calls?[0].index == 0)
        #expect(toolName.choices[0].delta.tool_calls?[0].id == "call_abc")
        #expect(toolName.choices[0].delta.tool_calls?[0].function?.name == "bash")

        // Tool call arguments chunk (no id/type, just arguments)
        let toolArgsJSON = """
        {"id":"chatcmpl-s1","object":"chat.completion.chunk","model":"qwen3.5-9b-paro","created":1712345678,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"command\\": \\"ls\\"}"}}]},"finish_reason":null}]}
        """
        let toolArgs = try JSONDecoder().decode(OpenAI.ChatCompletionChunk.self, from: Data(toolArgsJSON.utf8))
        #expect(toolArgs.choices[0].delta.tool_calls?[0].id == nil)
        #expect(toolArgs.choices[0].delta.tool_calls?[0].function?.arguments == #"{"command": "ls"}"#)

        // Reasoning chunk
        let reasoningJSON = """
        {"id":"chatcmpl-s1","object":"chat.completion.chunk","model":"qwen3.5-9b-paro","created":1712345678,"choices":[{"index":0,"delta":{"reasoning_content":"Thinking..."},"finish_reason":null}]}
        """
        let reasoning = try JSONDecoder().decode(OpenAI.ChatCompletionChunk.self, from: Data(reasoningJSON.utf8))
        #expect(reasoning.choices[0].delta.reasoning_content == "Thinking...")
        #expect(reasoning.choices[0].delta.content == nil)

        // Final chunk with finish_reason and usage
        let finalJSON = """
        {"id":"chatcmpl-s1","object":"chat.completion.chunk","model":"qwen3.5-9b-paro","created":1712345678,"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":150,"completion_tokens":42,"total_tokens":192,"prompt_tokens_details":{"cached_tokens":120}}}
        """
        let final = try JSONDecoder().decode(OpenAI.ChatCompletionChunk.self, from: Data(finalJSON.utf8))
        #expect(final.choices[0].finish_reason == .tool_calls)
        #expect(final.choices[0].delta.content == nil)
        #expect(final.choices[0].delta.tool_calls == nil)
        #expect(final.usage?.total_tokens == 192)
        #expect(final.usage?.prompt_tokens_details?.cached_tokens == 120)
    }

    // MARK: - Response Round-Trip Encoding

    @Test func encodesNonStreamingResponse() throws {
        let response = OpenAI.ChatCompletionResponse(
            id: "chatcmpl-test123",
            model: "qwen3.5-9b-paro",
            created: 1712345678,
            system_fingerprint: "tesseract-1.0-mlx",
            choices: [
                OpenAI.ChatCompletionChoice(
                    index: 0,
                    finish_reason: .stop,
                    message: OpenAI.ResponseMessage(
                        role: .assistant,
                        content: "Hello!"
                    )
                )
            ],
            usage: OpenAI.Usage(
                prompt_tokens: 150,
                completion_tokens: 42,
                total_tokens: 192,
                prompt_tokens_details: OpenAI.PromptTokensDetails(cached_tokens: 120)
            )
        )

        let data = try JSONEncoder().encode(response)
        let decoded = try JSONDecoder().decode(OpenAI.ChatCompletionResponse.self, from: data)

        #expect(decoded.id == "chatcmpl-test123")
        #expect(decoded.object == "chat.completion")
        #expect(decoded.model == "qwen3.5-9b-paro")
        #expect(decoded.created == 1712345678)
        #expect(decoded.system_fingerprint == "tesseract-1.0-mlx")
        #expect(decoded.choices.count == 1)
        #expect(decoded.choices[0].finish_reason == .stop)
        #expect(decoded.choices[0].message.content == "Hello!")
        #expect(decoded.usage?.prompt_tokens == 150)
        #expect(decoded.usage?.completion_tokens == 42)
        #expect(decoded.usage?.total_tokens == 192)
        #expect(decoded.usage?.prompt_tokens_details?.cached_tokens == 120)
    }

    @Test func encodesResponseWithToolCalls() throws {
        let response = OpenAI.ChatCompletionResponse(
            id: "chatcmpl-tc",
            model: "qwen3.5-9b-paro",
            created: 1712345678,
            choices: [
                OpenAI.ChatCompletionChoice(
                    index: 0,
                    finish_reason: .tool_calls,
                    message: OpenAI.ResponseMessage(
                        role: .assistant,
                        tool_calls: [
                            OpenAI.ToolCall(
                                id: "call_xyz",
                                type: "function",
                                function: OpenAI.FunctionCall(name: "bash", arguments: #"{"command":"ls"}"#)
                            )
                        ]
                    )
                )
            ]
        )

        let data = try JSONEncoder().encode(response)
        let decoded = try JSONDecoder().decode(OpenAI.ChatCompletionResponse.self, from: data)

        #expect(decoded.choices[0].finish_reason == .tool_calls)
        #expect(decoded.choices[0].message.tool_calls?.count == 1)
        #expect(decoded.choices[0].message.tool_calls?[0].function?.name == "bash")
    }

    @Test func encodesResponseWithReasoningContent() throws {
        let response = OpenAI.ChatCompletionResponse(
            id: "chatcmpl-reason",
            model: "qwen3.5-9b-paro",
            created: 1712345678,
            choices: [
                OpenAI.ChatCompletionChoice(
                    index: 0,
                    finish_reason: .stop,
                    message: OpenAI.ResponseMessage(
                        role: .assistant,
                        content: "Final answer",
                        reasoning_content: "Thinking..."
                    )
                )
            ]
        )

        let data = try JSONEncoder().encode(response)
        let decoded = try JSONDecoder().decode(OpenAI.ChatCompletionResponse.self, from: data)

        #expect(decoded.choices[0].message.content == "Final answer")
        #expect(decoded.choices[0].message.reasoning_content == "Thinking...")
    }

    // MARK: - Streaming Chunk

    @Test func encodesStreamingChunk() throws {
        let chunk = OpenAI.ChatCompletionChunk(
            id: "chatcmpl-stream",
            model: "qwen3.5-9b-paro",
            created: 1712345678,
            choices: [
                OpenAI.ChatCompletionChunkChoice(
                    index: 0,
                    delta: OpenAI.ChunkDelta(role: .assistant, content: "Hello")
                )
            ]
        )

        let data = try JSONEncoder().encode(chunk)
        let decoded = try JSONDecoder().decode(OpenAI.ChatCompletionChunk.self, from: data)

        #expect(decoded.object == "chat.completion.chunk")
        #expect(decoded.choices[0].delta.role == .assistant)
        #expect(decoded.choices[0].delta.content == "Hello")
        #expect(decoded.choices[0].finish_reason == nil)
    }

    @Test func encodesStreamingToolCallChunk() throws {
        let chunk = OpenAI.ChatCompletionChunk(
            id: "chatcmpl-stream",
            model: "qwen3.5-9b-paro",
            created: 1712345678,
            choices: [
                OpenAI.ChatCompletionChunkChoice(
                    index: 0,
                    delta: OpenAI.ChunkDelta(
                        tool_calls: [
                            OpenAI.ToolCall(id: "call_abc", type: "function", function: OpenAI.FunctionCall(name: "bash", arguments: ""), index: 0)
                        ]
                    )
                )
            ]
        )

        let data = try JSONEncoder().encode(chunk)
        let decoded = try JSONDecoder().decode(OpenAI.ChatCompletionChunk.self, from: data)

        #expect(decoded.choices[0].delta.tool_calls?.count == 1)
        #expect(decoded.choices[0].delta.tool_calls?[0].index == 0)
        #expect(decoded.choices[0].delta.tool_calls?[0].function?.name == "bash")
    }

    @Test func encodesStreamingReasoningChunk() throws {
        let chunk = OpenAI.ChatCompletionChunk(
            id: "chatcmpl-stream",
            model: "qwen3.5-9b-paro",
            created: 1712345678,
            choices: [
                OpenAI.ChatCompletionChunkChoice(
                    index: 0,
                    delta: OpenAI.ChunkDelta(reasoning_content: "Thinking...")
                )
            ]
        )

        let data = try JSONEncoder().encode(chunk)
        let decoded = try JSONDecoder().decode(OpenAI.ChatCompletionChunk.self, from: data)

        #expect(decoded.choices[0].delta.reasoning_content == "Thinking...")
        #expect(decoded.choices[0].delta.content == nil)
    }

    @Test func encodesFinalChunkWithUsage() throws {
        let chunk = OpenAI.ChatCompletionChunk(
            id: "chatcmpl-stream",
            model: "qwen3.5-9b-paro",
            created: 1712345678,
            choices: [
                OpenAI.ChatCompletionChunkChoice(
                    index: 0,
                    delta: OpenAI.ChunkDelta(),
                    finish_reason: .stop
                )
            ],
            usage: OpenAI.Usage(prompt_tokens: 100, completion_tokens: 50, total_tokens: 150)
        )

        let data = try JSONEncoder().encode(chunk)
        let decoded = try JSONDecoder().decode(OpenAI.ChatCompletionChunk.self, from: data)

        #expect(decoded.choices[0].finish_reason == .stop)
        #expect(decoded.usage?.total_tokens == 150)
    }

    // MARK: - Models Endpoint

    @Test func encodesModelListResponse() throws {
        let response = OpenAI.ModelListResponse(
            data: [
                OpenAI.ModelObject(
                    id: "qwen3.5-9b-paro",
                    type: "llm",
                    owned_by: "tesseract",
                    max_context_length: 131072,
                    loaded_context_length: 131072,
                    state: "loaded"
                )
            ]
        )

        let data = try JSONEncoder().encode(response)
        let decoded = try JSONDecoder().decode(OpenAI.ModelListResponse.self, from: data)

        #expect(decoded.object == "list")
        #expect(decoded.data.count == 1)
        #expect(decoded.data[0].id == "qwen3.5-9b-paro")
        #expect(decoded.data[0].object == "model")
        #expect(decoded.data[0].type == "llm")
        #expect(decoded.data[0].owned_by == "tesseract")
        #expect(decoded.data[0].max_context_length == 131072)
        #expect(decoded.data[0].state == "loaded")
    }

    @Test func encodesEmptyModelList() throws {
        let response = OpenAI.ModelListResponse(data: [])
        let data = try JSONEncoder().encode(response)
        let decoded = try JSONDecoder().decode(OpenAI.ModelListResponse.self, from: data)
        #expect(decoded.data.isEmpty)
    }

    // MARK: - MessageContent

    @Test func messageContentTextValueJoinsParts() {
        let content = OpenAI.MessageContent.parts([
            OpenAI.ContentPart(type: .text, text: "Hello"),
            OpenAI.ContentPart(type: .image_url, image_url: OpenAI.ImageURL(url: "data:image/png;base64,abc")),
            OpenAI.ContentPart(type: .text, text: "World"),
        ])
        #expect(content.textValue == "Hello\nWorld")
    }

    @Test func messageContentTextValueReturnsNilForImageOnlyParts() {
        let imageOnly = OpenAI.MessageContent.parts([
            OpenAI.ContentPart(type: .image_url, image_url: OpenAI.ImageURL(url: "data:image/png;base64,abc")),
        ])
        #expect(imageOnly.textValue == nil)

        let empty = OpenAI.MessageContent.parts([])
        #expect(empty.textValue == nil)
    }
}
