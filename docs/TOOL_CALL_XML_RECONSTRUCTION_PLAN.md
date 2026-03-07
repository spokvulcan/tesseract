# Implementation Plan: XML Function Tool Call Reconstruction

*Date: 2026-03-07*

## Problem

When Qwen 3.5 generates tool calls, they use the **XML function format**:

```xml
<tool_call>
<function=read>
<parameter=path>
/tmp/file.txt
</parameter>
</function>
</tool_call>
```

But when we reconstruct prior assistant tool calls in the conversation context (so the model can see what it called in earlier turns), we use **Qwen3-Instruct JSON format**:

```xml
<tool_call>
{"name":"read","arguments":{"path":"/tmp/file.txt"}}
</tool_call>
```

This means the model generates in one format but sees its own history in a different format. The Jinja chat template would produce XML function format for `message.tool_calls`, but we bypass that by inlining reconstructed text into the assistant message content.

The model tolerates this (both formats share the same `<tool_call>` token boundaries), but it's a mismatch that could degrade multi-turn tool calling quality — the model was trained on XML function format for Qwen 3.5, and feeding it JSON history creates an inconsistency in the context window.

## Scope

Three files need changes:

| File | Change | Priority |
|------|--------|----------|
| `tesseract/Features/Agent/Core/MessageConversion.swift` | Reconstruct tool calls in XML function format | **High** |
| `tesseract/Features/Agent/AgentTokenizer.swift` | Add `<tool_response>` / `</tool_response>` token IDs | Low |
| `tesseract/Features/Agent/Benchmark/BenchmarkTranscript.swift` | Match XML function format in transcript output | Low (cosmetic) |

## Implementation Steps

### Step 1: Update `reconstructAssistantContent` in `MessageConversion.swift`

**Current code** (`MessageConversion.swift:44-62`):

```swift
private func reconstructAssistantContent(
    _ content: String, toolCalls: [ToolCallInfo]?
) -> String {
    guard let toolCalls, !toolCalls.isEmpty else { return content }
    var result = content
    for call in toolCalls {
        let argsFragment: String
        if call.argumentsJSON.isEmpty {
            argsFragment = "{}"
        } else if let normalized = ToolArgumentNormalizer.decode(call.argumentsJSON) {
            argsFragment = ToolArgumentNormalizer.encode(normalized)
        } else {
            argsFragment = call.argumentsJSON
        }
        result += "\n<tool_call>\n{\"name\":\"\(call.name)\",\"arguments\":\(argsFragment)}\n</tool_call>"
    }
    return result
}
```

**New code:**

```swift
private func reconstructAssistantContent(
    _ content: String, toolCalls: [ToolCallInfo]?
) -> String {
    guard let toolCalls, !toolCalls.isEmpty else { return content }
    var result = content
    for call in toolCalls {
        result += "\n<tool_call>\n<function=\(call.name)>\n"
        // Decode JSON arguments into key-value pairs for XML parameter format
        if let args = ToolArgumentNormalizer.decode(call.argumentsJSON) {
            for (key, value) in args {
                let valueStr = formatParameterValue(value)
                result += "<parameter=\(key)>\n\(valueStr)\n</parameter>\n"
            }
        }
        result += "</function>\n</tool_call>"
    }
    return result
}

/// Format a JSONValue as a string for XML parameter content.
/// Objects and arrays are serialized as JSON; strings and scalars as plain text.
private func formatParameterValue(_ value: JSONValue) -> String {
    switch value {
    case .string(let s):
        return s
    case .int(let i):
        return String(i)
    case .double(let d):
        return String(d)
    case .bool(let b):
        return b ? "true" : "false"
    case .null:
        return "null"
    case .array, .object:
        // Complex values stay as JSON (matches template behavior)
        if let data = try? JSONEncoder().encode(value),
           let str = String(data: data, encoding: .utf8) {
            return str
        }
        return "{}"
    }
}
```

**What this produces** (matching the Jinja template output):

```xml
<tool_call>
<function=read>
<parameter=path>
/tmp/file.txt
</parameter>
</function>
</tool_call>
```

**Key considerations:**
- `ToolCallInfo.argumentsJSON` is a raw JSON string → use `ToolArgumentNormalizer.decode()` to get `[String: JSONValue]`
- Argument iteration order: dictionaries are unordered, but the template also iterates `tool_call.arguments|items` which is similarly unordered in Jinja. The model doesn't depend on parameter order.
- When `argumentsJSON` is empty or unparseable, produce `<function=name>\n</function>` (no parameters).
- For complex argument values (arrays, nested objects), serialize as JSON inside the `<parameter>` tag — this matches what the template does via `tojson`.

### Step 2: Update comment/docstring

Update the docstring on `reconstructAssistantContent` and the file-level comment in `MessageConversion.swift` to reflect the XML function format.

**Current** (`MessageConversion.swift:14-18`):
```swift
/// Since Tesseract uses local models with XML-based `<tool_call>` tags (not native
/// JSON tool calling), assistant messages reconstruct tool calls as inline XML in the
/// content string.
```

**New:**
```swift
/// Reconstructs assistant tool calls as inline XML in the content string,
/// using Qwen 3.5's XML function format (`<function=name><parameter=key>value</parameter></function>`)
/// inside `<tool_call>` token boundaries.
```

### Step 3: Add `<tool_response>` tokens to AgentTokenizer (optional)

**File:** `AgentTokenizer.swift`

Add two new token IDs to `SpecialTokens`:

```swift
struct SpecialTokens: Sendable {
    // ... existing tokens ...
    /// `<tool_response>` — tool result block start.
    let toolResponseStart: Int
    /// `</tool_response>` — tool result block end.
    let toolResponseEnd: Int
}
```

Resolve them in `resolveSpecialTokens`:

```swift
guard let toolResponseStart = tokenizer.convertTokenToId("<tool_response>") else {
    throw AgentTokenizerError.missingSpecialToken("<tool_response>")
}
guard let toolResponseEnd = tokenizer.convertTokenToId("</tool_response>") else {
    throw AgentTokenizerError.missingSpecialToken("</tool_response>")
}
```

**Note:** These aren't strictly needed now (the template handles tool response wrapping), but having them resolved enables future use for context boundary detection, stop-token logic, or custom prompt construction.

### Step 4: Update BenchmarkTranscript (cosmetic)

**File:** `BenchmarkTranscript.swift:77-83`

Update `writeToolCalls` to emit XML function format for consistency with what the model actually sees:

```swift
func writeToolCalls(calls: [(name: String, arguments: [String: JSONValue])]) {
    for call in calls {
        lines.append("<tool_call>")
        lines.append("<function=\(call.name)>")
        for (key, value) in call.arguments {
            let valueStr = formatParameterValue(value)
            lines.append("<parameter=\(key)>")
            lines.append(valueStr)
            lines.append("</parameter>")
        }
        lines.append("</function>")
        lines.append("</tool_call>")
    }
    lines.append("")
}
```

This is purely cosmetic (transcripts are for human debugging), but it makes them reflect the actual format the model receives.

## Verification

### Build check
```bash
scripts/dev.sh build
```

### Functional test

1. Start agent, ask something that requires a multi-turn tool calling chain (e.g. "read file X, then edit line Y")
2. Check transcript output (`/tmp/tesseract-debug/benchmark/`) — tool call reconstruction should show XML function format
3. Verify the model successfully reads its own prior tool calls and continues the chain

### Benchmark
```bash
# Run existing benchmarks to ensure no regression
# (benchmarks exercise multi-turn tool calling)
```

All 7 scenarios (S1-S7) should still pass. The model should handle XML function reconstruction at least as well as JSON (likely better, since it matches training format).

## Risk Assessment

**Low risk.** The change only affects how *historical* tool calls appear in the context window. The model already generates in XML function format. Aligning history to match is strictly an improvement.

The only edge case: if `ToolArgumentNormalizer.decode()` returns `nil` for malformed JSON, we fall back to an empty `<function=name></function>` block. This is safe — the model will see the tool was called with no args, which is better than showing garbled JSON.

## Reference: Qwen 3.5 Chat Template (Relevant Excerpt)

From `chat_template.jinja` — how the template reconstructs `message.tool_calls`:

```jinja
{%- if message.tool_calls and message.tool_calls is iterable %}
    {%- for tool_call in message.tool_calls %}
        {%- if tool_call.function is defined %}
            {%- set tool_call = tool_call.function %}
        {%- endif %}
        {{- '<tool_call>\n<function=' + tool_call.name + '>\n' }}
        {%- if tool_call.arguments is defined %}
            {%- for args_name, args_value in tool_call.arguments|items %}
                {{- '<parameter=' + args_name + '>\n' }}
                {%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value is sequence and args_value is not string) else args_value | string %}
                {{- args_value }}
                {{- '\n</parameter>\n' }}
            {%- endfor %}
        {%- endif %}
        {{- '</function>\n</tool_call>' }}
    {%- endfor %}
{%- endif %}
```

Note how it formats complex values (objects/arrays) with `tojson|safe` and simple values with `string`. Our `formatParameterValue` mirrors this logic.

## Reference: Qwen 3.5 Special Tokens

| Token ID | Content | Type | Notes |
|----------|---------|------|-------|
| 248044 | `<\|endoftext\|>` | special=true | EOS/padding |
| 248045 | `<\|im_start\|>` | special=true | ChatML role start |
| 248046 | `<\|im_end\|>` | special=true | ChatML role end |
| 248058 | `<tool_call>` | special=false | Single token, in model output |
| 248059 | `</tool_call>` | special=false | Single token, in model output |
| 248066 | `<tool_response>` | special=false | Single token, template-injected |
| 248067 | `</tool_response>` | special=false | Single token, template-injected |
| 248068 | `<think>` | special=false | Single token, in model output |
| 248069 | `</think>` | special=false | Single token, in model output |

`<function=...>`, `</function>`, `<parameter=...>`, `</parameter>` are **NOT** special tokens — they are regular BPE sub-tokens.

## Sources

- Qwen3.5-35B-A3B `tokenizer.json` — `~/.cache/huggingface/hub/models--RepublicOfKorokke--Qwen3.5-35B-A3B-mlx-lm-mxfp4/`
- Qwen3.5-27B `chat_template.jinja` — `~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-27B-4bit/`
- [Qwen-Agent NousFnCallPrompt](https://github.com/QwenLM/Qwen-Agent/blob/main/qwen_agent/llm/fncall_prompts/nous_fncall_prompt.py)
- [Qwen Function Calling Docs](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
- [Ollama Qwen3.5 Tool Calling Bug](https://github.com/ollama/ollama/issues/14493) — documents the XML vs JSON format difference
- [llama.cpp Qwen3-Coder Parser Request](https://github.com/ggml-org/llama.cpp/issues/15012) — XML function format spec
- [Open WebUI Native Tool Calling](https://github.com/open-webui/open-webui/discussions/19326)
- `docs/NATIVE_TOOL_CALLING_RESEARCH.md` — full special token analysis
