# ADR-0098: Tool prompts render as transformers renders them

- Status: Accepted
- Date: 2026-10-28
- Supersedes: the `tojson` carry of `docs/swift-transformers-fork.md`
  (`fe95f0a`), dropped by `c99afdf`
- Relates to: ADR-0063 (its rejected alternative "accommodating the model on
  the render side" was about echo fidelity; this decision is about the text
  the model reads, and the Emitted Path Index stays as decided)

## Context

Every chat template the catalog ships writes the tools into the system
prompt with Jinja's `tojson`: Qwen3.5, Qwen3.6, Qwen3.8, Ornith and
Nanbeige dump each tool (`tool | tojson`), and LFM3's `tool_to_json` macro
dumps each value of the function. The models were trained on the text
Hugging Face transformers renders there: `json.dumps(x, ensure_ascii=False)`,
`", "` and `": "` separators, the dict's insertion order. The app rendered
`{"function":{"description":…,"name":…,"parameters":{"properties":{…},"required":[…],"type":"object"}},"type":"function"}`:
compact, because the swift-transformers fork replaced swift-jinja's `tojson`
with a `JSONEncoder` one, and sorted, because a `ToolSpec` is a Swift
dictionary that swift-jinja converts with its keys sorted and the app's
`JSONSchema` kept its parameters in a dictionary too. Everything else in the
LFM3 and Qwen templates already matched transformers token for token.

On Qwen3.8-27B with the app's 13 file, agenda and profile tools, the old
render cost 0.17 nats on the called function's name against transformers'
render (24 of 30 tasks worse), once answered "Done — I've added a reminder"
without calling `add_reminder`, and made the model fill parameters in
alphabetical order: `add_event(duration_minutes:notes:start:title:)` for a
block given a title, a start and an end.
`benchmarks/tool-prompt-parity/2026-10-28/README.md` has the run.

## Decision

1. **`tojson` is swift-jinja's.** swift-jinja 2.5.0 renders it as
   transformers does; the fork drops its override and requires 2.5.0. A Swift
   dictionary still renders with its keys sorted, the same on every launch:
   swift-jinja sorts them when it converts the value.
2. **The tools reach the template in their builders' key order.** At the
   tokenizer seam (`AppTokenizerLoader`), `ChatTemplateTools` turns each
   `ToolSpec` into a Jinja object and passes the list as the
   `additionalContext` entry `tools`, which replaces the tokenizer's own
   sorted conversion; `tools` still selects a `tool_use` template. The keys
   OpenAI-style builders write come first in their order (`type`,
   `function`, `name`, `description`, `parameters`, `properties`,
   `required`, `enum`, `items`), any other key after them sorted, and data
   values (`enum`, `default`, …) keep sorted keys. The order is a function of
   the tool's content.
3. **A tool's parameters keep their declared order.** `JSONSchema.properties`
   is ordered, and `AgentToolDefinition.toolSpec` carries the order as a list
   of names under `x-tesseract-property-order`. The `ToolSpec` stays plain
   JSON, so `canonicalizeToolSpecs`, the vendor's `XMLFunctionParser`, the
   `ArgumentTranscoder` and every digest of the tools read it as before, and
   a digest covers the order. `ChatTemplateTools` applies the list to the
   function's parameters and drops it, so no template sees it. A tool
   without the list (an HTTP client's, an MCP server's) has its parameters
   sorted.

## Consequences

- A tools block is about 30 tokens per tool longer: the 13-tool prompt grows
  from 1,970 to 2,358 tokens on Qwen3.8 (+19.7%), 1,932 to 2,320 on Qwen3.5,
  1,697 to 1,972 on LFM3. It is paid when a system prefix is first
  prefilled; the prefix caches hold it after that.
- Every render with tools changes once, so prefixes and Emitted Path entries
  stored under the old render miss on the first request after the update.
- A render of an app tool that bypasses `AppTokenizerLoader` (the
  `#huggingFaceTokenizerLoader()` path) would show the order list to the
  model. No production tool render takes that path: Proofread and the
  Memory embedder pass no tools, the MTP drafter renders nothing, and
  `PrefixDetectBenchRunner` builds its own tools.
- `ToolPromptParityTests` compares the tools block, a tool-call history and a
  nested argument token for token against transformers on Qwen3.5, Qwen3.8,
  LFM3 and LFM3-Audio (each skipped without its model), from
  `Fixtures/ToolPromptParity/transformers.json`; `ChatTemplateToolsTests`
  checks the ordering offline.

## Alternatives considered

- **Keep the sorted, compact render.** It is fewer tokens, and the probe
  says it calls tools worse; the token saving does not buy the difference.
- **Spaced separators, sorted keys** (the fork change alone). Recovers part
  of the gap (0.11 nats against 0.17) and leaves the model reading
  `description` before `name` and the parameters alphabetically.
- **Conventional key order, sorted parameters** (no declared order). 0.04
  nats from transformers, but the `add_event` call above still fills
  `duration_minutes` and `notes` first.
- **An ordered `ToolSpec` through the vendor.** `MLXLMCommon.ToolSpec` is a
  dictionary in every signature from `UserInput` to the tokenizer protocol,
  and the vendor's parser reads it as one; changing the type is a fork-wide
  change for one ordering.
- **The order outside the `ToolSpec`** (a registry keyed by tool content).
  Leaves the digests blind to the order, and a tool rendered before its
  definition registered would render differently later for the same content.

## As built

- swift-transformers `c99afdf`, the head of `pin-tesseract-2026-09-15` (removes
  `ChatTemplateFilters.swift`, swift-jinja ≥ 2.5.0, two offline tests); the
  app resolves swift-jinja 2.5.1.
- `ChatTemplateTools.swift`, `AppTokenizerLoader.swift`,
  `AgentToolDefinition.swift` (ordered `JSONSchema.properties`),
  `MCPToolAdapter.swift` (MCP parameters sorted, deterministic).
- Not covered: an HTTP client's parameter order (its request decodes into
  dictionaries), MCP parameter order, JSON `null` in a client schema
  (`canonicalizeToolSpecs` writes `""`), and booleans written with `| string`
  in Qwen3.5-family tool calls (`true` here, `True` in Jinja2, until
  swift-jinja 3).
