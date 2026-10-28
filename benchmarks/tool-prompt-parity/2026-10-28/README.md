# Tool prompts against transformers — where they differed, and what it cost

The app's tool prompts did not render as Hugging Face `transformers` renders
them. A check of LFM3's templates found token-for-token parity everywhere but
the tool schemas, and the same held on every template the catalog ships. This
folder holds the measurements behind ADR-0098 and the probe that measured
what the difference does to tool calling.

## What differed, and where it came from

For a `get_weather` tool, transformers 5.21 writes

```
{"name": "get_weather", "description": "Weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
```

and the app wrote (LFM3's `tool_to_json` macro hand-writes the outer `", "`
and `": "`; Qwen's templates dump the whole tool with `tojson`)

```
{"description": "Weather for a city.", "name": "get_weather", "parameters": {"properties":{"city":{"type":"string"}},"required":["city"],"type":"object"}}
```

Three layers, each enough on its own:

1. **Compact separators**: the swift-transformers fork's own carry `fe95f0a`
   replaced swift-jinja's `tojson` with a `JSONEncoder` one. swift-jinja 2.5.0
   and later render `tojson` as `json.dumps(ensure_ascii=False)`, like
   transformers; the carry predates that and kept the encoder for sorted keys.
   The encoder also wrote `2.0` as `2` and an indented render as `"k" : v`.
2. **Sorted keys**: swift-jinja's `Value(any:)` sorts a Swift dictionary's
   keys, and a `ToolSpec` is one. transformers keeps the dict's insertion
   order, which for every OpenAI-style builder is `type`, `function`; `name`,
   `description`, `parameters`; `type`, `properties`, `required`.
3. **Sorted parameters**: `JSONSchema.properties` was a `[String:
   PropertySchema]`, so a tool's declared parameter order (`path`, `offset`,
   `limit`) was gone before any template ran.

Tool calls in the history matched once separators did, for string, integer,
array and object arguments given in sorted order (the app parses arguments
into a dictionary; see Not covered).

## Before and after

The app's file, agenda and profile tools (13), a system and a user message,
through the production loader (`AppTokenizerLoader`). "Before" is the old
pipeline (fe95f0a's `tojson`, no declared order), reproduced in one build; it
matched the old build's count on the parity suite's tools (591 tokens).

| Template | Before | After (= transformers) |
| --- | --- | --- |
| Qwen3.5 (`a4aee8af`) | 1,932 tokens | 2,320 (+388, +20.1%) |
| Qwen3.8 (`c3cf9e34`) | 1,970 tokens | 2,358 (+388, +19.7%) |
| LFM3 (`f45005c5`) | 1,697 tokens | 1,972 (+275, +16.2%) |

After the change the 13-tool prompt is byte-identical to transformers'
render of the same tools written as Python dicts in source order, on all
three. The parity suite's four tools: 591 → 715 on Qwen3.5 (transformers:
715), 629 → 753 on Qwen3.8, 386 → 479 on LFM3, 385 → 478 on LFM3-Audio.
About 30 more tokens per tool, paid once per cached system prefix.

The catalog templates with no tokenizer on this machine, fetched from their
repos and rendered through the same seam with the 13 tools and a tool-call
history: Qwen3.5-2B, Qwen3.6-27B (z-lab and mlx-community), Qwen3.6-35B-A3B
(z-lab and unsloth) and Ornith-1.0-9B are byte-identical to transformers.
Nanbeige4.2's tools and its JSON tool calls are too, but its render starts
with one `\n` that transformers does not write (below). Ornith-1.0-35B's
repo was not readable (HTTP 401).

## What it does to tool calling

`probe_tool_order.py`: mlx-community/Qwen3.8-27B-4bit (a catalog agent model),
the 13 tools above, 30 single-turn requests with a known right call, thinking
off. Each variant's tools block is prefilled once; per task, the
teacher-forced negative log-likelihood of `<tool_call>\n<function=NAME>\n` and
a greedy decode checked for the right function and argument values, in any
order. ΔNLL is paired against the transformers render, 95% bootstrap interval;
"worse" counts the tasks where the variant's NLL is higher.

| Variant | Tools block | Right call | NLL | ΔNLL vs transformers | Worse / better |
| --- | --- | --- | --- | --- | --- |
| transformers (the app now) | 2,330 | 28/30 | 0.157 | — | — |
| the app before | 1,942 | 26/30 | 0.330 | +0.173 [+0.078, +0.295] | 24 / 6 |
| sorted keys, spaced | 2,306 | 26/30 | 0.266 | +0.108 [+0.051, +0.170] | 27 / 3 |
| conventional keys, sorted parameters | 2,330 | 27/30 | 0.194 | +0.037 [+0.016, +0.068] | 24 / 6 |
| transformers order, compact | 1,975 | 29/30 | 0.210 | +0.053 [+0.004, +0.124] | 19 / 11 |

What the decodes show (`probe-qwen3.8-27b.txt`):

- **The model fills parameters in the order the schema lists them.** With
  sorted parameters it wrote `recall(days:query:)`, `agenda(days:from:)`,
  `add_reminder(due:title:)`, `update_reminder(completed:id:)`; with the
  declared order, `query`, `from`, `title`, `id` first.
- **That order changes the call.** "Block 2026-10-29 from 09:00 to 11:00 for
  Deep work": with the declared order, `add_event(title:start:end:)`; with
  sorted parameters, in every sorted variant,
  `add_event(duration_minutes: 120, notes: "Deep work", start:, title:)`.
- **The old render once answered without calling.** "Remind me to buy milk":
  "Done — I've added a reminder to buy milk…" and no `add_reminder`; sorted
  keys with spaces did the same there and on "Cancel my focus block E-8". The
  transformers render called the tool in both.

Thirty tasks are too few to rank the variants by accuracy; the paired NLL
moves the same way on 24 of 30 tasks, and separators and key order each
account for part of it. Two misses are common to most variants and are not
about rendering: reading README.md before editing it, and writing `True` for
a boolean argument.

## Reproduce

```bash
python3 tesseractTests/Fixtures/ToolPromptParity/generate.py OUT.json qwen3.5=MODEL_DIR …  # the parity reference
scripts/test.sh ToolPromptParityTests ChatTemplateToolsTests
python3 -I probe_tool_order.py ~/Library/Application\ Support/models/mlx-community_Qwen3.8-27B-4bit OUT.json  # ~10 min on an M3 Max
python3 -I probe_stats.py OUT.json
```

## Not covered

- **Tools an HTTP client sends**: `OpenAI.ToolDefinition` decodes through
  `JSONDecoder` into dictionaries, so a client's parameter order and any
  schema keyword outside the conventional ones render sorted. Keeping them
  needs an order-preserving parse of the request's `tools`.
- **MCP tools**: the MCP SDK's `JSONValue` has no key order; their parameters
  render sorted.
- **JSON `null` in a client schema**: `LLMActor.canonicalizeToolSpecs` turns
  it into `""` before the render; transformers writes `null`.
- **Argument order in a re-rendered tool call**: arguments are parsed into a
  dictionary and render sorted. The Emitted Path Index keys server-generated
  turns on what the model fed, so this changes no cache outcome.
- **Boolean and null arguments in Qwen3.5-family templates**: they write a
  non-container argument with `| string`, which swift-jinja 2.5.1 prints as
  `true` / `null` and Jinja2 as `True` / `None` (the probe's model writes
  `True`). swift-jinja 3.0 prints them as Python does (its #87); the fork
  allows only 2.x. Qwen3.8's template uses `tojson` for them and matches.
- **Nanbeige4.2's first token**: its template opens with two blank lines
  before `{%- macro …`; Jinja2's `{%-` strips both, swift-jinja 2.5.1 leaves
  one, so every Nanbeige prompt starts with `\n`. Not about tools; swift-jinja
  3.0 reworked whitespace control in its lexer (its #93), untested here.
