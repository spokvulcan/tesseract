# Agent System Prompt Engineering: Tools & Skills Across Frameworks

A comparative analysis of how four major AI agent frameworks describe tools and skills inside their system prompts.

## Table of Contents

1. [Claude Code (Anthropic)](#1-claude-code-anthropic)
2. [OpenAI Codex CLI](#2-openai-codex-cli)
3. [PydanticAI](#3-pydanticai)
4. [NousResearch Hermes](#4-nousresearch-hermes)
5. [Comparison Summary](#5-comparison-summary)

---

## 1. Claude Code (Anthropic)

**Sources**: [Piebald-AI/claude-code-system-prompts](https://github.com/Piebald-AI/claude-code-system-prompts), [system_prompts_leaks](https://github.com/asgeirtj/system_prompts_leaks/blob/main/Anthropic/claude-code.md), [Claude Code Skills docs](https://code.claude.com/docs/en/skills), [Mikhail Shilkov's analysis](https://mikhail.io/2025/10/claude-code-skills/), [Mario Zechner's cchistory](https://mariozechner.at/posts/2025-08-03-cchistory/), [HN discussion](https://news.ycombinator.com/item?id=43909409)

### Overview

Claude Code dynamically assembles its system prompt from **110+ strings** that change based on mode (Plan, Explore, Delegate), enabled tools, and configuration. The total prompt ranges from **15,000–40,000+ tokens** depending on session config. The prompt is not a single monolithic string — it's assembled from conditional fragments at runtime.

### System Prompt Organization

The prompt assembles from these categories:

| Category | Count | Purpose |
|----------|-------|---------|
| Identity/Behavior | ~15 fragments | Tone, style, output efficiency, over-engineering avoidance, security |
| Tool Usage Policy | ~12 fragments | Prefer Read over cat, Grep over grep, etc. |
| Tool Descriptions | ~23 tools | Each assembled from multiple sub-fragments |
| System Reminders | ~40 types | Dynamic context: plan mode, file alerts, token budget |
| Agent Prompts | ~30 sub-agents | Explore, Plan, security review, conversation summary |
| Data/Reference | ~25 embedded docs | API refs for Python/TS/Go/Java, Agent SDK patterns |
| Skills | 5 builtin + user-defined | Loaded on-demand via the Skill tool |

### Tool Definition Format

Tools use the **Anthropic API tool format** — each tool is a JSON object with `name`, `description`, and `input_schema` (JSON Schema). The key distinguishing feature is that tool descriptions contain **massive behavioral instructions** — effectively "micro-system-prompts" inside each tool.

The Anthropic API injects tools into the system prompt via XML:

```xml
In this environment you have access to a set of tools you can use to answer the user's question.
You can invoke functions by writing a "<function_calls>" block...

<functions>
<function>
{"description": "Executes a given bash command...", "name": "Bash", "parameters": {...}}
</function>
</functions>
```

Example tool definition:

```json
{
  "name": "Bash",
  "description": "Executes a given bash command and returns its output.\n\nThe working directory persists between commands...\n\n# Instructions\n - If your command will create new directories...\n - IMPORTANT: Avoid using this tool to run `find`, `grep`, `cat`...",
  "parameters": {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "additionalProperties": false,
    "properties": {
      "command": {
        "description": "The command to execute",
        "type": "string"
      },
      "description": {
        "description": "Clear, concise description of what this command does...",
        "type": "string"
      },
      "timeout": {
        "description": "Optional timeout in milliseconds (max 600000)",
        "type": "number"
      }
    },
    "required": ["command"],
    "type": "object"
  }
}
```

### Composite Tool Descriptions

A unique feature: tool descriptions are **assembled from dozens of sub-fragments**. The Bash tool description alone is built from **~45 separate files** covering:

- Overview, working directory persistence, timeout behavior
- 6 "prefer dedicated tool" notes (Read over cat, Grep over grep, Glob over find, Edit over sed, Write over echo)
- 3 git safety fragments (avoid destructive ops, never skip hooks, prefer new commits over amend)
- 6 sleep guidance fragments
- 17 sandbox policy fragments
- Parallel vs sequential command guidance

Descriptions use **template variables** that resolve at runtime:

```
ALWAYS use ${GREP_TOOL_NAME} for search tasks. NEVER invoke `grep` or `rg` as a ${BASH_TOOL_NAME} command.
Use ${TASK_TOOL_NAME} tool for open-ended searches requiring multiple rounds.
```

### Complete Tool List (23 builtin)

| Tool | Tokens | Purpose |
|------|--------|---------|
| `Bash` | ~45 files | Shell command execution |
| `Read` | 469 | File content retrieval |
| `Edit` | 246 | Exact string replacement in files |
| `Write` | 129 | File creation/overwriting |
| `Glob` | 122 | File pattern matching |
| `Grep` | 300 | Content search (ripgrep) |
| `Skill` | 326 | Execute specialized skills |
| `ToolSearch` | 709 | Discover/load deferred tools |
| `Agent` | ~1500 | Launch autonomous sub-agents |
| `TodoWrite` | 2161 | Task list management |
| `TaskCreate` | 528 | Structured task creation |
| `TaskOutput` | ~200 | Retrieve agent results |
| `TaskStop` | ~100 | Terminate background tasks |
| `AskUserQuestion` | 287 | Gather user input |
| `EnterPlanMode` | 878 | Start implementation planning |
| `ExitPlanMode` | 417 | Present plan for approval |
| `EnterWorktree` | 334 | Git worktree isolation |
| `WebFetch` | 297 | URL content retrieval |
| `WebSearch` | 319 | Web search |
| `NotebookEdit` | 121 | Jupyter notebook editing |
| `Computer` | 161 | Chrome browser automation |
| `SendMessageTool` | 1241 | Team/swarm messaging |
| `CronCreate` | 738 | Schedule recurring tasks |

### Skills System

Skills are **on-demand prompt expansions** — not separate tools or sub-agents. They inject instructions into the main conversation only when invoked. This keeps the base system prompt lean.

**Definition format** (`SKILL.md`):
```yaml
---
name: pdf
description: Extract and analyze text from PDF documents. Use when users ask to process or read PDFs.
disable-model-invocation: true    # optional — user-only
user-invocable: false             # optional — Claude-only
allowed-tools: Read, Grep, Glob   # optional — restrict tools
context: fork                     # optional — run in subagent
agent: Explore                    # optional — which subagent type
argument-hint: [issue-number]     # optional — autocomplete hint
model: claude-opus-4-6              # optional — specific model
---

# PDF Processing Skill

Use the extract_text.py script in this folder to extract text from PDFs:

    python3 extract_text.py <input_file>

After extraction, summarize key points in structured format.
```

**System prompt integration** — only skill metadata appears initially (budget: 2% of context window):
```
The following skills are available for use with the Skill tool:

- pdf: Extract and analyze text from PDF documents.
- commit: Use when the user wants to commit changes...
- simplify: Review changed code for reuse, quality, and efficiency...
```

**Invocation** — the model calls the `Skill` tool:
```json
{
  "type": "tool_use",
  "name": "Skill",
  "input": {"skill": "pdf", "args": "document.pdf"}
}
```

The system expands the full `SKILL.md` body (without frontmatter) into the conversation. Supports `$ARGUMENTS`, `$0`, `$1`, `${CLAUDE_SESSION_ID}`, `${CLAUDE_SKILL_DIR}` placeholders and `!`shell commands`` for dynamic context injection.

**Invocation control matrix**:

| Frontmatter | User can invoke | Claude can invoke |
|---|---|---|
| (default) | Yes | Yes |
| `disable-model-invocation: true` | Yes | No |
| `user-invocable: false` | No | Yes |

**Builtin skills**: `/simplify` (728 tks), `/batch` (1136 tks), `/debug` (412 tks), `/loop` (984 tks), `/claude-api` (5137 tks)

### Deferred Tools (Lazy Loading)

Claude Code uses `ToolSearch` to lazy-load tools not needed in every conversation:

```xml
<available-deferred-tools>
AskUserQuestion
WebFetch
WebSearch
mcp__xcode__BuildProject
...
</available-deferred-tools>
```

Tools must be discovered before calling — by keyword search or direct selection (`"select:WebFetch"`). Both modes load the tool immediately.

### Sub-Agent System

The `Agent` tool launches autonomous sub-agents with restricted tool sets:

| Agent Type | Tools | Purpose |
|------------|-------|---------|
| `general-purpose` | All tools | Complex multi-step tasks |
| `Explore` | Read-only tools | Fast codebase search |
| `Plan` | Read-only tools | Architecture design |

Sub-agents get their own system prompts (~30 different agent prompt templates exist).

---

## 2. OpenAI Codex CLI

**Sources**: [openai/codex GitHub](https://github.com/openai/codex), [Codex CLI docs](https://developers.openai.com/codex/cli/), [Codex CLI features](https://developers.openai.com/codex/cli/features/), [PromptLayer analysis](https://blog.promptlayer.com/how-openai-codex-works-behind-the-scenes-and-how-it-compares-to-claude-code/), [prompt.md](https://github.com/openai/codex/blob/main/codex-rs/core/prompt.md), [apply_patch instructions](https://github.com/openai/codex/blob/main/codex-rs/apply-patch/apply_patch_tool_instructions.md)

### Overview

Codex CLI is written in **Rust** (`codex-rs/`) and uses the **OpenAI Responses API**. It takes a more minimalist approach than Claude Code, with a smaller core tool set but a sophisticated tool registration system. The system prompt "literally teaches the model a mini-API."

### System Prompt Structure

The system prompt lives in markdown files (`codex-rs/core/prompt.md`, `gpt_5_2_prompt.md`). Organized in clear sections:

```
You are Codex, based on GPT-5, running as a coding agent in the Codex CLI on a user's computer.
Codex CLI is an open source project led by OpenAI.
You are expected to be precise, safe, and helpful.
```

**Sections (in order)**:
1. **Identity and capabilities** — three capabilities: receive prompts, communicate (streaming thinking/responses, plans), emit function calls
2. **Personality** — "concise, direct, and friendly"
3. **AGENTS.md spec** — how to discover and obey per-directory instruction files
4. **Responsiveness / Preamble messages** — 8-12 word pre-action messages with 8 examples
5. **Planning** — `update_plan` tool usage with quality examples
6. **Task execution** — behavioral rules, `apply_patch` for edits, coding guidelines
7. **Validating your work** — testing philosophy: specific-to-broad
8. **Ambition vs. precision** — creative for greenfield, surgical for existing code
9. **Progress updates** — how to report intermediate progress
10. **Final message formatting** — headers, bullets, monospace, file references, tone

### Tool Definition Format (Rust Implementation)

Codex defines tools as Rust structs serialized to JSON. The `ToolSpec` enum:

```rust
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type")]
pub(crate) enum ToolSpec {
    #[serde(rename = "function")]
    Function(ResponsesApiTool),       // Standard JSON Schema tools

    #[serde(rename = "local_shell")]
    LocalShell {},                     // Built-in shell

    #[serde(rename = "web_search")]
    WebSearch { /* filters, etc. */ }, // Built-in web search

    #[serde(rename = "custom")]
    Freeform(FreeformTool),           // Lark grammar-based tools
}
```

**Function tools** use the OpenAI format:

```json
{
  "type": "function",
  "name": "shell_command",
  "description": "Runs a shell command and returns its output...",
  "strict": false,
  "parameters": {
    "type": "object",
    "properties": {
      "command": {
        "type": "string",
        "description": "The shell script to execute in the user's default shell"
      },
      "workdir": {
        "type": "string",
        "description": "The working directory to execute the command in"
      },
      "timeout_ms": {
        "type": "number",
        "description": "The timeout for the command in milliseconds"
      }
    },
    "required": ["command"],
    "additionalProperties": false
  }
}
```

**Freeform tools** — for tools where input is not structured JSON (like `apply_patch`), Codex uses **Lark grammar-based** definitions:

```rust
pub struct FreeformTool {
    pub name: String,
    pub description: String,
    pub format: FreeformToolFormat,  // type: "grammar", syntax: "lark"
}
```

The `apply_patch` tool uses a Lark CFG grammar (`tool_apply_patch.lark`):

```
start: begin_patch hunk+ end_patch
begin_patch: "*** Begin Patch" LF
end_patch: "*** End Patch" LF?
hunk: add_hunk | delete_hunk | update_hunk
add_hunk: "*** Add File: " filename LF add_line+
delete_hunk: "*** Delete File: " filename LF
update_hunk: "*** Update File: " filename LF change_move? change?
```

### Complete Tool List

| Tool | Type | Purpose |
|------|------|---------|
| `shell` / `shell_command` | Function | Execute shell commands |
| `apply_patch` | Freeform (Lark) | Edit files via diff patches |
| `update_plan` | Function | Track task steps with status |
| `read_file` | Function | Read files with line offsets |
| `list_dir` | Function | List directory contents |
| `grep_files` | Function | Regex search across files |
| `view_image` | Function | View local image files |
| `web_search` | Built-in | Search the web |
| `image_generation` | Built-in | Generate images |
| `spawn_agent` | Function | Spawn sub-agents for parallel work |
| `send_input` | Function | Message an existing sub-agent |
| `wait` | Function | Wait for sub-agent completion |
| `close_agent` | Function | Close a sub-agent |
| `resume_agent` | Function | Resume a closed sub-agent |
| `request_user_input` | Function | Ask user with structured options |
| `request_permissions` | Function | Sandbox permission escalation |
| `js_repl` | Freeform (Lark) | Persistent Node.js REPL |
| `artifacts` | Freeform (Lark) | Create presentations/spreadsheets |
| `search_tools_bm25` | Function | BM25 search over available tools |
| `spawn_agents_on_csv` | Function | Batch process CSV rows |
| MCP tools | Function (dynamic) | From configured MCP servers |

**Naming conventions**: `snake_case`, concise action-oriented descriptions, all `strict: false`, all `additionalProperties: false`.

### The apply_patch Format

Taught inline in the system prompt. Three operations:

```
*** Begin Patch
*** Add File: path/to/new_file.py
+import os
+
+def hello():
+    print("Hello")
*** End Patch
```

```
*** Begin Patch
*** Update File: path/to/existing.py
@@ def existing_function():
-    old_line
+    new_line
*** End Patch
```

```
*** Begin Patch
*** Delete File: path/to/remove.py
*** End Patch
```

Rules: 3 lines context before/after, `@@` markers for disambiguation, `+` prefix for new lines, relative paths only.

### Instructions System (AGENTS.md)

Three-tier precedence:
1. **Global**: `~/.codex/AGENTS.override.md` or `~/.codex/AGENTS.md`
2. **Project**: Walking from repo root to CWD
3. **Nested**: More deeply nested files take precedence

Each file becomes a `user`-role message: `# AGENTS.md instructions for <directory>`

### Skills System

Codex has skills (replacing deprecated custom prompts) with a directory structure:

```
my-skill/
  SKILL.md              # Required — name, description, instructions
  scripts/              # Optional executable scripts
  references/           # Optional reference files
  agents/openai.yaml    # Optional extended configuration
```

**SKILL.md format**:
```yaml
---
name: skill-name
description: Explain scope and boundaries for triggering
---

# Instructions body in Markdown
```

**agents/openai.yaml** (optional extended config):
```yaml
interface:
  display_name: "User-facing name"
  short_description: "User-facing description"
  icon_small: "./assets/small-logo.svg"
  default_prompt: "Optional surrounding prompt"

policy:
  allow_implicit_invocation: false

dependencies:
  tools:
    - type: "mcp"
      value: "toolName"
      description: "Tool description"
```

**Progressive disclosure**: Codex initially receives only skill metadata (name, description). Full SKILL.md instructions load only on invocation — same pattern as Claude Code.

---

## 3. PydanticAI

**Sources**: [PydanticAI tools docs](https://ai.pydantic.dev/tools/), [PydanticAI advanced tools](https://ai.pydantic.dev/tools-advanced/), [PydanticAI agents docs](https://ai.pydantic.dev/agent/), [GitHub](https://github.com/pydantic/pydantic-ai), [Alexander Junge's "Show me the prompt"](https://www.alexanderjunge.net/blog/show-me-the-prompt-pydanticai/), [DeepWiki analysis](https://deepwiki.com/pydantic/pydantic-ai/3-tools-and-function-calling)

### Overview

PydanticAI is a **Python agent framework** that automatically generates tool schemas from function signatures and docstrings. It uses the **native tool/function calling API** of whichever LLM provider it connects to (OpenAI, Anthropic, Google, etc.), acting as a universal translation layer. **Tools are NOT embedded in the system prompt text.**

### Tool Definition Pipeline

```
Python function → inspect.signature() + get_type_hints()
    → Docstring parsing (Google/NumPy/Sphinx via griffe)
    → Pydantic JSON Schema generation (strips `title` from properties)
    → ToolDefinition dataclass
    → Provider-specific mapping (OpenAI/Anthropic/Google)
    → Sent via native API `tools` parameter
```

### Tool Registration Methods

**Decorator with context** (`@agent.tool`):
```python
@agent.tool
def get_player_name(ctx: RunContext[str]) -> str:
    """Get the player's name."""
    return ctx.deps
```

**Decorator without context** (`@agent.tool_plain`):
```python
@agent.tool_plain
def roll_dice() -> str:
    """Roll a six-sided die and return the result."""
    return str(random.randint(1, 6))
```

**Constructor registration**:
```python
agent = Agent(
    'openai:gpt-5-mini',
    tools=[roll_dice, Tool(get_player_name, takes_ctx=True)],
)
```

**Explicit schema** (`Tool.from_schema`):
```python
tool = Tool.from_schema(
    function=foobar,
    name='sum',
    description='Sum two numbers.',
    json_schema={
        'additionalProperties': False,
        'properties': {
            'a': {'description': 'the first number', 'type': 'integer'},
            'b': {'description': 'the second number', 'type': 'integer'},
        },
        'required': ['a', 'b'],
        'type': 'object',
    },
    takes_ctx=False,
)
```

### Schema Generation from Docstrings

```python
@agent.tool_plain(docstring_format='google', require_parameter_descriptions=True)
def foobar(a: int, b: str, c: dict[str, list[float]]) -> str:
    """Get me foobar.

    Args:
        a: apple pie
        b: banana cake
        c: carrot smoothie
    """
    return f'{a} {b} {c}'
```

**Generated JSON Schema**:
```json
{
  "additionalProperties": false,
  "properties": {
    "a": {"description": "apple pie", "type": "integer"},
    "b": {"description": "banana cake", "type": "string"},
    "c": {
      "additionalProperties": {"items": {"type": "number"}, "type": "array"},
      "description": "carrot smoothie",
      "type": "object"
    }
  },
  "required": ["a", "b", "c"],
  "type": "object"
}
```

**Single-parameter simplification**: If a tool has one Pydantic model parameter, the schema collapses to just that model's schema (no wrapper object).

### Internal Representation

The `ToolDefinition` dataclass:

```python
@dataclass
class ToolDefinition:
    name: str                                  # e.g. "foobar"
    parameters_json_schema: ObjectJsonSchema   # The JSON schema dict
    description: str | None                    # From docstring or explicit
    outer_typed_dict_key: str | None           # For output tools
    strict: bool | None                        # OpenAI strict mode
    sequential: bool                           # Serial execution required
    kind: ToolKind                             # 'function', 'output', 'external'
    metadata: dict[str, Any] | None            # Not sent to model
    timeout: float | None                      # Execution timeout
```

### Provider-Specific Mapping

PydanticAI maps `ToolDefinition` to each provider's native format:

**OpenAI**:
```python
def _map_tool_definition(self, f: ToolDefinition) -> ChatCompletionToolParam:
    return {
        'type': 'function',
        'function': {
            'name': f.name,
            'description': f.description or '',
            'parameters': f.parameters_json_schema,
        },
    }
```

**Anthropic/Claude**:
```python
def _map_tool_definition(self, f: ToolDefinition) -> BetaToolParam:
    return {
        'name': f.name,
        'description': f.description or '',
        'input_schema': f.parameters_json_schema,
    }
```

### What the LLM Actually Receives

The complete API request (e.g., to OpenAI):

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_lat_lng",
        "parameters": {
          "type": "object",
          "required": ["location_description"],
          "properties": {
            "location_description": {
              "type": "string",
              "description": "A description of a location."
            }
          }
        }
      }
    }
  ],
  "messages": [
    {
      "role": "system",
      "content": "Be concise, reply with one sentence."
    },
    {
      "role": "user",
      "content": "What is the weather like in London?"
    }
  ]
}
```

**The system prompt contains ONLY the agent's `instructions`** — tool definitions are separate API parameters.

### System Prompt Architecture

**Static prompts** — passed at construction:
```python
agent = Agent('openai:gpt-5.2', system_prompt="Use the customer's name while replying.")
```

**Dynamic prompts** — decorated functions called at runtime:
```python
@agent.system_prompt
def add_the_date() -> str:
    return f'The date is {date.today()}.'
```

**`instructions` vs `system_prompt`**: Both produce system-level content, but `instructions` survive agent handoffs while `system_prompt` doesn't. The docs recommend `instructions` as the default.

### Dynamic Tool Preparation

PydanticAI supports **runtime tool modification** via `prepare` callbacks:

```python
async def prepare_greet(
    ctx: RunContext[Literal['human', 'machine']], tool_def: ToolDefinition
) -> ToolDefinition | None:
    d = f'Name of the {ctx.deps} to greet.'
    tool_def.parameters_json_schema['properties']['name']['description'] = d
    return tool_def  # or return None to HIDE the tool
```

**Agent-wide filtering**:
```python
async def filter_tools(
    ctx: RunContext[bool], tool_defs: list[ToolDefinition]
) -> list[ToolDefinition] | None:
    if ctx.deps:
        return [td for td in tool_defs if td.name != 'launch_potato']
    return tool_defs
```

### No Skills Concept

PydanticAI has no separate "skills" concept. Complex behaviors compose via tools, nested agents (agent-as-tool), and dynamic system prompts.

---

## 4. NousResearch Hermes

**Sources**: [Hermes-Function-Calling GitHub](https://github.com/NousResearch/Hermes-Function-Calling), [Hermes-2-Pro-Llama-3-8B](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B), [Hermes-3-Llama-3.1-8B](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B), [Hermes-4-70B](https://huggingface.co/NousResearch/Hermes-4.3-36B), [Hermes 3 Technical Report](https://arxiv.org/pdf/2408.11857), [hermes-function-calling-v1 dataset](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)

### Overview

Hermes models use an **XML-based, in-prompt tool definition system** built on top of the ChatML format. Unlike API-based approaches (Claude, OpenAI), Hermes embeds tool definitions **directly in the system prompt text** using XML tags. This was designed for open-source models that don't have native function calling APIs, making it the most portable approach.

### System Prompt Template (Exact)

The complete system prompt for function calling (from tokenizer_config.json `tool_use` template):

```
<|im_start|>system
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools:

<tools>
[TOOL_DEFINITIONS_JSON]
</tools>

Use the following pydantic model json schema for each tool call you will make:
{"properties": {"arguments": {"title": "Arguments", "type": "object"}, "name": {"title": "Name", "type": "string"}}, "required": ["arguments", "name"], "title": "FunctionCall", "type": "object"}

For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:

<tool_call>
{"arguments": <args-dict>, "name": <function-name>}
</tool_call>
<|im_end|>
```

### Tool Definition Format

Tools are JSON objects inside `<tools>` XML tags, following OpenAI-style function schema. The Jinja2 chat template auto-generates Python-docstring style descriptions:

```json
<tools>
[
  {
    "type": "function",
    "function": {
      "name": "get_stock_fundamentals",
      "description": "get_stock_fundamentals(symbol: str) -> dict - Get fundamental data for a given stock symbol using yfinance API.\n\n    Args:\n        symbol (str): The stock symbol.\n\n    Returns:\n        dict: A dictionary containing fundamental data.\n            Keys:\n                - 'symbol': The stock symbol.\n                - 'company_name': The long name.\n                - 'sector': The sector.\n                - 'market_cap': Market capitalization.",
      "parameters": {
        "type": "object",
        "properties": {
          "symbol": {
            "type": "string"
          }
        },
        "required": ["symbol"]
      }
    }
  }
]
</tools>
```

The Jinja2 template in `tokenizer_config.json` converts tool definitions into Python-docstring format automatically:

```
function_name(param1: type, param2: type) -> return_type - Description

    Args:
        param1(type): description
        param2(type): description
    Returns:
        return description
```

### Tool Call Format (Model Output)

Single tool call:
```xml
<|im_start|>assistant
<tool_call>
{"arguments": {"symbol": "TSLA"}, "name": "get_stock_fundamentals"}
</tool_call><|im_end|>
```

Multiple tool calls in one turn:
```xml
<|im_start|>assistant
<tool_call>
{"arguments": {"a": 1}, "name": "func1"}
</tool_call>
<tool_call>
{"arguments": {"b": 2}, "name": "func2"}
</tool_call><|im_end|>
```

### Tool Response Format

Single response:
```
<|im_start|>tool
<tool_response>
{"name": "get_stock_fundamentals", "content": {"symbol": "TSLA", "company_name": "Tesla, Inc.", "sector": "Consumer Cyclical", "market_cap": 611384164352}}
</tool_response>
<|im_end|>
```

Multiple responses grouped under single `tool` turn:
```
<|im_start|>tool
<tool_response>
{"name": "func1", "content": {"result": "value1"}}
</tool_response>
<tool_response>
{"name": "func2", "content": {"result": "value2"}}
</tool_response>
<|im_end|>
```

### Complete Conversation Flow

```
<|im_start|>system
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags...
<tools>[{"type": "function", "function": {"name": "get_stock_fundamentals", ...}}]</tools>
...
<|im_end|>

<|im_start|>user
Fetch the stock fundamentals data for Tesla (TSLA)
<|im_end|>

<|im_start|>assistant
<tool_call>
{"arguments": {"symbol": "TSLA"}, "name": "get_stock_fundamentals"}
</tool_call>
<|im_end|>

<|im_start|>tool
<tool_response>
{"name": "get_stock_fundamentals", "content": {"symbol": "TSLA", "company_name": "Tesla, Inc.", ...}}
</tool_response>
<|im_end|>

<|im_start|>assistant
The stock fundamentals data for Tesla (TSLA) are as follows:
- **Symbol**: TSLA
- **Company Name**: Tesla, Inc.
...
<|im_end|>
```

### Special Token Optimization

Critical design decision: `<tools>`, `</tools>`, `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>` are each **single tokens** in the vocabulary:

- **Faster parsing**: tool call start/end emitted in one token
- **Streaming efficiency**: parsers detect tool calls immediately
- **Reduced token count**: XML wrappers cost 1 token each vs multiple

### Hermes 4 Changes

Hermes 4 switched from ChatML to **Llama 3 tokens** (`<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`) but retained the same XML tag structure. It adds `<think>...</think>` reasoning blocks — the model reasons before emitting tool calls and can continue reasoning after receiving responses.

### Jinja2 Chat Template (Key Sections)

**Type conversion macro** (JSON Schema → Python type hints):
```jinja2
{# string→str, number→float, integer→int, boolean→bool, array/object handling #}
```

**Assistant tool call rendering**:
```jinja2
{%- elif message.role == "assistant" %}
    {{- '<|im_start|>' + message.role }}
    {%- for tool_call in message.tool_calls %}
        {{- '\n<tool_call>\n' }}
        {%- if tool_call.function is defined %}
            {%- set tool_call = tool_call.function %}
        {%- endif %}
        {{- '{"name": "' + tool_call.name + '", "arguments": ' }}
        {{- tool_call.arguments|tojson }}
        {{- '}\n</tool_call>' }}
    {%- endfor %}
    {{- '<|im_end|>\n' }}
```

**Tool response grouping** (multiple responses under single `tool` turn):
```jinja2
{%- elif message.role == "tool" %}
    {%- if loop.previtem and loop.previtem.role != "tool" %}
        {{- '<|im_start|>tool\n' }}
    {%- endif %}
    {{- '<tool_response>\n' + message.content + '\n</tool_response>' }}
    {%- if not loop.last and loop.nextitem.role != "tool" %}
        {{- '<|im_end|>' }}
    {%- elif loop.last %}
        {{- '<|im_end|>' }}
    {%- endif %}
```

### No Skills Concept

Hermes has no skills system. All capabilities are tools defined upfront in `<tools>`. Behavior is controlled entirely through system prompt text.

---

## 5. Comparison Summary

### Tool Definition Approach

| Framework | Where tools are defined | Format | Transport |
|-----------|------------------------|--------|-----------|
| **Claude Code** | API `tools` parameter + prose in description | JSON Schema with embedded behavioral guidelines | Anthropic API native tool use |
| **Codex CLI** | API `tools/functions` parameter + Lark grammars | OpenAI function schema + Lark CFGs for freeform tools | OpenAI Responses API |
| **PydanticAI** | Auto-generated from Python functions | JSON Schema from type hints + docstrings | Provider-native API (adapts per LLM) |
| **Hermes** | Embedded in system prompt text | JSON inside `<tools>` XML tags | In-prompt (no separate API parameter) |

### Tool Description Style

| Framework | Style | Verbosity |
|-----------|-------|-----------|
| **Claude Code** | Extensive prose with do/don't rules, examples, and behavioral instructions assembled from ~45 sub-fragments per tool | Very high (~200-500+ words per tool) |
| **Codex CLI** | Concise function descriptions + system prompt sections teaching formats (apply_patch) + Lark grammars for validation | Medium (~50-100 words per tool + prompt sections) |
| **PydanticAI** | Auto-extracted from docstrings; `title` stripped from properties to save tokens | Variable (developer-controlled) |
| **Hermes** | Python-docstring signature + args format in JSON `description` field, auto-generated by Jinja2 template | Medium (~50-200 words, mirrors source docstring) |

### Skills / Extensibility

| Framework | Mechanism | How it works |
|-----------|-----------|--------------|
| **Claude Code** | `Skill` tool + `SKILL.md` files | On-demand prompt expansion — frontmatter listed in system prompt, body injected when invoked; supports invocation control matrix |
| **Codex CLI** | `SKILL.md` + `agents/openai.yaml` + `AGENTS.md` | Progressive disclosure — metadata loaded initially, full instructions on invocation; AGENTS.md for repo-level instructions |
| **PydanticAI** | No separate concept | Everything is a tool; compose via nested agents and dynamic system prompts |
| **Hermes** | None | Static tool list in `<tools>` block; all capabilities defined upfront |

### Key Architectural Differences

1. **Claude Code** embeds **massive behavioral instructions** inside tool descriptions (~45 fragments for Bash alone). Tool descriptions use template variables (`${GREP_TOOL_NAME}`) and are effectively "micro-system-prompts." Unique `ToolSearch` lazy-loading prevents bloat.

2. **Codex CLI** uses **three tool types**: function (JSON Schema), freeform (Lark grammar), and built-in (web_search, etc.). The Lark grammar approach for `apply_patch` and `js_repl` is unique — it provides formal parsing validation rather than relying on the model to follow natural language format descriptions. Has the most tools overall (~20+).

3. **PydanticAI** is the most **developer-ergonomic** — tools are Python functions with type hints. The framework handles all schema generation and multi-provider adaptation. No system prompt engineering needed. The `prepare` callback system allows runtime tool modification/hiding.

4. **Hermes** is the most **self-contained** — everything in the prompt text (no API parameters). Most portable across inference engines. Jinja2 chat template does the heavy lifting of converting tool definitions. Single-token XML tags are a clever optimization.

### Lazy Loading / Dynamic Tools

| Framework | Mechanism |
|-----------|-----------|
| **Claude Code** | `ToolSearch` for deferred tools; `<available-deferred-tools>` listing; MCP tools on demand |
| **Codex CLI** | `search_tools_bm25` for tool discovery; MCP server integration; progressive skill loading |
| **PydanticAI** | `prepare` functions can dynamically show/hide/modify tools per request at runtime |
| **Hermes** | None — all tools must be in the initial `<tools>` block |

### Token Efficiency

| Framework | Approach |
|-----------|----------|
| **Claude Code** | High per-tool cost (verbose descriptions from ~45 fragments) but deferred loading amortizes total |
| **Codex CLI** | Medium — function tools are concise; freeform tools add Lark grammar overhead; `search_tools_bm25` helps discover without loading |
| **PydanticAI** | Variable — depends on docstring length; strips `title` from property schemas; uses API parameters (not prompt tokens for tool defs) |
| **Hermes** | Medium — XML overhead but single-token tags reduce cost significantly |

### Naming Conventions

| Framework | Convention | Examples |
|-----------|-----------|----------|
| **Claude Code** | PascalCase | `Bash`, `Read`, `Edit`, `WebFetch`, `ToolSearch` |
| **Codex CLI** | snake_case | `shell_command`, `apply_patch`, `read_file`, `update_plan` |
| **PydanticAI** | snake_case (from function name) | `roll_dice`, `get_player_name`, `get_lat_lng` |
| **Hermes** | snake_case (from function name) | `get_stock_fundamentals`, `get_current_temperature` |
