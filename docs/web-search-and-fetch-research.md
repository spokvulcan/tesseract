# Web Search & Web Fetch for Tesseract Agent — Feature Research

> **Date**: 2026-03-13
> **Status**: Research / Options Analysis
> **Next step**: Select approach variant, then create technical implementation plan

---

## 1. Problem Statement

Tesseract Agent runs fully offline on Apple Silicon — all LLM inference is local. This is a core differentiator for privacy. However, the agent currently **cannot access any external information**: no current events, no documentation lookups, no fact-checking against live data. This makes it significantly less useful for many real-world tasks.

**User expectations are clear**: 52% of US adults have used an LLM, and two-thirds of those use them "like search engines." Web access is not a niche feature — it is a core expectation. Users of a local agent specifically expect:

1. **Current events & recent information** — the #1 gap, since local models have knowledge cutoffs
2. **Fact checking & verification** — confirm claims against live sources
3. **Documentation lookup** — "How do I use API X?"
4. **Technical troubleshooting** — error messages, Stack Overflow-type queries
5. **Product/service research** — comparisons, reviews, pricing

Users do **not** expect a local agent to be a full web browser, maintain login sessions, or crawl the web in the background. **Search + Fetch covers 90%+ of real-world web access needs.**

---

## 2. How Other Agents Solve This

### 2.1 Claude Code (Anthropic CLI)

Claude Code has two built-in web tools: **WebSearch** and **WebFetch**.

**WebSearch**:
- Uses Anthropic's server-side `web_search` tool, powered by **Brave Search** under the hood
- Search happens entirely on Anthropic's servers — results return `url`, `title`, `page_age`, and `encrypted_content`
- Claude Code discards `page_age` and `encrypted_content`, keeping only titles and URLs
- Supports `allowed_domains` and `blocked_domains` filtering
- Pricing: $10 per 1,000 searches (separate from token costs)
- Only available on Anthropic's first-party API

**WebFetch**:
- Uses **Axios** (Node.js HTTP client) to fetch pages locally
- Converts HTML to Markdown using **Turndown** library
- Content truncated to **100KB** of text
- Processed by **Claude Haiku 3.5** with a user-provided prompt — never returns raw content
- Does **not** execute JavaScript — plain HTTP GET only
- 15-minute TTL cache per URL
- Cross-host redirects require explicit new fetch (security measure)
- Domain safety checks via `claude.ai/api/web/domain_info`

**Key insight**: Claude Code's web tools are tightly coupled to Anthropic's cloud. The WebSearch is a black box. WebFetch is local HTTP + Turndown + a secondary LLM for summarization. This secondary-LLM pattern is worth noting — it prevents dumping raw HTML into the main agent context.

> Sources: [Inside Claude Code's Web Tools](https://mikhail.io/2025/10/claude-code-web-tools/), [The Claude Code WebSearch Black Box](https://www.vmunix.com/posts/claude-websearch-blackbox), [Anthropic Web Search API docs](https://claude.com/blog/web-search-api)

---

### 2.2 OpenClaw (60K GitHub Stars)

OpenClaw is a general-purpose AI assistant with the most flexible web tool implementation among open-source agents.

**web_search** (`src/agents/tools/web-search.ts`):
- Supports **5 providers** with auto-detection fallback chain: Brave → Gemini → Grok → Kimi → Perplexity
- Each provider requires its own API key via environment variables
- Brave: structured results with snippets, supports `llm-context` mode
- Gemini: AI-synthesized answers with Google Search grounding
- Community plugin **claw-search** provides self-hosted search via **SearXNG** — zero tracking, zero API costs

**web_fetch** (`src/agents/tools/web-fetch.ts`):
- Does **not** execute JavaScript — plain HTTP GET only
- Extraction pipeline: **Readability** (main content extraction) → **Firecrawl** API fallback (optional)
- Known limitation: Readability almost always returns non-empty content (even just a `<title>` tag), causing the Firecrawl fallback to rarely trigger
- Limits: 50K chars default, 2MB max response, 15-min cache, 3 max redirects
- SSRF guards: blocks private/internal hostnames, validates redirects

> Sources: [OpenClaw Web Tools docs](https://docs.openclaw.ai/tools/web), [OpenClaw GitHub](https://github.com/openclaw/openclaw), [claw-search GitHub](https://github.com/binglius/claw-search)

---

### 2.3 OpenCode (120K+ GitHub Stars)

The most direct open-source Claude Code alternative.

- **websearch**: Uses **Exa AI** (semantic/neural search) via Exa's hosted MCP service — no API key required from users
- **webfetch**: Uses Bun's `fetch` with browser user-agent spoofing
- Community plugin [opencode-websearch-cited](https://github.com/ghoulr/opencode-websearch-cited) adds inline citations

> Source: [OpenCode GitHub](https://github.com/opencode-ai/opencode)

---

### 2.4 Cursor

- Uses **Exa.ai** (neural/semantic search API) for `@Web` search
- A separate LLM analyzes the message + conversation + current file to formulate the search query
- Built-in browser (Cursor 2.0) with DevTools for previewing/debugging web apps
- Supports **Browser MCP** (browsermcp.io) for advanced browser automation
- No API key required from users — Cursor handles Exa.ai integration internally

> Source: [Cursor @Web docs](https://docs.cursor.com/context/@-symbols/@-web)

---

### 2.5 Aider

- **`/web <url>`** command: fetches and scrapes a specific URL, adds content as chat context
- Uses **httpx** (Python HTTP client) for static pages
- Optional **Playwright** with Chromium for JavaScript-rendered pages
- **BeautifulSoup** for HTML-to-text conversion
- **No built-in search** — only URL fetching. User must provide specific URLs
- No API keys required

> Source: [Aider scrape.py](https://github.com/paul-gauthier/aider/blob/main/aider/scrape.py)

---

### 2.6 Continue.dev

- **@Web** context provider: hits Continue's own proxy endpoint (underlying search engine undisclosed)
- **@Google** provider: uses **Serper.dev** API (Google Search wrapper), requires Serper API key
- **@URL** provider: fetches specific URLs
- Privacy concern noted by users — @Web sends queries to Continue's undocumented proxy

> Source: [Continue.dev Context Providers](https://docs.continue.dev/customization/context-providers)

---

### 2.7 Open Interpreter

Takes a fundamentally different approach — no dedicated web tools. Instead, the LLM generates and executes code (Python/JS/Bash) using libraries like `requests`, `selenium`, or `playwright` as needed. In "OS Mode," it can control a browser's GUI directly.

---

### 2.8 Local-First Agents (Jan, Ollama, GPT4All, LocalAI, AnythingLLM)

| Agent | Web Search | Approach | Notes |
|-------|-----------|----------|-------|
| **Ollama** | Yes (2025) | First-party API (`ollama.com/api/web_search`), 5 results default, max 10 | Recommends 32K+ context. Injected as `role: tool` messages |
| **GPT4All** | Beta (2024) | **Brave Search API**, user provides key in settings | Closest precedent to Tesseract's use case |
| **AnythingLLM** | Yes | 15+ providers (SearXNG, Brave, Tavily, DDG, Kagi...) | Agent invoked via `@agent <prompt>` |
| **LocalAI** | Yes | Agent system with web search, MCP support (Oct 2025) | Drop-in OpenAI API replacement |
| **LM Studio** | Via MCP (2025) | Tavily MCP, Brave Search MCP servers | Not built-in — requires MCP config |
| **Jan.ai** | No | Limited tool-calling, web via community extensions | |
| **PrivateGPT** | No (by design) | RAG-only, no web access — contradicts core premise | |

**Key finding**: Among privacy-first local agents, web search is universally implemented as an **opt-in feature** routing queries through an external search API. No local-first agent attempts to do web crawling/indexing locally — they all acknowledge that search requires external infrastructure.

---

### 2.9 Unsloth Studio (Agentic Web Search — Deep Dive)

Unsloth Studio is a local-first UI for running and training open models on-device. Its agentic web search implementation is notable for its simplicity, self-healing tool call parsing, and the way multi-step research emerges from its agentic loop without any special "deep research" mode.

> Sources: [Unsloth GitHub](https://github.com/unslothai/unsloth), [Unsloth Studio docs](https://unsloth.ai/docs/new/studio), [Training AI Agents with RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/training-ai-agents-with-rl)

#### Architecture (3 Layers)

| Layer | Stack | Key Files |
|-------|-------|-----------|
| **Inference Backend** | Python / FastAPI | `studio/backend/core/inference/tools.py`, `llama_cpp.py`, `studio/backend/routes/inference.py` |
| **Model Server** | llama.cpp (GGUF) | Spawned as subprocess, OpenAI-compatible `/v1/chat/completions` |
| **Frontend** | React / TypeScript | `studio/frontend/src/features/chat/api/chat-adapter.ts`, `tool-ui-web-search.tsx` |

The inference backend acts as an **orchestrator** between the frontend and a llama-server subprocess. Tool calls are detected, executed, and re-injected into the conversation before the next model turn — all server-side.

#### Search Tool: DuckDuckGo via `ddgs`

Search is implemented in `studio/backend/core/inference/tools.py` using the `ddgs` Python library (DuckDuckGo search scraper). No API key required.

```python
def _web_search(query: str, max_results: int = 5, timeout: int = _EXEC_TIMEOUT) -> str:
    from ddgs import DDGS
    results = DDGS(timeout=timeout).text(query, max_results=max_results)
    parts = []
    for r in results:
        parts.append(
            f"Title: {r.get('title', '')}\n"
            f"URL: {r.get('href', '')}\n"
            f"Snippet: {r.get('body', '')}"
        )
    return "\n\n---\n\n".join(parts)
```

Key characteristics:
- **5 results** by default, plaintext formatted with `---` separators
- **Error-resilient**: catches all exceptions and returns the error string to the model
- **300-second timeout** per tool execution
- **No API key** required (uses DuckDuckGo HTML scraping under the hood)

Three tools are defined total: `web_search`, `python` (sandboxed code exec), and `terminal` (sandboxed bash). All use OpenAI function-calling schema format.

#### No Web Fetch — Snippets Only (Significant Limitation)

**Unsloth Studio has no `web_fetch` tool whatsoever.** This was verified exhaustively:

- `tools.py` defines exactly 3 tools: `ALL_TOOLS = [WEB_SEARCH_TOOL, PYTHON_TOOL, TERMINAL_TOOL]`. Any unknown tool name returns `"Unknown tool: {name}"`.
- No HTTP/scraping libraries (`beautifulsoup4`, `readability`, `html2text`, `trafilatura`, etc.) appear in `studio/backend/requirements/studio.txt`.
- No function in the entire repo fetches URL content for the purpose of page extraction.
- No GitHub issues or PRs requesting web fetch exist.

**What the model actually receives per search**: DuckDuckGo snippets are typically **150-250 characters** each. With 5 results, the model gets roughly **1,000 characters (~250 tokens)** of actual search content per query. This is enough for factual lookups (dates, definitions, "who won X") but grossly insufficient for any real research — the model never sees article content, documentation pages, or detailed technical information.

**Partial workaround via `terminal` tool**: The bash blocklist (`rm`, `sudo`, `dd`, `chmod`, `mkfs`, `shutdown`, `reboot`) does not block `curl` or `wget`. A sufficiently capable model *could* emergently run `curl https://example.com` to fetch raw HTML via the terminal tool. However, this is unintentional — raw HTML without extraction is noisy, token-wasteful, and unreliable.

**This is a meaningful gap.** Every other agent in this research document that implements web search also implements web fetch (or uses a search API that returns substantial content like Tavily/Exa). Unsloth's snippet-only approach is the most minimal implementation surveyed — it works for quick factual lookups but cannot support deep research, documentation reading, or any task requiring actual page content.

#### The Agentic Loop

The core loop lives in `LlamaCppBackend.generate_chat_completion_with_tools()` (`studio/backend/core/inference/llama_cpp.py`, ~line 1646). This is the most architecturally significant piece:

```
for iteration in range(max_tool_iterations):     # default: 10
    1. Send NON-STREAMING request to llama-server with tools=[...], tool_choice="auto"
    2. Parse response for tool_calls (native structured OR auto-healed from XML)
    3. If tool_calls found:
       a. Append assistant message (with tool_calls) to conversation
       b. For EACH tool call:
          - Yield SSE {"type": "tool_start", ...}
          - Execute tool via execute_tool()
          - Yield SSE {"type": "tool_end", ...}
          - Append {"role": "tool", "content": result} to conversation
       c. CONTINUE loop → model processes tool results
    4. If NO tool_calls:
       - Break to final streaming pass

# AFTER loop: Final STREAMING pass with full conversation context
# Model synthesizes response incorporating all tool results
```

**Critical design decisions**:

1. **Non-streaming for tool detection, streaming for final response only.** Tool detection passes use `stream: false` because the full response is needed to detect tool calls. Only the final synthesis response streams to the user. This is a key tradeoff: the user sees nothing during tool rounds, then gets a streaming final answer.

2. **Up to 10 iterations** (`max_tool_calls_per_message`). The model can chain multiple search→refine→search-again rounds, enabling multi-step research without any explicit "deep research" mode.

3. **Conversation accumulation.** Each tool call and its result are appended to the conversation, so the model has full context of all prior searches and findings when deciding whether to search again or synthesize.

4. **"Deep research" is shallow in practice.** There is no separate deep research pipeline. Multi-step search emerges from the loop (search → read snippets → refine query → search again), but without a `web_fetch` tool, the model only ever sees ~250-char DuckDuckGo snippets — never actual page content. This limits research to triangulating facts across multiple snippet-level previews rather than reading and synthesizing full articles or documentation.

#### Self-Healing Tool Call Parser (Novel Technique)

This is the most distinctive feature. Many open models emit tool calls as XML text in their content rather than as structured `tool_calls` objects. Unsloth auto-detects and parses these with `_parse_tool_calls_from_text()` (~line 1212):

**Pattern 1 — JSON inside XML tags:**
```xml
<tool_call>{"name":"web_search","arguments":{"query":"python async"}}</tool_call>
```

**Pattern 2 — XML-style parameter tags:**
```xml
<tool_call><function=web_search><parameter=query>python async</parameter></function></tool_call>
```

Key parser behaviors:
- **All closing tags are optional** — models frequently omit `</tool_call>`, `</function>`, `</parameter>`
- Uses **balanced-brace extraction** for JSON bodies
- Careful boundary detection: uses `</tool_call>` and next `<function=` as boundaries, but NOT `</function>` since code parameters may contain that literal string
- After extraction, XML markup is **stripped from content text** so the user never sees raw tool-call XML
- The final streaming pass also strips tool markup via `_strip_tool_markup()`, with a distinction between "closed" patterns (safe to strip mid-stream) and "open-ended" patterns (only stripped on final flush to avoid non-monotonic content)

**Relevance for Tesseract**: This technique is directly applicable. Local GGUF models running on MLX may not reliably produce structured tool calls. A self-healing parser that extracts tool calls from XML/text output would significantly improve tool-calling reliability without requiring model-specific chat templates.

#### Tool Support Auto-Detection

Tool calling is only enabled for models whose GGUF metadata contains a Jinja2 chat template referencing tool/function roles:

```python
tool_markers = [
    "{%- if tools %}", "{% if tools %}",
    '"role" == "tool"', "'role' == 'tool'",
    'message.role == "tool"', "message.role == 'tool'",
]
self._supports_tools = any(marker in chat_template for marker in tool_markers)
```

If the chat template doesn't reference tools, tool calling is silently disabled. Safetensors models (loaded via transformers, not llama.cpp) do NOT currently support tool calling in Unsloth Studio.

#### Frontend SSE Protocol

Three event types during tool calling:

| SSE Event | Payload | Frontend Action |
|-----------|---------|-----------------|
| `tool_status` | `{"type": "tool_status", "content": "Searching: ..."}` | Status bar text |
| `tool_start` | `{"type": "tool_start", "tool_name": "web_search", "tool_call_id": "...", "arguments": {...}}` | Create tool-call UI part in loading state |
| `tool_end` | `{"type": "tool_end", "tool_name": "web_search", "tool_call_id": "...", "result": "..."}` | Show result, transition to complete |

The chat adapter (`chat-adapter.ts`) accumulates `toolCallParts` as an array and yields them alongside text parts. When a web_search completes, `parseSourcesFromResult()` extracts structured citations from the `Title: / URL: / Snippet:` format. The `WebSearchToolUI` component renders results as clickable links with favicons, auto-collapsing when the model starts generating text.

#### Sandbox and Safety

Each session gets an isolated directory under `~/studio_sandbox/{session_id}/`:
- **Path traversal prevention**: strips `..`, verifies resolved paths stay under sandbox root
- **Bash command blocklist**: `rm`, `sudo`, `dd`, `chmod`, `mkfs`, `shutdown`, `reboot`
- **Python AST analysis**: `_check_signal_escape_patterns()` detects signal manipulation that could escape timeouts
- **Cancellation**: all execution supports cancellation via `threading.Event`

#### Configurable Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `enable_tools` | None | Master toggle for tool calling |
| `enabled_tools` | None | Whitelist: `["web_search", "python", "terminal"]` |
| `auto_heal_tool_calls` | True | Auto-detect XML tool markup in text output |
| `max_tool_calls_per_message` | 10 | Maximum agentic loop iterations |
| `tool_call_timeout` | 300 | Per-tool execution timeout (seconds) |
| `session_id` | None | Sandbox isolation key |

#### Agent Training Pipeline (ART / GRPO)

Separately from the runtime, Unsloth provides tooling for **training** models to be better at tool calling:

- **ART (Agent Reinforcement Trainer)**: Built on Unsloth's `GRPOTrainer`. Uses "trajectories" capturing multi-turn agent execution sequences (tool calls + responses), scored via GRPO
- **RULER (Relative Universal LLM-Elicited Rewards)**: Automatic zero-shot reward function eliminating hand-crafted reward signals
- **NeMo Gym-style environments**: Agent server orchestrates interaction, resource server implements tools as HTTP endpoints with session isolation, verification logic scores outcomes

This is a training-time pipeline — the trained models then run through the same `generate_chat_completion_with_tools` loop at inference time.

#### Key Takeaways for Tesseract

| Aspect | Unsloth Approach | Tesseract Implications |
|--------|-----------------|----------------------|
| **Search provider** | DuckDuckGo (free, no API key) | Validates Variant B as viable for zero-config fallback |
| **No URL fetching** | Snippet-only (~250 chars each), no page content extraction | **Critical gap.** Tesseract must implement `web_fetch` — search without fetch is insufficient for real research. Every other mature agent (Claude Code, OpenClaw, Cursor, Aider) pairs search with fetch |
| **Self-healing tool calls** | Parses XML/text tool calls from models that don't output structured JSON | Directly applicable — local MLX models may emit XML tool calls. Implement similar fallback parser in `AgentEngine` |
| **Agentic loop** | Non-streaming tool rounds + streaming final pass, up to 10 iterations | Similar pattern to Tesseract's double-loop. Non-streaming detection is a pragmatic choice |
| **"Deep research"** | No special mode — multi-step search emerges from the loop, but limited to snippet triangulation without fetch | Validates that a general agentic loop is sufficient architecture-wise, but the tool set must include fetch for research to be meaningful |
| **Tool support detection** | Chat template inspection for tool role markers | Relevant for GGUF models; less so for MLX safetensors which use transformers tokenizers |
| **Sandbox isolation** | Per-session directories, command blocklists, AST analysis | Tesseract already has `PathSandbox` — comparable approach |

---

## 3. Comparison Summary

| Agent | Search Provider | Fetch Method | JS Rendering | API Key | Offline LLM |
|-------|----------------|-------------|-------------|---------|-------------|
| **Claude Code** | Brave (via Anthropic) | Axios + Turndown + Haiku | No | Anthropic sub | No |
| **OpenClaw** | Brave/Gemini/Grok/Kimi/Perplexity | Readability + Firecrawl | No (fetch); Yes (Firecrawl) | Per provider | No |
| **OpenCode** | Exa.ai | Bun fetch | No | No | No |
| **Cursor** | Exa.ai | Built-in browser | Yes | No (internal) | No |
| **Aider** | None (URL only) | httpx + BeautifulSoup + Playwright | Optional | No | Yes |
| **Continue.dev** | Proxy / Serper | URL provider | No | Serper key | No |
| **Unsloth Studio** | DuckDuckGo (`ddgs`) | None (snippets only) | No | No | Yes (GGUF) |
| **GPT4All** | Brave Search | N/A | No | Brave key | Yes |
| **Ollama** | Ollama API | N/A | No | No | Yes |
| **AnythingLLM** | 15+ providers | Web scraper | No | Varies | Yes |

---

## 4. Available Search Providers

### 4.1 Free / No-API-Key Options

#### DuckDuckGo (Community Libraries)
- No official API for organic web results — relies on community wrappers scraping DDG HTML
- Python: [`duckduckgo-search`](https://pypi.org/project/duckduckgo-search/), Node: [`ddgs`](https://github.com/eudalabs/ddgs)
- **Free, no API key** required
- Rate limit: ~1 req/sec. Exceeding triggers `RatelimitException`. VPN IPs blocked more aggressively
- **Verdict**: Good zero-config fallback. Fragile for heavy use. Could be reimplemented in Swift via HTTP scraping

#### SearXNG (Self-Hosted Meta Search)
- [Open-source](https://github.com/searxng/searxng) meta-search engine aggregating **246+ search services**
- API: `GET /search?q=<query>&format=json` — returns `url`, `title`, `content`, `publishedDate`, `engine`, `score`
- **Completely free. No API key. No rate limits** (user controls the instance)
- Runs via Docker on the same Mac as Tesseract
- Privacy: excellent — no tracking, no profiling, all data stays local
- Used by OpenClaw's community plugin (claw-search)
- **Verdict**: Maximum privacy, zero cost. Requires Docker setup (friction for non-technical users)

#### Jina AI Search (`s.jina.ai`)
- `GET https://s.jina.ai/?q=<query>` — returns top 5 results with content in LLM-friendly text
- Free API key: 100 RPM, 10M token trial
- **Verdict**: Extremely simple to integrate (single HTTP GET). Good complement to a fetch tool

### 4.2 Paid Search APIs (Recommended Tier)

| Provider | Free Tier | Paid Price | Quality | AI-Optimized | Privacy | Best For |
|----------|----------|-----------|---------|-------------|---------|----------|
| **Brave Search** | $5/mo credits (~1K queries) | $5/1K queries | Independent index, good | Supports `llm-context` mode | No tracking, GDPR, Zero Data Retention option | Privacy-aligned agents |
| **Tavily** | 1,000 credits/mo | $0.008/credit | Purpose-built for AI | Yes — structured JSON, summaries, citations | Standard | AI agents needing rich results |
| **Serper.dev** | 2,500 free queries | $1/1K queries | Google-quality (proxies Google) | Structured JSON | Standard | Best value for Google results |
| **Exa.ai** | 1,000 req/mo | $7/1K requests | Neural/semantic search | Yes — understands meaning | Zero-data-retention option | Semantic queries beyond keywords |
| **Google CSE** | 100 queries/day | $5/1K queries | Google quality | No | Standard | Google results directly |
| **Perplexity Sonar** | None | $5/1K requests | AI-native with citations | Yes | Standard | AI-synthesized answers |

**Bing Search API**: Retired August 2025. Replacement ("Grounding with Bing") costs $35/1K queries — not recommended.

> Sources: [Brave Search API](https://api-dashboard.search.brave.com/documentation/pricing), [Tavily](https://www.tavily.com/pricing), [Serper.dev](https://serper.dev/), [Exa.ai](https://exa.ai/pricing), [Bing retirement](https://ppc.land/microsoft-ends-bing-search-apis-on-august-11-alternative-costs-40-483-more/)

### 4.3 MCP Servers for Web Search

Existing MCP servers that could be used as Tesseract extensions:

| MCP Server | Provider | Notes |
|-----------|----------|-------|
| **Brave Search MCP** | [Brave official](https://github.com/brave/brave-search-mcp-server) | Web, local, pagination, freshness controls |
| **Fetch MCP** | [MCP reference](https://github.com/modelcontextprotocol/servers) | URL fetching + markdown conversion |
| **DuckDuckGo MCP** | Community | DDG search via MCP |
| **You.com MCP** | You.com | Real-time web search |
| **Tavily MCP** | Tavily | AI-optimized search |

**MCP relevance for Tesseract**: MCP servers are typically Node.js processes communicating via stdio JSON-RPC. Tesseract could spawn them as subprocesses (how Claude Code and Cursor do it), but for built-in tools it is simpler to call search APIs directly from Swift. MCP integration is better suited as an optional extension mechanism.

---

## 5. Web Fetch & Content Extraction

### 5.1 HTML-to-Markdown Conversion Approaches

| Approach | Language | Quality (F1) | Speed | JS Required | Notes |
|----------|----------|-------------|-------|------------|-------|
| **Mozilla Readability** | JavaScript | 0.970 median | Fast | JavaScriptCore/WKWebView | Highest median accuracy. Firefox Reader View algorithm |
| **Trafilatura** | Python | 0.937 mean | Fast | No | Highest precision (0.978). Best overall mean performance |
| **Turndown** | JavaScript | Good | Fast | JavaScriptCore/WKWebView | Used by Claude Code. HTML → Markdown conversion |
| **Crawl4AI** | Python | Good | Medium | Optional | 58K+ GitHub stars. Multiple output modes. Self-hosted |
| **Jina Reader API** | Cloud | Good | ~200ms | No (cloud-side) | `r.jina.ai/<url>`. Free 20 RPM / 500 RPM with key |
| **ReaderLM-v2** | Any (1.5B model) | Good | ~1s | No | Self-hostable Jina model. 512K tokens. Could run on MLX |
| **Firecrawl** | Cloud | Good | ~2s | Yes (cloud-side) | 500 free credits, then $16+/mo |

**Key finding from benchmarks**: Heuristic models (Readability, Trafilatura) generally **outperform** neural models for content extraction, even on complex web pages.

> Sources: [Readability GitHub](https://github.com/mozilla/readability), [Web Content Extraction Benchmark](https://chuniversiteit.nl/papers/comparison-of-web-content-extraction-algorithms), [Jina Reader](https://jina.ai/reader/), [ReaderLM-v2](https://huggingface.co/jinaai/ReaderLM-v2)

### 5.2 Swift/macOS Specific Libraries

| Library | Type | Dependencies | Notes |
|---------|------|-------------|-------|
| **SwiftSoup** | HTML parsing (DOM, CSS selectors) | Pure Swift, SPM | Standard Swift HTML parser. jQuery-like API |
| **Demark** | HTML → Markdown | WKWebView (wraps Turndown.js) | ~100ms first conversion. By Peter Steinberger |
| **SwiftHTMLToMarkdown** | HTML → Markdown | Pure Swift, no WKWebView | Lighter but less capable with malformed HTML |
| **HTMLToMarkdown** | HTML → Markdown | JavaScriptCore | Another pure-Swift option |
| **WKZombie** | Headless browsing | WKWebView + libxml2 | Navigate, fill forms, extract data without UI |

> Sources: [SwiftSoup](https://github.com/scinfu/SwiftSoup), [Demark](https://steipete.me/posts/2025/introducing-demark-html-to-markdown-in-swift), [SwiftHTMLToMarkdown](https://github.com/ActuallyTaylor/SwiftHTMLToMarkdown)

### 5.3 JavaScript-Rendered Pages

For the ~15% of pages requiring JS execution:

- **WKWebView** (best fit for macOS): Create offscreen at 1x1, load URL, wait for `didFinish`, extract DOM via `evaluateJavaScript("document.documentElement.outerHTML")`. Use `WKWebsiteDataStore.nonPersistent()` for privacy
- **Playwright**: Cross-browser, powerful, but requires Node.js/Python runtime + browser binaries (150-400MB). Too heavy for a macOS app
- **macOS 26**: Apple introduced native SwiftUI `WebView` + `WebPage` (Observable), which may simplify headless rendering

**Verdict for Tesseract**: WKWebView is the natural choice — already available on macOS, runs JavaScript natively, no external dependencies. Playwright is overkill.

### 5.4 Content Extraction Best Practices for LLMs

- **Markdown is the native format for LLMs** — clean markdown improves RAG accuracy by ~35% and reduces tokens by 20-30%
- Strip `<script>`, `<style>`, `<nav>`, `<footer>`, `<aside>`, ad containers, cookie banners before conversion
- Preserve semantic structure: headings, lists, tables, code blocks, blockquotes
- Convert links to `[text](url)` format
- Preserve `alt` text for images; discard image data unless vision processing is available
- **Truncation strategy**: Apply Readability-style extraction first (removes 50-80% of noise), then token-aware truncation with markers like `[Content truncated at {N} tokens]`

### 5.5 Token Budget for Web Content

- **Ollama**: recommends minimum 32K context for web-search models
- **Claude Code WebFetch**: 100KB text limit, then summarized by secondary LLM
- **Tavily**: 500 chars/chunk × 3 chunks/source × 5-20 results
- **Practical consensus**: 5 search results is the sweet spot. 8-16K tokens per web tool call
- **Tesseract context**: 120K window, 16K reserve, 20K recent — roughly 8-16K tokens available for web content per tool call

---

## 6. Privacy Considerations

### 6.1 Data Flow Boundaries

| Stays Local | Transits Externally |
|------------|-------------------|
| Model weights & inference | Search queries (to Brave/Tavily/etc.) |
| Conversation history | Raw web page fetches (URLs) |
| Extracted/processed content | - |
| User files & documents | - |

**Critical principle**: Only the search query string leaves the device. Conversation context, user files, and previous responses are never sent to search APIs.

### 6.2 Privacy-Preserving Fetch Implementation

- Use `URLSessionConfiguration.ephemeral` — no persistent cookies, no disk cache, no credential storage
- Do not send cookies from the user's browser — the agent fetches in a completely isolated session
- Strip `Referer` headers
- Set honest User-Agent: `TesseractAgent/1.0 (macOS; Apple Silicon)` — do not impersonate browsers
- Respect `robots.txt` (ethical + legal protection)
- Clear any WKWebView data stores after extraction (`WKWebsiteDataStore.nonPersistent()`)
- Do not persist fetched content beyond the current conversation unless explicitly saved
- Consider DNS-over-HTTPS (DoH) for DNS-level privacy (macOS supports natively)

### 6.3 User Consent Patterns

Based on best practices from privacy-first agents:

- **Off by default** — user explicitly enables web access
- **Per-session or per-request consent** — not a blanket permanent permission
- **Visible queries** — show what the agent is searching for in the UI
- **API key ownership** — user provides their own key (stored in Keychain), not a shared key
- **Least privilege** — read-only web search, no form submission or authentication

---

## 7. Implementation Variants

### Variant A: "Minimal — API Search + Native Fetch" (Recommended for MVP)

**Search**: User provides a Brave Search API key (or Tavily). Agent calls the search API when it needs current information. Returns 5 results with titles, URLs, and snippets.

**Fetch**: Native `URLSession` (ephemeral) fetches the URL. SwiftSoup parses and cleans HTML. SwiftHTMLToMarkdown (or Demark) converts to markdown. Token-aware truncation at ~50KB.

| Pros | Cons |
|------|------|
| Simple to implement in Swift | Requires user to obtain API key |
| No external runtime dependencies | No JS rendering for SPAs |
| Privacy-preserving (ephemeral sessions) | Brave free tier limited (~1K queries/mo) |
| Follows GPT4All / AnythingLLM proven pattern | Cannot handle login-protected pages |
| 2 new tools alongside existing read/write/edit/ls | |

**Effort**: Low-medium. Similar complexity to existing built-in tools.

---

### Variant B: "DuckDuckGo Free + Native Fetch" (Zero-Config)

**Search**: DuckDuckGo HTML scraping — no API key needed. Reimplemented in Swift.

**Fetch**: Same as Variant A.

| Pros | Cons |
|------|------|
| Zero configuration, works out of the box | Fragile — DDG blocks aggressive scrapers |
| No API key, no account, no cost | Rate limited to ~1 req/sec |
| True zero-dependency setup | No official API, could break anytime |
| | Lower result quality than Brave/Google |

**Effort**: Medium. Requires reverse-engineering DDG's HTML structure and maintaining it.

---

### Variant C: "SearXNG Self-Hosted + Native Fetch" (Maximum Privacy)

**Search**: User runs SearXNG locally via Docker. Tesseract calls `localhost:8888/search?q=...&format=json`. Aggregates results from 246+ search engines.

**Fetch**: Same as Variant A.

| Pros | Cons |
|------|------|
| Maximum privacy — search never leaves the machine | Requires Docker (high friction for non-technical users) |
| Zero cost, no API keys | User must maintain SearXNG instance |
| Aggregates 246+ search engines | Docker adds resource overhead |
| Used by OpenClaw community (claw-search) | Google results require CAPTCHA solving at scale |

**Effort**: Low (for the Tesseract integration). High (for the user's setup).

---

### Variant D: "Multi-Provider with Fallback Chain" (OpenClaw-Style)

**Search**: Support multiple providers — Brave → Tavily → DuckDuckGo (fallback). User configures their preferred provider and API key in Settings. Agent tries them in order.

**Fetch**: Native fetch + optional Jina Reader API (`r.jina.ai`) for difficult pages.

| Pros | Cons |
|------|------|
| Flexible — user picks their provider | More complex settings UI |
| Graceful degradation if one provider fails | More code to maintain |
| Can add providers over time | Testing matrix grows |
| Follows OpenClaw's proven pattern | |

**Effort**: Medium-high. Requires provider abstraction layer.

---

### Variant E: "Jina All-in-One" (Simplest Integration)

**Search**: Jina Search API (`s.jina.ai`) — single HTTP GET returns top 5 results with extracted content.

**Fetch**: Jina Reader API (`r.jina.ai`) — single HTTP GET returns page as clean markdown.

| Pros | Cons |
|------|------|
| Simplest possible integration (URL prefix) | Sends all URLs through Jina's servers |
| Both search and fetch in one provider | Free tier: 10M tokens, then paid |
| Returns LLM-ready content, minimal post-processing | Single point of failure |
| Free API key with generous trial | Privacy concern — Jina sees all queries/URLs |

**Effort**: Very low. Two HTTP GET calls.

---

### Variant F: "WKWebView Browser Tool" (Full Browsing)

**Search**: Any search API (Brave recommended).

**Fetch**: WKWebView-based headless browser — renders JS, extracts full DOM, converts to markdown. Can handle SPAs, dynamic content, even basic interactions.

| Pros | Cons |
|------|------|
| Handles JS-heavy pages (SPAs, React apps) | Significantly more complex |
| No external dependencies (WKWebView is native) | Resource intensive (spinning up WebKit instances) |
| Could evolve into full browser-use tool | Security surface area increases |
| Handles more of the web correctly | Slower than plain HTTP fetch |

**Effort**: High. WKWebView offscreen rendering, DOM extraction, timeout handling, memory management.

---

### Variant G: "Hybrid — Simple Fetch + WebPage Fallback" (Recommended)

**Search**: Brave Search API (or any from Variant D).

**Fetch (Tier 1)**: URLSession + swift-readability + Demark — fast, handles ~85% of pages.

**Fetch (Tier 2)**: If Tier 1 returns < 200 chars from 10KB+ HTML, retry with macOS 26 `WebPage` headless rendering — handles JS-heavy SPAs natively. No WKWebView visibility hacks needed.

| Pros | Cons |
|------|------|
| Best coverage — handles both static and dynamic pages | Two code paths to maintain |
| Fast path for most pages, WebPage fallback for SPAs | Heuristic for "suspiciously little content" may misfire |
| WebPage is `@Observable`, native Swift concurrency | Requires macOS 26 (already our target) |
| No view hierarchy hack — WebPage works standalone | Each WebPage spawns ~50-150 MB WebContent process |
| Follows Aider's two-tier pattern | |
| No external runtime dependencies | |

**Effort**: Medium. WebPage integration is cleaner than WKWebView — no delegates, no visibility tricks. Amortizable — Tier 1 first, Tier 2 later.

---

## 8. Recommended Approach for Tesseract

### Phase 1 (MVP): Variant A + elements of D

1. **Two new built-in tools**: `web_search` and `web_fetch` in `Features/Agent/Tools/BuiltIn/`
2. **Search**: Brave Search API as primary (privacy-aligned, independent index, official MCP server exists). Store API key in Keychain via Settings. DuckDuckGo as zero-config fallback (no key needed)
3. **Fetch**: `URLSession` (ephemeral) → swift-readability (content extraction via SwiftSoup) → Demark html-to-md engine (markdown conversion via JSC, ~5ms) → token-aware truncation (~50KB / ~12K tokens). WKWebView fallback for JS-rendered SPAs (see Section 11.1). See Section 11 for detailed implementation research
4. **Settings**: "Web Access" toggle (off by default), search provider picker, API key field
5. **UI**: Show search queries in the agent conversation (transparency). Badge/indicator when web tools are active
6. **Defaults**: 5 results, snippets only. Fetch limited to 50KB extracted text

### Phase 2 (Enhancement):

7. Add Tavily and/or Serper as alternative search providers
8. WebPage (macOS 26) fallback for JS-heavy SPAs (Variant G's Tier 2) — headless, `@Observable`, no WKWebView hack needed. See Section 11.1 for full API reference
9. SearXNG support for power users who self-host

### Phase 3 (Future):

10. Jina Reader as an optional extraction enhancement
11. MinerU-HTML / Dripper-style 0.5B classification model on MLX for high-accuracy extraction (~0.90 F1 vs Readability's 0.65 on diverse pages). Apache 2.0 license. See Section 11.3 for benchmarks. **Not ReaderLM-v2** — it scores only 0.23 F1 on independent benchmarks despite Jina's own claims (CC-BY-NC license also blocks commercial use)
12. Full browser-use tool via WebPage API for interactive tasks (no WKWebView hack needed)

---

## 9. Tool Schema Design

Based on patterns from Ollama, Claude Code, OpenClaw, and Tavily:

### `web_search`

```
web_search:
  description: Search the web for current information
  parameters:
    query: string (required) — the search query
    max_results: integer (optional, default 5, range 1-10)
  returns:
    results: [{title, url, snippet}]
```

### `web_fetch`

```
web_fetch:
  description: Fetch a web page and extract its content as markdown
  parameters:
    url: string (required) — the URL to fetch
    max_chars: integer (optional, default 50000) — maximum characters to return
  returns:
    content: string (markdown-formatted page content)
    title: string (page title)
    url: string (final URL after redirects)
```

---

## 11. Local Web Fetch — Deep Implementation Research

> **Date**: 2026-03-25
> **Focus**: How to implement `web_fetch` locally on macOS — rendering JS-heavy pages, extracting clean content, and the role of small local models

---

### 11.1 Rendering JS-Heavy Pages (React, Next.js, SPAs)

The core problem: ~15% of pages serve empty HTML shells (`<div id="root"></div>`) and render all content via JavaScript. A plain `URLSession` GET returns no useful content for these pages.

#### WKWebView Headless Rendering (Current Best Option)

WKWebView is the natural choice on macOS — no external dependencies, built-in JS execution. However, it was not designed for headless operation and requires workarounds.

**The visibility hack**: WKWebView throttles/stops JS execution when it detects it's not in the view hierarchy. The proven workaround (used by [WKZombie](https://github.com/mkoehnke/WKZombie)):

```swift
// Must be attached to real view hierarchy with non-zero size and alpha
if let window = NSApplication.shared.keyWindow, let view = window.contentView {
    webView.frame = CGRect(origin: .zero, size: view.frame.size)
    webView.alphaValue = 0.01  // Near-invisible but "visible" to WebKit
    view.addSubview(webView)
}
```

Using `CGRect.zero` or zero alpha is insufficient — the WebContent process becomes inactive.

**Detecting when SPAs finish rendering**: `WKNavigationDelegate.didFinish` fires too early (before React/Vue mounts). Use a MutationObserver idle-detection pattern via injected `WKUserScript`:

```javascript
// Injected at .atDocumentEnd, forMainFrameOnly: true
(function() {
    let timer = null;
    const IDLE_MS = 1500;  // No DOM mutations for 1.5s = "done"

    const observer = new MutationObserver(() => {
        clearTimeout(timer);
        timer = setTimeout(() => {
            observer.disconnect();
            window.webkit.messageHandlers.renderComplete.postMessage(
                document.documentElement.outerHTML
            );
        }, IDLE_MS);
    });

    observer.observe(document.body || document.documentElement, {
        childList: true, subtree: true, attributes: true
    });

    // Fallback: if no mutations within 3s, assume static page
    timer = setTimeout(() => {
        observer.disconnect();
        window.webkit.messageHandlers.renderComplete.postMessage(
            document.documentElement.outerHTML
        );
    }, 3000);
})();
```

Receive natively via `WKScriptMessageHandler`, with a hard timeout (15-30s) as safety net using `Task` cancellation.

**Blocking unnecessary resources** via `WKContentRuleListStore` for faster rendering:

```swift
let rules = """
[
    {
        "trigger": { "url-filter": ".*", "resource-type": ["image", "media", "font", "style-sheet"] },
        "action": { "type": "block" }
    },
    {
        "trigger": { "url-filter": ".*google-analytics\\\\.com" },
        "action": { "type": "block" }
    },
    {
        "trigger": { "url-filter": ".*doubleclick\\\\.net" },
        "action": { "type": "block" }
    }
]
"""
```

Block `image`, `media`, `font`, `style-sheet` — but keep `script` enabled since SPAs need JS. Achieves **30-60% faster** rendering.

**Privacy**: Use `WKWebsiteDataStore.nonPersistent()` — no cookies, caches, localStorage, or sessionStorage persisted to disk. Create a fresh `WKWebViewConfiguration` per fetch to prevent cross-site state.

**Performance characteristics**:

| Scenario | Time | Memory |
|----------|------|--------|
| WebKit process cold start | 200-500ms | — |
| Simple static page | 0.5-1.5s | 50-100 MB |
| React SPA with API calls | 2-6s | 100-200 MB |
| Heavy Next.js app | 3-10s | 150-250 MB |
| With resource blocking | 30-60% faster | — |

**Concurrency**: Each WKWebView spawns its own WebContent process. Share a single `WKProcessPool` to reduce overhead. Practical limit: 1-2 concurrent instances given Tesseract's 20GB memory budget (LLM + TTS already co-resident). Recommendation: single reusable instance with serial queue.

> Sources: [WKZombie Renderer.swift](https://github.com/mkoehnke/WKZombie/blob/master/Sources/WKZombie/Renderer.swift), [Filip Nemecek — WKWebView headless mode](https://nemecek.be/blog/19/using-wkwebview-in-headless-mode), [Apple Forums](https://developer.apple.com/forums/thread/90732), [Embrace — WKWebView Memory](https://embrace.io/blog/wkwebview-memory-leaks/)

#### macOS 26 WebPage API (Primary Approach)

Apple introduced `WebPage` and `WebView` at WWDC 2025 ([Meet WebKit for SwiftUI](https://developer.apple.com/videos/play/wwdc2025/231/), session 231). This is the most significant new option for Tesseract.

**The key architectural insight**: `WebPage` is an `@Observable` class that represents, loads, controls, and communicates with web content — **completely separate from any visible `WebView`**. Apple explicitly states: "WebPage can be used completely on its own." This makes it the first Apple-sanctioned headless browser primitive, eliminating the need for the WKWebView visibility hack.

Available on: iOS 26, macOS 26, visionOS 26, tvOS 26, watchOS 26.

##### Observable Properties

All reactive with SwiftUI, `Observations` async sequences, and `@Bindable`:

| Property | Type | Description |
|----------|------|-------------|
| `url` | `URL?` | Current page URL, updates on navigation |
| `title` | `String?` | Page title |
| `isLoading` | `Bool` | Loading state |
| `estimatedProgress` | `Double` | 0.0 → 1.0 |
| `hasOnlySecureContent` | `Bool` | All resources loaded over HTTPS |
| `serverTrust` | `SecTrust?` | TLS trust object for the current page |
| `themeColor` | `Color?` | From `<meta name="theme-color">` |
| `customUserAgent` | `String?` | Settable custom user agent string |
| `mediaType` | `String?` | Document media type |
| `isInspectable` | `Bool` | Whether Web Inspector can attach |
| `backForwardList` | `BackForwardList` | Navigation history (Swift value types) |
| `currentNavigationEvent` | `NavigationEvent?` | Current navigation state machine |
| `fullscreenState` | Observable | Fullscreen presentation state |
| `mediaPlaybackState` | Observable | Media playback state |
| `canGoBack` | `Bool` | Whether back navigation is possible |

##### Loading Content

Three methods, all usable without a WebView:

```swift
// 1. Remote URL
let page = WebPage()
page.load(URLRequest(url: URL(string: "https://example.com")!))

// 2. HTML string (useful for post-processing extracted content)
page.load(html: "<h1>Hello</h1>", baseURL: URL(string: "about:blank")!)

// 3. Raw data (web archives, documents)
page.load(data: archiveData, mimeType: "application/x-webarchive",
          characterEncoding: .utf8, baseURL: baseURL)
```

Additional: `page.reload()`, `page.stopLoading()`, `page.load(backForwardListItem)`.

##### Navigation Events

Navigation flows through a clear state machine via the `currentNavigationEvent` observable property:

```
startedProvisionalNavigation → receivedServerRedirect (optional)
        → committed → finished
```

Error cases: `failedProvisionalNavigation`, `failed(Error)`, `pageClosed`, `webContentProcessTerminated`.

Observed via `Observations` — which Tesseract already uses in DependencyContainer and MenuBarManager:

```swift
for await event in Observations({ page.currentNavigationEvent }) {
    switch event?.kind {
    case .finished:
        let text = try await page.callJavaScript("document.body.innerText") as? String
        // process content
    case .failed(let error):
        throw error
    default: break
    }
}
```

##### JavaScript Execution

`callJavaScript` — async/await with typed arguments:

```swift
// Simple extraction
let title = try await page.callJavaScript("document.title") as? String

// With arguments — values become local JS variables (safe, no string interpolation)
let offset = try await page.callJavaScript(
    "document.getElementById(sectionId).offsetTop",
    arguments: ["sectionId": "introduction"]
) as? Double

// Complex structured extraction
let headers = try await page.callJavaScript("""
    const headers = document.querySelectorAll("h2")
    return [...headers].map(h => ({ id: h.id, title: h.textContent }))
""") as? [[String: Any]]
```

Returns `Any?` — cast to `String`, `Double`, `Bool`, `[Any]`, `[String: Any]`, etc. (anything JSON-serializable from JS). Apple recommends arguments over string interpolation for security.

##### Configuration & Privacy

```swift
var config = WebPage.Configuration()
config.websiteDataStore = .nonPersistent()  // Ephemeral — nothing persists to disk
config.applicationNameForUserAgent = "TesseractAgent/1.0"

// Custom URL schemes for serving bundled resources (e.g., injected extraction scripts)
let scheme = URLScheme("myapp")!
config.urlSchemeHandlers[scheme] = MySchemeHandler()

let page = WebPage(configuration: config)
```

`.nonPersistent()` means no cookies, caches, localStorage, or sessionStorage survive the page's lifetime — exactly what Tesseract needs for privacy-first fetching.

**URLSchemeHandler** supports async streaming — yield multiple `.data()` chunks:

```swift
struct MySchemeHandler: URLSchemeHandler {
    func reply(for request: URLRequest) -> some AsyncSequence<URLSchemeTaskResult, any Error> {
        AsyncThrowingStream { continuation in
            let response = URLResponse(url: request.url!, mimeType: "text/html",
                                       expectedContentLength: -1, textEncodingName: "utf-8")
            continuation.yield(.response(response))
            continuation.yield(.data(htmlData))
            continuation.finish()
        }
    }
}
```

##### NavigationDeciding Protocol

Controls which navigations are allowed — useful for restricting fetches to the target domain:

```swift
class FetchNavigationDecider: WebPage.NavigationDeciding {
    let allowedHost: String

    func decidePolicy(
        for action: WebPage.NavigationAction,
        preferences: inout WebPage.NavigationPreferences
    ) async -> WKNavigationActionPolicy {
        guard let host = action.request.url?.host, host == allowedHost else {
            return .cancel  // Block cross-domain navigations
        }
        return .allow
    }
}

page.navigationDecider = FetchNavigationDecider(allowedHost: "example.com")
```

`NavigationPreferences` includes: `allowsContentJavaScript`, `preferredHTTPSNavigationPolicy`, `preferredContentMode`.

##### Headless Fetch Pattern for Tesseract

```swift
/// Fetch a URL, render JS, extract content — no UI, no window, no view hierarchy hack
func fetchRenderedContent(from url: URL, timeout: Duration = .seconds(20)) async throws -> String? {
    var config = WebPage.Configuration()
    config.websiteDataStore = .nonPersistent()
    config.applicationNameForUserAgent = "TesseractAgent/1.0"

    let page = WebPage(configuration: config)
    page.load(URLRequest(url: url))

    return try await withThrowingTaskGroup(of: String?.self) { group in
        // Navigation observation
        group.addTask {
            for await event in Observations({ page.currentNavigationEvent }) {
                if case .finished = event?.kind {
                    // Wait a beat for post-load JS rendering (SPA hydration)
                    try await Task.sleep(for: .seconds(1.5))
                    return try await page.callJavaScript(
                        "document.documentElement.outerHTML"
                    ) as? String
                }
                if case .failed(let error) = event?.kind {
                    throw error
                }
            }
            return nil
        }

        // Hard timeout
        group.addTask {
            try await Task.sleep(for: timeout)
            throw FetchError.timeout
        }

        let result = try await group.next()!
        group.cancelAll()
        return result
    }
}
```

For more reliable SPA detection, inject a MutationObserver via `callJavaScript` after `.committed` to detect when rendering stabilizes, rather than a fixed 1.5s sleep.

##### WebPage vs WKWebView

| Aspect | WKWebView | WebPage |
|--------|-----------|---------|
| **Headless** | Hacks required (alpha 0.01 trick) | **Designed for standalone use** |
| **State model** | KVO + 8+ delegate methods | **`@Observable` + `Observations`** |
| **JS execution** | `evaluateJavaScript` + completion handler | **`callJavaScript` async/await + typed args** |
| **Navigation** | `WKNavigationDelegate` protocol | **`NavigationDeciding` + observable event** |
| **Swift concurrency** | Retrofitted | **Native from the ground up** |
| **Memory** | ~50-150 MB per WebContent process | Same |
| **Process model** | Multi-process (WebContent + Network + Storage) | Same |
| **Privacy** | `WKWebsiteDataStore.nonPersistent()` | Same via `config.websiteDataStore` |
| **Min target** | macOS 10.10 | **macOS 26** |

##### Process Model

WebPage uses the same multi-process WebKit architecture as Safari and WKWebView:
- **WebContent process** — JS execution and DOM, in a separate process (security sandbox)
- **Network process** — shared across all WebPage/WKWebView instances (connection pooling)
- **Storage process** — cookies, databases, service workers

Each `WebPage` spawns a WebContent process (~50-150 MB). For Tesseract's 20 GB memory budget with LLM + TTS co-resident, use a **single reusable `WebPage` instance** with a serial queue rather than creating one per fetch.

##### Why WebPage is the Right Choice for Tesseract

1. **Tesseract already targets macOS 26** — no deployment floor issue
2. **Already uses `Observations` throughout** — DependencyContainer, AppDelegate, MenuBarManager
3. **Eliminates the WKWebView visibility hack entirely** — no window attachment, no alpha tricks
4. **Native Swift concurrency** — async/await JS execution, `Observations` for navigation
5. **Handles SPAs natively** — full JS execution renders React/Next.js/Vue content
6. **Privacy-first** — `.nonPersistent()` data store, `NavigationDeciding` for domain restriction
7. **The `com.apple.security.network.client` entitlement** is already in both Debug and Release entitlements

##### Known Beta-Period Issues (Summer 2025)

- Some WWDC sample code did not compile in betas 6-7 (navigation observation issues)
- History menus from `backForwardList` did not refresh UI when list changed
- Console log spam on macOS when `NavigationDeciding` cancels requests (looks like crash logs, is not)
- External links opening in new tabs/windows failed to load

These are expected to be resolved in the macOS 26 release.

> Sources: [WWDC25 Session 231 — Meet WebKit for SwiftUI](https://developer.apple.com/videos/play/wwdc2025/231/), [Apple — WebKit for SwiftUI Documentation](https://developer.apple.com/documentation/webkit/webkit-for-swiftui), [Daniel Saidi — WebView in SwiftUI](https://danielsaidi.com/blog/2025/06/10/webview-is-finally-coming-to-swiftui), [TrozWare — SwiftUI WebView](https://troz.net/post/2025/swiftui-webview/), [AppCoda — WebView and WebPage in SwiftUI](https://www.appcoda.com/swiftui-webview/), [WebKit Blog — News from WWDC25](https://webkit.org/blog/16993/news-from-wwdc25-web-technology-coming-this-fall-in-safari-26-beta/), [Nathan Borror on WebPage as headless browser](https://x.com/nathanborror/status/2029023900294889664), [WebKit commit — Observable properties](https://www.mail-archive.com/webkit-changes@lists.webkit.org/msg222853.html)

#### Ruled Out Alternatives

| Approach | Why Not |
|----------|---------|
| **Playwright/Chromium** | 400MB-1GB browser binaries, Node.js runtime, incompatible with App Store sandbox |
| **Safari WebDriver** | No headless mode, requires user to manually enable "Allow Remote Automation" |
| **JavaScriptCore alone** | No DOM, no CSS, no rendering — cannot process pages that require JS rendering |

---

### 11.2 Content Extraction: Heuristic vs. Model-Based

#### Two-Tier Fetch Architecture (Recommended)

Follow the pattern proven by Aider and Crawl4AI — fast HTTP fetch first, browser fallback only when needed:

```
                       web_fetch(url)
                            |
                  [URLSession.ephemeral GET]
                            |
                   [swift-readability extraction]
                            |
                 Content > 200 chars?
                     /            \
                   Yes              No (likely SPA)
                    |                |
          [Demark → markdown]   [WebPage headless render]
                    |                |
            [Truncate, return]  [Observations → .finished]
                                     |
                              [callJavaScript → outerHTML]
                                     |
                             [swift-readability]
                                     |
                             [Demark → markdown]
                                     |
                              [Truncate, return]
```

**SPA detection heuristic** (when to trigger WebPage fallback):
- Extracted text < 200 chars from HTML > 10KB (strong signal)
- HTML contains `<div id="root"></div>`, `<div id="app"></div>`, or `<div id="__next"></div>` with minimal text
- Presence of `<noscript>` with substantial content alongside empty body
- Large inline `<script>` blocks but minimal visible text

#### swift-readability — Pure Swift Readability Port (Key Discovery)

[`lake-of-fire/swift-readability`](https://github.com/lake-of-fire/swift-readability) — BSD-3-Clause, Swift 6.2, macOS 13+. A full port of Mozilla Readability.js v0.6.0 using SwiftSoup. Pure Swift, no JS, no WKWebView.

```swift
let reader = Readability(html: htmlString, url: URL(string: "https://example.com")!)
let result = try reader.parse()
// result.title, result.content (clean HTML), result.textContent (plain text),
// result.excerpt, result.byline, result.siteName, result.publishedTime

// Pre-check without full parse:
let isReadable = Readability.isProbablyReaderable(html: htmlString)
```

The Readability algorithm scores DOM nodes by text density, link density, and class/ID heuristics. Core scoring:
- **Positive classes/IDs**: "article", "body", "content", "entry", "main", "page", "post", "text"
- **Negative classes/IDs**: "ad", "footer", "masthead", "sidebar", "banner", "comment", "share", "login"
- **Link density**: `linkTextLength / totalTextLength` — values > 0.5 indicate navigation
- **Tag weights**: DIV +5, PRE/BLOCKQUOTE +3, FORM/UL -3, H1-H6 -5
- **Retry**: 3 extraction passes with progressively relaxed filtering

The maintainer also maintains SwiftSoup (v2.13.3, March 2026, actively optimized). Test suite includes Mozilla's full fixture corpus.

#### Demark — HTML-to-Markdown via JavaScriptCore (Recommended Converter)

[`steipete/Demark`](https://github.com/steipete/Demark) — MIT, macOS 14+. Two engines:

| Feature | Turndown.js engine | html-to-md engine (recommended) |
|---------|-------------------|-------------------------------|
| Runtime | WKWebView | **JavaScriptCore** |
| First conversion | ~100ms | **~5-10ms** |
| Subsequent | ~10-50ms | **~5-10ms** |
| Memory | ~20MB | **~5MB** |
| Threading | Main thread only | **Any thread (serial queue)** |

The **html-to-md engine** runs via JavaScriptCore, needs no WKWebView, can run off the main thread, and handles GFM tables, fenced code blocks with language detection, and all standard markdown elements. This is the right fit for an agent pipeline.

#### Other Swift Libraries Evaluated

| Library | Status | Verdict |
|---------|--------|---------|
| **SwiftSoup** (v2.13.3) | Active, 5K+ stars, pure Swift | Foundation layer — used by swift-readability |
| **SwiftHTMLToMarkdown** | Dormant since Oct 2023, 54 stars | **No table support**, too limited |
| **HTMLToMarkdown** (jaywcjlove) | Early stage, JSC-based, 18 stars | Not production-ready yet |
| **swift-readability** (Ryu0118) | WKWebView wrapper, 52 stars | Different lib — wraps JS Readability, less suitable for headless |

#### JavaScriptCore as an Alternative Runtime

Turndown.js (~30KB minified) and Readability.js (~70KB + JSDOMParser ~25KB) can run directly in `JSContext` without WKWebView:

- `JSContext` can run off the main thread on a dedicated `JSVirtualMachine`
- Turndown.js works without a full DOM (it parses HTML internally)
- Readability.js requires JSDOMParser.js (bundled by Mozilla) for non-browser use
- Bundle size impact: trivial (~125KB total)

However, since `swift-readability` already ports Readability to pure Swift, JSC is only needed for markdown conversion (Demark already handles this).

---

### 11.3 Local Models for Content Extraction

#### MinerU-HTML / Dripper (Major Discovery — Best-in-Class)

Published Nov 2025, updated March 2026. **A 0.5B model achieving near-parity with GPT-5 on web content extraction.**

**Core innovation**: Instead of generating markdown token-by-token, it treats extraction as **binary sequence classification** — each HTML block is labeled "main" or "other". A deterministic finite state machine constrains decoding, **eliminating hallucination entirely**.

**Pipeline**:
1. Strip non-content tags (`<script>`, `<style>`, `<header>`, `<aside>`)
2. Prune attributes (keep only `class` and `id`)
3. Chunk at block-level rendering breakpoints
4. Truncate long blocks (200-char limit per block)
5. Result: input reduced to **12.83% of raw HTML** (median 9.72%)
6. Model classifies each block as main/other
7. Markdown reconstructed from "main" blocks of the original HTML

**WebMainBench results** (7,887 diverse pages, ROUGE-N F1):

| Method | F1 Score | Type | Size |
|--------|----------|------|------|
| DeepSeek-V3 | 0.9098 | Proprietary LLM | — |
| GPT-5 | 0.9024 | Proprietary LLM | — |
| **MinerU-HTML v1.1** | **0.9001** | **Local model** | **0.5B** |
| Dripper+fallback | 0.8399 | Local model | 0.6B |
| Magic-HTML | 0.7138 | Heuristic | — |
| Readability | 0.6542 | Heuristic | — |
| Trafilatura | 0.6402 | Heuristic | — |
| html2text | 0.6042 | Baseline | — |
| **ReaderLM-v2** | **0.2279** | **Generative model** | **1.5B** |

**Key takeaway**: MinerU-HTML outperforms Readability by **~38 percentage points** on diverse pages (tables, forums, complex layouts) while being a tiny 0.5B model. ReaderLM-v2 performs catastrophically on independent benchmarks despite Jina's own favorable results.

**Content-type performance** (Dripper):

| Content Type | Dripper | Best Heuristic |
|-------------|---------|----------------|
| Tables | 0.7693 | 0.6681 |
| Code | 0.8368 | 0.8471 |
| Equations | 0.8889 | 0.8470 |
| Forums/conversational | 0.7671 | 0.5766 |

Particularly strong on tables and forums — exactly where heuristics struggle.

**Practical specs for Tesseract**:
- Model: Qwen3-0.6B (Dripper) or Hunyuan-0.5B-compact (MinerU-HTML v1.1)
- Memory: **~300-400 MB** at 4-bit quantization — fits within 20GB budget alongside LLM + TTS
- Speed: Output tokens are minimal (classification labels, not full markdown) — perhaps 50-200 tokens per page
- License: **Apache 2.0** (commercial use OK)
- MLX: No published MLX conversion yet — would need `mlx-lm.convert` (straightforward for Qwen3-0.6B)
- Context: 32K window (sufficient for 98.7% of pages after HTML simplification; falls back to heuristic for remaining 1.3%)

> Sources: [Dripper Paper (arXiv:2511.23119)](https://arxiv.org/html/2511.23119v1), [MinerU-HTML GitHub](https://github.com/opendatalab/MinerU-HTML)

#### ReaderLM-v2 — Not Recommended

Despite Jina's marketing, ReaderLM-v2 has critical problems:

| Issue | Detail |
|-------|--------|
| **Independent benchmark** | 0.2279 F1 on WebMainBench — worst of all methods tested |
| **Jina's own admission** | "the Jina Reader API [heuristic] still remains the best option" |
| **License** | CC-BY-NC-4.0 — **blocks commercial use** |
| **Speed** | Generative (token-by-token markdown). ~30 seconds per page at 100 tok/s for 3K output tokens |
| **Size** | 1.5B params — 3x larger than MinerU-HTML for dramatically worse results |

The discrepancy between Jina's benchmarks (0.84 ROUGE-L) and independent benchmarks (0.23 F1) suggests severe overfitting to their evaluation set.

#### Using a Small Qwen Model as a "Reader" (Claude Code Pattern)

Claude Code uses Haiku 3.5 as a secondary LLM to summarize fetched content before returning it to the main agent. The prompt template:

```
Web page content:
---
{content}
---
{user_query}
Provide a concise response based only on the content above.
- Enforce a strict 125-character maximum for quotes from any source document.
- Use quotation marks for exact language; language outside quotation marks
  should never be word-for-word the same.
```

**For Tesseract**: The main agent LLM could serve this role itself (no need for a separate "reader" model) if the heuristic extraction (swift-readability) produces clean enough output. A dedicated small model only makes sense if extraction quality on diverse pages becomes a problem — in which case, MinerU-HTML's 0.5B classifier is the right choice over a general-purpose Qwen model, since it eliminates hallucination via constrained decoding.

---

### 11.4 How Production Agents Handle Web Fetch

#### Claude Code WebFetch Pipeline

6-stage pipeline, all running locally on the user's machine:

1. **URL validation**: Max 2,000 chars, HTTP→HTTPS upgrade, credentials stripped
2. **Domain safety**: Backend call to `claude.ai/api/web/domain_info?domain=${hostname}`
3. **HTTP fetch**: Axios with `Accept: text/markdown, text/html, */*`. `maxRedirects: 0` (cross-host redirects require new tool call). ~10MB max
4. **HTML→Markdown**: Turndown (default config). Known bug: doesn't strip `<style>`/`<script>` inner text
5. **Truncation**: 100,000 characters AFTER markdown conversion
6. **Haiku summarization**: Content sent to Claude 3.5 Haiku. ~80 trusted domains (docs.python.org, developer.apple.com, react.dev, etc.) bypass Haiku when returning markdown under 100K chars

**Cache**: 15-min TTL, 50MB max LRU. **No JS rendering** — plain HTTP GET only.

#### Crawl4AI Content Filtering (PruningContentFilter)

The most detailed open-source content quality scoring, no LLM needed:

| Metric | Weight | Calculation |
|--------|--------|-------------|
| Text density | 0.4 | `text_len / tag_len` |
| Link density | 0.2 | `1 - (link_text_len / text_len)` |
| Tag weight | 0.2 | `article: 1.5`, `main: 1.4`, `p: 1.0`, `div: 0.5`, `span: 0.3` |
| Class/ID pattern | 0.1 | -0.5 for `nav\|footer\|sidebar\|ads\|comment` |
| Text length | 0.1 | `log(text_len + 1)` |

Default prune threshold: 0.48. Blocks scoring below are stripped. This is implementable in Swift with SwiftSoup.

#### Firecrawl Multi-Engine Waterfall

12 engine types tried in priority order, cascading on failure. HTML cleaning strips 50+ CSS selectors for boilerplate. Markdown conversion: Go shared library → Turndown fallback. Uses a Rust WASM module for fast HTML transformation.

---

### 11.5 Token Budget Analysis

#### Typical Page Token Counts

| Stage | Tokens | Notes |
|-------|--------|-------|
| Raw HTML (median) | 3,200 | Average is 10,400; complex pages 30K-50K+ |
| Raw HTML (news article) | 15,000-25,000 | Nav, ads, sidebar, footer |
| After Readability extraction | 1,500-3,000 | **80-90% reduction** |
| After Dripper/MinerU-HTML | 1,500-3,000 | **87-95% reduction**, better accuracy on complex pages |
| After markdown conversion | ~70% of HTML tokens | Additional 20-30% reduction vs clean HTML |

#### Sweet Spot for Tesseract's 120K Context

With 120K window, 16K reserve, 20K recent tokens kept (per ContextManager):
- **Available for web content**: ~10-20K tokens per tool call
- Readability output (1,500-3,000 tokens) fits easily — room for **4-8 pages** in context simultaneously
- Raw HTML would consume 1-3 pages worth of context for a single page — unacceptable
- **50KB text limit** (~12K tokens) is the right truncation ceiling per fetch

---

### 11.6 Recommended Implementation for Tesseract

#### Phase 1 — MVP Pipeline (~25-90ms per page)

```
URLSession.ephemeral GET
        ↓
swift-readability (content extraction, pure Swift)
        ↓
Demark html-to-md engine (JSC, ~5ms, off main thread)
        ↓
Token-aware truncation (~50KB / ~12K tokens)
        ↓
Return to agent
```

**Dependencies**: SwiftSoup (SPM), swift-readability (SPM), Demark (SPM). All pure Swift / JSC, no WKWebView in the fast path.

**When swift-readability extraction returns < 200 chars from 10KB+ HTML** (SPA fallback):

```
WebPage (nonPersistent, headless)
        ↓
page.load(URLRequest(url: url))
        ↓
Observations({ page.currentNavigationEvent }) → .finished
        ↓
callJavaScript("document.documentElement.outerHTML")
        ↓
swift-readability + Demark
        ↓
Truncate, return
```

No visibility hack needed — WebPage works standalone. Single reusable instance with serial queue for memory efficiency.

#### Phase 2 — Enhanced Extraction

Add MinerU-HTML / Dripper-style 0.5B classifier on MLX:
- Convert Qwen3-0.6B or Hunyuan-0.5B-compact weights to MLX format
- Implement HTML simplification pipeline in Swift (strip tags, prune attributes, chunk at block boundaries)
- Binary classification: "main" vs "other" for each block
- Reconstruct markdown from "main" blocks
- ~300-400 MB additional memory, minimal inference time

#### Content Cleaning Best Practices

**Strip before extraction**: `<script>`, `<style>`, `<noscript>`, `<svg>`, `<canvas>`, tracking pixels, cookie banners, social widgets, comment sections, inline styles/data attributes.

**Preserve**: Semantic headings (h1-h6), paragraphs, lists, tables, code blocks, blockquotes, image alt text, link anchor text.

**Truncation strategy**: Cut at section boundaries (headings), not mid-paragraph. Always keep title + first paragraph. Append `[Content truncated at {N} characters]` marker.

---

### 11.7 Sources (Section 11)

#### WebPage API & macOS Web APIs
- [WWDC25 Session 231 — Meet WebKit for SwiftUI](https://developer.apple.com/videos/play/wwdc2025/231/)
- [Apple — WebKit for SwiftUI Documentation](https://developer.apple.com/documentation/webkit/webkit-for-swiftui)
- [Daniel Saidi — WebView in SwiftUI](https://danielsaidi.com/blog/2025/06/10/webview-is-finally-coming-to-swiftui)
- [TrozWare — SwiftUI WebView](https://troz.net/post/2025/swiftui-webview/)
- [AppCoda — WebView and WebPage in SwiftUI](https://www.appcoda.com/swiftui-webview/)
- [SwiftUI Snippets — Introducing WebKit](https://swiftuisnippets.wordpress.com/2025/08/20/introducing-webkit-native-swiftui-web-views-in-wwdc25/)
- [Create with Swift — Displaying web content in SwiftUI](https://www.createwithswift.com/displaying-web-content-in-swiftui/)
- [Nathan Borror on WebPage as headless browser](https://x.com/nathanborror/status/2029023900294889664)
- [WebKit Blog — News from WWDC25](https://webkit.org/blog/16993/news-from-wwdc25-web-technology-coming-this-fall-in-safari-26-beta/)
- [WebKit commit — Observable properties for WebPage](https://www.mail-archive.com/webkit-changes@lists.webkit.org/msg222853.html)
- [WKZombie — Headless Browser Framework (legacy WKWebView approach)](https://github.com/mkoehnke/WKZombie)
- [Filip Nemecek — WKWebView headless mode (legacy)](https://nemecek.be/blog/19/using-wkwebview-in-headless-mode)
- [Embrace — WKWebView Memory](https://embrace.io/blog/wkwebview-memory-leaks/)
- [Apple — WKContentRuleListStore](https://developer.apple.com/documentation/webkit/wkcontentruleliststore)
- [Playwright Browser Footprint](https://datawookie.dev/blog/2025-06-06-playwright-browser-footprint/)

#### Content Extraction Models
- [Dripper Paper (arXiv:2511.23119)](https://arxiv.org/html/2511.23119v1)
- [MinerU-HTML GitHub](https://github.com/opendatalab/MinerU-HTML)
- [ReaderLM-v2 HuggingFace](https://huggingface.co/jinaai/ReaderLM-v2)
- [ReaderLM-v2 Blog Post](https://jina.ai/news/readerlm-v2-frontier-small-language-model-for-html-to-markdown-and-json/)
- [MLX Community ReaderLM-v2](https://huggingface.co/mlx-community/jinaai-ReaderLM-v2)

#### Swift Libraries
- [swift-readability (lake-of-fire)](https://github.com/lake-of-fire/swift-readability) — Pure Swift Readability port
- [SwiftSoup](https://github.com/scinfu/SwiftSoup) — HTML parsing
- [Demark](https://github.com/steipete/Demark) — HTML→Markdown via JSC
- [SwiftHTMLToMarkdown](https://github.com/ActuallyTaylor/SwiftHTMLToMarkdown) — Pure Swift (dormant)
- [javascript-core-extras](https://github.com/mhayes853/javascript-core-extras) — JSC concurrency helpers

#### Production Agent Pipelines
- [Inside Claude Code's Web Tools](https://mikhail.io/2025/10/claude-code-web-tools/)
- [Crawl4AI GitHub](https://github.com/unclecode/crawl4ai)
- [Firecrawl GitHub](https://github.com/mendableai/firecrawl)
- [Browser-use GitHub](https://github.com/browser-use/browser-use)
- [Jina Reader API](https://jina.ai/reader/)
- [Readability.js for RAG](https://philna.sh/blog/2025/01/09/html-content-retrieval-augmented-generation-readability-js/)
- [Readability Algorithm Deep Dive (DeepWiki)](https://deepwiki.com/mozilla/readability)

#### Benchmarks
- [WebMainBench — Web Content Extraction Benchmark](https://chuniversiteit.nl/papers/comparison-of-web-content-extraction-algorithms)
- [Markdown vs HTML for LLM Performance](https://markdownconverters.com/blog/markdown-vs-html-ai-performance)
- [Cloudflare — Markdown for Agents](https://blog.cloudflare.com/markdown-for-agents/)

---

## 12. Sources & References

### Agent Implementations
- [Inside Claude Code's Web Tools: WebFetch vs WebSearch](https://mikhail.io/2025/10/claude-code-web-tools/)
- [The Claude Code WebSearch Black Box](https://www.vmunix.com/posts/claude-websearch-blackbox)
- [Anthropic Web Search API](https://claude.com/blog/web-search-api)
- [OpenClaw Web Tools](https://docs.openclaw.ai/tools/web) | [GitHub](https://github.com/openclaw/openclaw)
- [OpenCode GitHub](https://github.com/opencode-ai/opencode)
- [Aider scrape.py](https://github.com/paul-gauthier/aider/blob/main/aider/scrape.py)
- [Continue.dev Context Providers](https://docs.continue.dev/customization/context-providers)
- [Cursor @Web docs](https://docs.cursor.com/context/@-symbols/@-web)
- [GPT4All Web Search Beta](https://github.com/nomic-ai/gpt4all/wiki/Web-Search-Beta-Release)
- [Ollama Web Search](https://docs.ollama.com/capabilities/web-search)
- [AnythingLLM Agent Usage](https://docs.useanything.com/agent/usage)
- [LocalAI Agents](https://localai.io/features/agents/)
- [claw-search (SearXNG plugin for OpenClaw)](https://github.com/binglius/claw-search)

### Search APIs & Services
- [Brave Search API Pricing](https://api-dashboard.search.brave.com/documentation/pricing)
- [Tavily Pricing](https://www.tavily.com/pricing)
- [Serper.dev](https://serper.dev/)
- [Exa.ai Pricing](https://exa.ai/pricing)
- [Google Custom Search API](https://developers.google.com/custom-search/v1/)
- [Bing Search API Retirement](https://ppc.land/microsoft-ends-bing-search-apis-on-august-11-alternative-costs-40-483-more/)
- [Jina AI Reader & Search](https://jina.ai/reader/)
- [SearXNG](https://github.com/searxng/searxng) | [API docs](https://docs.searxng.org/dev/search_api.html)
- [DuckDuckGo Search (Python)](https://pypi.org/project/duckduckgo-search/)

### Content Extraction & Libraries
- [Mozilla Readability](https://github.com/mozilla/readability)
- [Web Content Extraction Benchmark](https://chuniversiteit.nl/papers/comparison-of-web-content-extraction-algorithms)
- [SwiftSoup](https://github.com/scinfu/SwiftSoup)
- [swift-readability — Pure Swift Readability port](https://github.com/lake-of-fire/swift-readability)
- [Demark (HTML→Markdown in Swift)](https://steipete.me/posts/2025/introducing-demark-html-to-markdown-in-swift)
- [SwiftHTMLToMarkdown](https://github.com/ActuallyTaylor/SwiftHTMLToMarkdown)
- [MinerU-HTML](https://github.com/opendatalab/MinerU-HTML) | [Dripper Paper](https://arxiv.org/html/2511.23119v1)
- [ReaderLM-v2](https://huggingface.co/jinaai/ReaderLM-v2) (CC-BY-NC, poor independent benchmarks — see Section 11.3)
- [Crawl4AI](https://github.com/unclecode/crawl4ai)
- [Firecrawl](https://www.firecrawl.dev/)

### MCP Servers
- [Brave Search MCP Server](https://github.com/brave/brave-search-mcp-server)
- [MCP Reference Servers](https://github.com/modelcontextprotocol/servers)
- [MCP 2026 Roadmap](http://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/)

### Privacy & Architecture
- [User Consent Best Practices for AI Agents](https://curity.io/blog/user-consent-best-practices-in-the-age-of-ai-agents/)
- [Building a Private AI Search Agent](https://zsiegel.com/building-my-own-private-ai-search-agent-that-actually-works/)
- [How to Enable Internet Access for Local LLMs](https://localllm.in/blog/how-to-local-llm-internet-access)
- [70+ AI Search Stats for 2026](https://seranking.com/blog/ai-statistics/)
- [browser-use GitHub](https://github.com/browser-use/browser-use)
