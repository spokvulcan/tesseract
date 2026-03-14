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

## 3. Comparison Summary

| Agent | Search Provider | Fetch Method | JS Rendering | API Key | Offline LLM |
|-------|----------------|-------------|-------------|---------|-------------|
| **Claude Code** | Brave (via Anthropic) | Axios + Turndown + Haiku | No | Anthropic sub | No |
| **OpenClaw** | Brave/Gemini/Grok/Kimi/Perplexity | Readability + Firecrawl | No (fetch); Yes (Firecrawl) | Per provider | No |
| **OpenCode** | Exa.ai | Bun fetch | No | No | No |
| **Cursor** | Exa.ai | Built-in browser | Yes | No (internal) | No |
| **Aider** | None (URL only) | httpx + BeautifulSoup + Playwright | Optional | No | Yes |
| **Continue.dev** | Proxy / Serper | URL provider | No | Serper key | No |
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

### Variant G: "Hybrid — Simple Fetch + WKWebView Fallback"

**Search**: Brave Search API (or any from Variant D).

**Fetch (Tier 1)**: URLSession + SwiftSoup + markdown conversion — fast, handles ~85% of pages.

**Fetch (Tier 2)**: If Tier 1 returns suspiciously little content (< 500 chars after extraction), retry with WKWebView rendering — handles JS-heavy pages.

| Pros | Cons |
|------|------|
| Best coverage — handles both static and dynamic pages | Two code paths to maintain |
| Fast path for most pages, fallback for edge cases | Heuristic for "suspiciously little content" may misfire |
| Follows Aider's two-tier pattern (httpx → Playwright) | WKWebView adds complexity |
| No external runtime dependencies | |

**Effort**: Medium-high. But amortizable — Tier 1 first, Tier 2 later.

---

## 8. Recommended Approach for Tesseract

### Phase 1 (MVP): Variant A + elements of D

1. **Two new built-in tools**: `web_search` and `web_fetch` in `Features/Agent/Tools/BuiltIn/`
2. **Search**: Brave Search API as primary (privacy-aligned, independent index, official MCP server exists). Store API key in Keychain via Settings. DuckDuckGo as zero-config fallback (no key needed)
3. **Fetch**: `URLSession` (ephemeral) → SwiftSoup (parse + clean) → SwiftHTMLToMarkdown (convert) → token-aware truncation (~50KB / ~12K tokens)
4. **Settings**: "Web Access" toggle (off by default), search provider picker, API key field
5. **UI**: Show search queries in the agent conversation (transparency). Badge/indicator when web tools are active
6. **Defaults**: 5 results, snippets only. Fetch limited to 50KB extracted text

### Phase 2 (Enhancement):

7. Add Tavily and/or Serper as alternative search providers
8. WKWebView fallback for JS-heavy pages (Variant G's Tier 2)
9. SearXNG support for power users who self-host

### Phase 3 (Future):

10. Jina Reader as an optional extraction enhancement
11. ReaderLM-v2 on MLX for fully offline HTML-to-markdown (1.5B model, feasible on Apple Silicon)
12. Full browser-use tool via WKWebView for interactive tasks

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

## 10. Sources & References

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
- [Demark (HTML→Markdown in Swift)](https://steipete.me/posts/2025/introducing-demark-html-to-markdown-in-swift)
- [SwiftHTMLToMarkdown](https://github.com/ActuallyTaylor/SwiftHTMLToMarkdown)
- [ReaderLM-v2](https://huggingface.co/jinaai/ReaderLM-v2)
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
