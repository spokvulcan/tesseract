# The Browser MCP is the sole web surface; search renders in WebKit

Status: accepted

The agent's web access had two generations wired in parallel: standalone tools
(`web_search`, `web_fetch` via `WebToolsExtension`) and the Browser MCP
(ADR-0026/0027, `browser.*`). This duplicated capability — `web_search` and
`browser.search` both called the same `DuckDuckGoClient`; `web_fetch` and
`browser.fetch` both called the same `EphemeralPageReader` — and contradicted the
glossary, which already declares the **Agent Browser** "the single web-access path
for every Tesseract web capability." Worse, `DuckDuckGoClient` scrapes
`html.duckduckgo.com/html/` with a bot User-Agent (`TesseractAgent/1.0 (macOS)`)
over `URLSession`, and DuckDuckGo now serves 202/403 challenges to non-browser
clients — so *both* search paths were broken, and the agent would retry the
failing tool repeatedly before falling back to driving the browser.

**Decision:**

- **The Browser MCP server is the sole web-access surface.** Remove
  `WebToolsExtension` (`web_search`, `web_fetch`) and `DuckDuckGoClient`. Search,
  fetch, and interactive browsing all live under `browser.*`. This is required,
  not merely tidy: the server is consumed by external coding agents (OpenCode et
  al.) that have no web tools of their own (ADR-0027), so it must be
  self-contained.
- **Search renders in a real WebKit Ephemeral Page — no HTTP scrape.**
  `browser.search` navigates a cookieless **Ephemeral Page** (real browser UA, JS,
  cookies — not a fingerprintable bot client) to a configurable engine
  (**DuckDuckGo default**) and extracts structured `{title, url, snippet}` via a
  SERP DOM query, falling back to readable page text on zero results. This aligns
  code with the glossary (the Ephemeral Page backs *search as well as fetch*) and
  moves search off the exact scrape endpoint DDG blocks.
- **Fetch is `browser.fetch`, unchanged** — the Ephemeral Page → Readability →
  markdown path (ADR-0026). A cheaper two-stage fetch was considered and deferred
  (see Consequences).
- **Two enablement switches, both default-on** — refining ADR-0027's single
  "Browser Access" switch. *Web Access* gates whether the server's tools reach the
  in-app agent over the in-process transport (no port). *HTTP exposure* gates the
  loopback `/mcp` listener that admits outside clients. They are separable because
  `InProcessMCPTransport` calls `MCPBrowserServer.handle(request:)` directly and
  needs no listener; keeping them distinct lets the in-app agent browse without
  opening a port, though both default on.
- **Web strategy is layered across three altitudes.** A short web-orientation
  block is injected into the system prompt *only when web tools are present* (the
  search → fetch → browse ladder + cite sources), so a small model knows the
  ordering up front instead of flailing on a failing tool. The `web-research`
  skill carries the full discipline (broad-then-narrow queries, never cite
  snippets, escalate to `browser.navigate`/`read_page`/`page_map`/`click`/`type`
  for gated or interactive sources, freshness, citations). Per-tool descriptions
  carry per-tool specifics.

**Considered options:**

- *Keep `web_search`/`web_fetch` as a top-level "cheap layer" (Model 1)* —
  rejected: the MCP server must be self-contained for external consumers, and the
  two "cheap" tools were not actually cheaper (both already rendered WebKit), so
  they were pure duplication.
- *Local SearXNG for search* — keyless and self-hostable, but adds a bundled
  service to ship and run; deferred.
- *Fix the DDG scraper (real UA, `vqd` token, cookies)* — perpetual cat-and-mouse;
  the exact failure that just occurred.
- *Remove search entirely, make the agent drive the browser to Google* —
  token-heavy navigate→read→click loop and the source of the flailing we are
  trying to fix; kept only as the natural fallback.

**Consequences:**

- Every search and fetch renders in WebKit; there is no non-browser fast path.
  Accepted. A two-stage adaptive fetch (cheap `URLSession` GET + Readability,
  escalate to WebKit on thin/JS-gated results) is a **deferred follow-up** — it
  reverses ADR-0026 and adds a second failure surface (SPAs 403 a non-browser
  GET), so it is out of scope here.
- Deleting `web_fetch` drops the 15-minute `WebFetchCache`, which was wired only
  into that tool; `browser.fetch` currently renders every call uncached.
  Preserving the cache by moving it into the `browser.fetch` path is a
  **low-priority follow-up**, not a gate.
- **Unverified:** that DuckDuckGo will serve a cookieless *real-WebKit* Ephemeral
  Page without a challenge is not established from code — it must be confirmed live
  against `EphemeralPageReader`. The configurable engine + text fallback exist
  precisely because this is unverified.
