# ADR-0031: Local-only tool telemetry for the Browser MCP Server

- Status: Accepted
- Date: 2026-07-09
- Relates to: ADR-0026 (Agent Browser), ADR-0027 (Browser MCP server + dogfooded client), ADR-0028 (sole web surface)

## Context

The Browser MCP tools are consumed two ways — by Tesseract's own agent
over the in-process transport, and by external coding clients (OpenCode,
Claude Code) over the loopback `/mcp` listener. We suspect the tools can
be tuned for better results with fewer calls (one concrete suspicion:
screenshots are capped at 1200 CSS px wide and may be too low-resolution
for models to read), but we had **zero recorded evidence**: the server
emitted no logs at all, and unified logging retains only days and is not
analyzable as a dataset.

Industry practice points one way. MCP's own `logging` capability is
deprecated (SEP-2577) in favor of OpenTelemetry-shaped observability;
OTel's GenAI conventions define an `execute_tool` span (tool name, call
id, arguments/result as opt-in attributes, error type); Playwright MCP's
`--save-trace` records per-session action traces to disk; and Anthropic's
tool-improvement guidance is to collect real agent transcripts and
analyze tool-call frequency, latency, errors, and retries offline. The
OTLP file-exporter spec blesses local JSON Lines as a first-class
exporter target.

Tesseract's privacy stance ("nothing ever leaves the Mac — no cloud, no
telemetry") makes cloud observability a non-starter, but *local* capture
is the exact mechanism that makes the stance compatible with
evidence-driven tool improvement.

## Decision

Record **local-only, durable, analyzable telemetry** for every Browser
MCP interaction, at the server's protocol choke point, so both entry
paths are covered by one recorder:

- **Where it hooks**: `MCPBrowserServer.handle(request:origin:)` — the
  single seam both the in-process transport and the HTTP listener
  converge on. The new `origin` parameter (`in_process` | `http`) names
  the entry path; `initialize`'s `clientInfo` supplies the client
  identity (`tesseract-agent`, `opencode`, …) that every later event in
  the session inherits.
- **What it records**: `session_start`, `tools_list`, one `tool_call`
  event per `tools/call` (tool name, verbatim arguments with long string
  values capped, latency, `ok`/`error` outcome, error message, result
  text size + capped preview, per-image pixel dimensions and byte
  counts), `protocol_error` (parse errors, unknown methods, missing
  sessions, bad params), `session_end`, and `server_shutdown`.
- **Schema**: a flat versioned event record whose field semantics follow
  the OTel GenAI `execute_tool` conventions (tool name, arguments,
  error, session id) so the corpus stays convertible to OTLP, encoded in
  the repo's own discriminated-line JSONL with a schema header line
  (`CompletionTraceLine` precedent) rather than the verbose OTLP
  envelope. The OTel conventions are still Development-status, so the
  header's `schemaVersion` is the compatibility gate.
- **Where it lives**:
  `Application Support/BrowserMCPTelemetry/mcp-<yyyy-MM-dd>.jsonl` via
  `RotatingJSONLWriter` (32 MB/day size cap, 30-day retention, crash-safe
  `O_APPEND` lines) and `TelemetryEnvironment` (test runs divert to a
  per-process temp dir, issue #159). Screenshots are recorded as
  **dimensions + byte counts**, not pixel artifacts — that answers the
  resolution question without a second retention regime for image files.
- **Live visibility**: each event also emits one `Log.browser` line
  (subsystem `app.tesseract.agent`, category `browser`), so
  `log stream --predicate 'category == "browser"'` shows tool traffic
  without opening the corpus.
- **Toggle**: `browserMCPTelemetryEnabled`, default **on**. Local-only
  capture is consistent with the privacy stance (the corpus is product
  data on the user's own disk, like the prefix-cache traces); the switch
  exists so it can be turned off wholesale.

## Consequences

- Tool-improvement work (screenshot resolution, result phrasing, call
  budgets, error recovery) can now start from measured usage instead of
  suspicion: `jq`/DuckDB over the JSONL answers "what resolution do
  screenshots actually ship at", "which tools error most", "how many
  calls does a task take", per client and per entry path.
- Arguments are recorded verbatim (with long-string caps). That includes
  URLs and typed text — acceptable *because* the corpus is local-only
  and bounded, same as the existing `http-completions` request
  recordings; secrets in password fields are not distinguishable at this
  layer, which is the standing caveat inherited from recording tool
  arguments at all.
- The `handle(request:)` signature gained a required `origin:` — all
  callers (HTTP route, in-process transport, tests) name their path
  explicitly, so a new entry path cannot silently go unattributed.
- Result *text* is capped at a 2,000-char preview (full length always
  recorded), so the corpus is a usage dataset, not a full transcript
  store; the agent's own transcript remains the place to read full
  results for the in-app path.
