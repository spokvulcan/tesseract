# Browser MCP server lives in-app; Tesseract's agent dogfoods it via a full MCP client

Status: accepted

The browser capability (ADR-0026) must serve two consumers: external coding
agents (Claude Code et al., including offline ones using the local browser as
their window to the web) and Tesseract's own agent.

**Decision:**

- **The server is in-app.** Tesseract hosts a streamable-HTTP MCP endpoint on
  the existing loopback `HTTPServer` (alongside `/v1/chat/completions`) — one
  process owns the Agent Browser and arbitrates it among all clients. External
  agents connect with one line (`claude mcp add --transport http …`). No
  standalone binary: per-client stdio servers would fight over one browser and
  need a broker daemon anyway.
- **Per-client Browser Sessions.** Each MCP client gets private tabs (own
  `WebPage` instances) over the shared Agent Profile data store — concurrent
  agents never trample each other's navigation; login state is common.
- **Tesseract's agent consumes it via MCP, not in-process.** A full MCP client
  ships in the agent (arbitrary user-configured servers, settings UI, consent
  flow), and the Browser MCP Server is just its first registered server. Dogfooding
  keeps the one tool surface honest and buys the MCP ecosystem for the local
  agent in the same stroke, at the cost of shipping a client before browser-use
  lands for the internal agent.
- **HTTP transports only (v1).** Streamable HTTP + SSE-legacy. Stdio servers are
  excluded because the App Sandbox makes them unworkable in-process — children
  of a sandboxed app inherit its sandbox at the kernel level, `/usr/local` /
  `/opt/homebrew` are not even exec-able under the profile, and the
  SMAppService escape was deliberately closed in macOS 14.2
  (https://developer.apple.com/forums/thread/706390,
  https://developer.apple.com/forums/thread/743395). The verified future path,
  if stdio demand materializes, is a bundled **non-sandboxed XPC service**
  (legal for Developer-ID, launchd-spawned so its sandbox is independent of the
  host's) handing pipe FDs back over XPC — not dropping the app sandbox.

Consequences:
- Browser tools exist only while Tesseract runs — acceptable; the inference
  server already sets that expectation.
- The MCP client and the browser server are separable workstreams; external
  agents can use the browser server before the internal client lands.
- The loopback HTTP endpoint must reject non-local origins (DNS-rebinding
  guard) per MCP transport security guidance.

**Realization (#190):** the full MCP client shipped as an ``MCPClient`` over a
transport abstraction: ``HTTPMCPTransport`` (URLSession, for arbitrary
user-configured servers) and ``InProcessMCPTransport`` for the built-in Browser
server. The in-process transport dispatches real MCP JSON-RPC straight to
``MCPBrowserServer.handle(request:)`` — same protocol, same request handler, no
socket. This refines "the agent connects to `127.0.0.1:<port>/mcp`": consuming
the browser server *is* dogfooding at the protocol layer, but going in-process
(rather than over the loopback listener) decouples browser-use in chat from the
inference HTTP server, which only starts with `isServerEnabled`. The built-in
Browser server's enablement is the one "Browser Access" switch
(`browserMCPServerEnabled`); the web-access switch continues to gate its tools
(US #16), pinned to the canonical `browser` namespace so the gated set cannot
drift from the tools the server materializes. No stdio in v1 (App Sandbox), as
above.

**v1 scope (#190).** Tools-only, Streamable HTTP only. Consent is add-time: a
second gate on first tool-list materialization (US #3 as literally worded) would
contradict the zero-config pre-registered Browser server and is redundant for a
user server that was approved seconds earlier, so it was not built. Deferred with
tracking issues: a standing server→client SSE channel for idle
`tools/list_changed` (#193, US #13 while no call is in flight), the SSE-legacy
fallback transport for older servers (#194), and Keychain-backed header secrets
(#195, US #14 — v1 persists them in the sandboxed settings store).
