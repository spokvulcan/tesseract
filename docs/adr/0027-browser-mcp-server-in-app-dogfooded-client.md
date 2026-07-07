# Browser MCP server lives in-app; Tesseract's agent dogfoods it via a full MCP client

Status: proposed

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
