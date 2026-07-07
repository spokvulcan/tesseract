# Agent Browser: embedded WebKit over a dedicated, user-curated profile

Status: proposed

Agents (Tesseract's own and external coding agents) need to browse the web with
the user's authentication. The industry-standard paths were researched and
rejected on evidence (2026-07-07 session):

- **CDP against the user's real Chrome** is dead: Chrome 136+ (stable
  2025-04-29) refuses `--remote-debugging-port`/`--remote-debugging-pipe` on the
  default data dir, and a non-default `--user-data-dir` gets a different
  App-Bound-Encryption key — i.e. no live auth
  (https://developer.chrome.com/blog/remote-debugging-port).
- **Chrome extension + native-messaging host** (the Claude-in-Chrome shape) does
  reuse real logins, but the host manifest lives outside any sandbox container —
  it forces a second distribution artifact, an extension install, and puppets
  the user's live browser, fighting them for it.
- **Importing Safari/Chrome cookie stores** is the infostealer threat model
  (App-Bound Encryption exists specifically to stop it) and is permanently out
  of bounds for a privacy-first app.

**Decision:** the **Agent Browser** is embedded WebKit (the macOS 26 `WebPage`
API) inside Tesseract, over a persistent `WKWebsiteDataStore` that forms the
**Agent Profile** — a credential silo the user logs into deliberately, one site
at a time, in visible browser windows. Tool semantics are kept engine-neutral so
a CDP/Chromium backend could slot behind the same surface if WebKit site
compatibility ever demands it.

Three deliberate corollaries:

- **Always-visible browsing.** Agent sessions render in real windows, never
  headless. Transparency is the oversight mechanism, and the same windows are
  where the user logs sites into the profile.
- **The profile boundary is the security model.** No server-side content
  sanitization, action gating, or per-site permissions in v1 — an explicit,
  owner-accepted risk. Injected page content can steer whichever agent reads
  it; containment comes from curating which identities the Agent Profile holds
  and from watching it work. Layered mitigations (untrusted-content framing,
  credential-field blocks, confirmations) are additive later if the profile
  accumulates sensitive logins.
- **Full replacement of the web stack.** The Agent Browser absorbs
  `HeadlessRenderer`, `web_fetch`, and `web_search`; anonymous reads survive as
  **Ephemeral Pages** (cookieless, outside the profile). One WebKit owner, one
  extraction pipeline.

Consequences:
- The user re-authenticates sites once inside the Agent Browser; sessions
  persist thereafter. No access to existing Safari/Chrome logins — by design.
- Zero TCC prompts and full App Sandbox compatibility (only the existing
  `network.client`/`network.server` entitlements).
- WebKit limits are accepted: no trusted synthetic input events, no
  network-tab inspection; JS runs via isolated `WKContentWorld` injection.
