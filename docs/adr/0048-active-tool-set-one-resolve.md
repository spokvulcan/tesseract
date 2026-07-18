# ADR-0048: The Active Tool Set is one resolve; the prompt derives from it

- Status: Accepted
- Date: 2026-07-19
- Relates to: ADR-0028 (browser.* is the sole web surface; the gate's name
  set), ADR-0040 §10 (companion audience), ADR-0046 #372 (dialogue audience),
  ADR-0044 (the same one-decision-many-callers move on the prefill route)

## Context

"Which tools does the agent see right now" was computed at eight sites with
three precedence encodings and two audience rules:

- `ToolRegistry` concatenated built-ins and extension tools;
  `ExtensionHost` deduped first-wins across extensions; `MCPToolsExtension`
  deduped first-wins within MCP.
- `AgentRunController.syncActiveTools` filtered audience (`.companionOnly`
  always, `.dialogueOnly` unless a dialogue is open) and applied the Web
  Access gate — per turn.
- `DependencyContainer`'s companion factory applied a *different* audience
  rule inline (drop only `.dialogueOnly`) as a post-construction
  `updateTools`.
- `AgentFactory` seeded the agent — and the system-prompt assembler — with
  the **unfiltered** registry.
- `SystemPromptAssembler` re-inspected tool membership to decide its
  orientation sections.

The prompt/callable split was a live defect in both directions. The prompt
is assembled once at bootstrap and never rebuilt, from the unfiltered
registry; the callable set is filtered per turn. With Web Access off, the
static prompt kept instructing the model to work through `browser.search`
while every browser tool was stripped from the turn. And because the
built-in Browser MCP server connects asynchronously, an agent built before
the connection landed omitted the web orientation even with Web Access on —
the section was decided by build timing, not by the setting.

## Decision

One pure module, **`ActiveToolSet`**, is the home of the resolve:

- **`ToolGating`** names the consumer (`.interactiveChat`, `.dialogueChat`,
  `.companionHeadless`) and the Web Access switch. Audience rules are
  consumer facts here, nowhere else.
- **`ActiveToolSet.resolve(from:gating:)`** maps the registry-ordered tools
  to the live set. Order is preserved — registry order remains the loop's
  dispatch precedence.
- **`ActiveToolSet.promptFacts(for:)`** derives **`PromptToolFacts`**
  (`hasSkillTool`, `carriesBrowserTools`) from the *resolved* set — never
  from the raw registry. `SystemPromptAssembler` consumes the facts; its
  membership predicates are deleted.

Both agents resolve through it: `AgentFactory.makeAgent` takes a
`ToolGating`, seeds the agent with the resolved set, assembles the prompt
from the resolved facts, and wires a **system-prompt reassembler** onto the
agent (capturing the loaded context, skills, and agent root).
`AgentRunController.syncActiveTools` shrinks to naming its gating context,
resolving, and calling `agent.syncSystemPrompt(facts:)` — which rebuilds the
prompt through the reassembler **only when the facts change**, guarded like
`updateTools`. The companion factory's inline filter is deleted; it passes
`.companionHeadless` instead.

The prompt is no longer static-but-wrong: it is stable until a real
orientation change (a Web Access flip, or browser tools materializing after
a late MCP connect), and each such change costs one deliberate prefix-cache
invalidation on the system-prompt head — priced against the permanent
alternative of a prompt that lies to the model about its own tool surface.

Deliberately preserved, not decided here: the Companion's headless turns
are **not** governed by the Web Access switch (`webAccessEnabled: true` is
pinned at its factory call). That was the shipped behavior; gating the
Companion's web access is a product decision this refactor declines to make
silently.

## Consequences

- The audience × web-gate × consumer interplay is a pure decision table
  (`ActiveToolSetTests`), and the prompt/callable consistency invariant —
  previously unrepresentable in any suite — is pinned: for every gating,
  `promptFacts(for: resolve(...))` agrees with the resolved set.
- `ToolRegistry` keeps identity and precedence; `ExtensionHost`/MCP keep
  their dedupe; the loop's dispatch scan operates on the already-resolved
  `context.tools`. Those are identity/order concerns, not gating — folding
  them was considered and rejected as a shallow rename.
- Future audiences (the Notification Hub's companion tools ride
  `.companionOnly` unchanged) and future MCP servers add rows to the data,
  not homes to the code.
