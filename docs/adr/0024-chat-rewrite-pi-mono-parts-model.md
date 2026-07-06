# Chat rewrite: pi-mono parts model, live-part rendering, flat document UI

The agent chat is rewritten vertically — message model, event protocol, state
ownership, persistence format, and every view — despite the previous stack
(AgentEvent → AgentStateReducer → ChatTranscript projection → throttled
tail-splice) being functional and heavily perf-tuned. Four indictments drove
the decision, and each maps to a specific replacement:

1. **Message model too weak** → the canonical model becomes a verbatim Swift
   mirror of pi-ai's types (`packages/ai/src/types.ts`): `Message =
   user | assistant | toolResult`, assistant content is an ordered
   `[ContentPart]` (`.text` / `.thinking` / `.toolCall`), and the stream
   protocol is `start/delta/end` events addressed by `contentIndex`, each
   carrying the partial message. OpenCode's parts-as-identified-records model
   was considered and rejected to preserve strict pi-mono alignment; row
   identity derives from (message ID, part index) instead.
2. **Streaming pipeline was a tower of workarounds** (50 ms throttle +
   tail-splice + `streamingRowVersion` + eager/lazy stack swap) → replaced by
   **live-part rendering**: committed content is pure `Equatable` value rows;
   exactly one `@Observable` live-part box exists while a part streams, so a
   token delta invalidates exactly one `Text` view by construction. On
   part-end the box commits into the value rows. Live markdown renders during
   streaming via throttled re-parse inside the box (no commit-time style jump).
3. **Coordinator god-object** (670-line `AgentCoordinator` dispatching eight
   sub-controllers) → one `ChatSession` `@Observable` store holds the single
   event fold (messages, live part, run phase) as the only agent-event
   subscriber; everything not derived from agent events (composer draft,
   voice, skill pills, palette) becomes leaf controllers owned by the views
   that use them. An event bus with independent subscribers was rejected: it
   reintroduces the subscriber-ordering hazards the old code documented.
4. **Perceived UI performance** → gated by evidence at merge: Instruments
   capture proving per-delta work is confined to the live part, unit tests on
   the fold/projection/persistence, a repeatable streaming stress scenario,
   and a manual acceptance checklist.

UI decisions: flat document transcript (assistant text directly on the window
background in a readable column; user messages as neutral compact blocks) —
fully monochrome, no brand color in the transcript (magenta retreats to
chrome). Tool calls and thinking are flat sequential collapsible one-line rows
with **no turn-grouping header** (the "N steps" layer and its
`activeTurnStart` splice machinery are deliberately dropped). Composer layout
is preserved, gaining a model pill (with honest load-latency state) and an
inline skill pill; all errors/status/hints consolidate into the single
in-composer banner slot (the PRD #170 Appshot-explainer style), retiring the
status strip and standalone error banner. Conversations stay in the history
popover — a session sidebar was considered and rejected to keep the focused
single-column feel. Liquid Glass is restricted to exactly two custom surfaces
(composer bar, slash popup) sharing one `GlassEffectContainer`; the content
layer never gets glass.

Consequences: the persistence format is replaced and existing v2 conversations
are **wiped on first launch** (no migrator — matching the store's existing
version-mismatch policy); the rewrite lands as **one big-bang PR**; the old
pipeline's hard-won workarounds (deferred scroll, eager/lazy swap) are
re-derived only if the live-part design actually re-hits those AppKit
re-entrancy bugs.
