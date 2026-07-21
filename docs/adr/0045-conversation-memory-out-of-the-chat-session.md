# ADR-0045: Conversation Memory — the chat's memory fold, out of the Chat Session

- Status: Accepted
- Date: 2026-07-18
- Relates to: ADR-0035 (the living memory this is the chat side of),
  ADR-0024 (the Chat Session event fold this returns to), ADR-0034 (the
  closure-injected policy shape this follows), ADR-0042/0043/0044 (the same
  extraction move elsewhere in the 2026-07-18 review)

## Context

CONTEXT.md defines the **Chat Session** as the event fold with leaves outside
it — but ~130 lines of ADR-0035 memory decoration sat inside it, off the
`handle(event)` test seam: the injection dedupe set, `attachMemory`, the
wrapper unwrap/rewrap statics, and `captureEpisode`.

Two costs, both already paid once:

- **The seam had no tests.** The send/memory path bypasses `handle(event)`,
  and a bare `outgoing as? UserMessage` unwrap shipped — the pipeline hands
  `prepare` a `CoreMessage.user`, so memory was retrieved and then silently
  dropped on the floor on every single turn. Not one unit test caught it.
- **The "is this a `CoreMessage.user` or a bare `UserMessage`" fact was
  re-decided at four sites across three files** — `ChatSession`'s tested
  static, hand-written ladders in `AgentConversation.title` and
  `AgentConversationStore.persists` — while the canonical dual-shape
  accessor, `AgentMessageProtocol.asUser`, already existed beside them.

## Decision

Extract the fold as **`ConversationMemory`** — a `@MainActor` policy object
over two injected closures (the ADR-0034 shape; production wires
`MemoryEngine.injection` / `MemoryEngine.record` via `init(memory:)`, tests
hand canned closures), owned optionally by the Chat Session:

- **`enrich(outgoing)`** — the §5 read path: unwrap through the canonical
  `asUser`, recall with `forEpisode: user.id` (the lifecycle's sensor), dedupe
  against `injectedMemoryIDs`, and return the enriched message in the same
  wrapper it arrived in. Identity for non-user messages and quiet turns.
- **`capture(reply:context:conversationID:)`** — the §6 write path, the chat
  door of One Door Per Testimony: the last user message in the context is the
  turn's testimony, the episode takes that message's own id (one turn, one id,
  in both directions), the trimmed reply rides in `meta` capped at 2,000
  chars. Detached; returns the task handle for tests, nil when the turn
  writes nothing.
- **`reset()`** — the conversation boundary: a new window carries none of the
  old injections.

The Chat Session keeps what is genuinely its own: raising/swapping the
Pending Row (byte-identity between the row and the message the agent got, now
enforced by using `enrich`'s return value directly) and knowing which
conversation is current. The two hand-written ladders collapse onto `asUser`;
`persists` keeps its third, legacy `AgentChatMessage` leg, which `asUser`
does not cover.

## Consequences

- `ConversationMemoryTests` pins the fold as decision tables through the
  interface — both wrapper shapes, the dedupe set's lifecycle across `reset`,
  the episode-identity rule, reply trimming/capping — with no store, no
  embedder, no model. The bug class that shipped is representable in a test.
- The dual-shape fact has one home (`asUser`); drift between the unwrap
  sites is unrepresentable.
- The Chat Session returns to the shape its glossary entry claims, and no
  longer knows `MemoryEngine` at all — its dependency is the two-verb
  collaborator.
- Behavior-preserving: no observable delta on main. `title` and `persists`
  accept exactly the message shapes they did before.

## Continuation (#408, 2026-07-22)

The move this ADR made once was made twice more when the Chat Session diet
finished (the 2026-07-21 architecture review's Worth-exploring #9). The leaf
list off the fold's `handle(event)` seam grew: memory (here), then skill
execution and opening context. **`SkillExecution`** took `executeSkill`'s
inline home — argument assembly, the Skill Envelope injection render (#401),
usage recording — leaving the Chat Session one thin call that sends the leaf's
rendered injection on the spine. **`openingContext`** collapsed the session's
two `companionIdentity` + `foldBriefing` sources into one composed closure the
container wires per ADR-0052, so the session stops knowing there are two
sources (the boundary reset of both rides `onConversationSwitch`). Same shape,
same win: the extracted logic gains its own decision-table tests
(`SkillExecutionTests`), the session's init shrank (20 → 18 parameters), and
the type moves further toward the fold-only shape its glossary entry claims.
No observable delta — the rendered injection and the injected-context order
(identity, then briefing, then memory) are byte-identical to the inline paths.
