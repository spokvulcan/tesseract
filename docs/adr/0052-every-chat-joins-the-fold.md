# ADR-0052: Every chat is a conversation of the fold

- Status: Accepted
- Date: 2026-07-19
- Relates to: ADR-0040 (the entity/harness split; §10 delivery ladder; §11
  never-silent guarantees), ADR-0045 (injected context rides the message,
  not the system prompt), ADR-0046 (the Event Fold; Report-Back; the
  dialogue ledger #372), ADR-0048 (tool audiences resolve per consumer),
  ADR-0051 (reaction effects)

## Context

The Event Fold made Mission Control the one mind: every perception an
Event, every turn on the record, dialogues owing a Report-Back so "the one
mind knows what its conversations concluded" (CONTEXT.md). But only
*summoned* dialogues were its conversations. An owner-opened chat was a
free-floating stranger: it received the IDENTITY persona block and shared
memory retrieval, and nothing else — no fold state, no wakes, no contract,
no deliveries, no conclusions. The asymmetry was total in one direction:
the fold sees chats (their turns become episodes the loop's memory
enrichment retrieves), chats see nothing of the fold.

The first field night proved the cost. The evening-journal wake fired at
22:00; Jarvis bannered three times; the owner opened a fresh chat instead —
and that chat's Jarvis, mid-campaign by his own fold's account, answered "I
don't have a specific agenda for today." Clicking the banner was equally
dead: the ping→conversation correlation lived only in an in-memory dict
(gone on relaunch), and even alive it routed to the Mission Control
transcript — read-only by ADR-0046's guard, a viewing room with nothing to
type into.

## Decision

The interactive/dialogue *contract* split dies. Every owner-facing
conversation is a conversation of the one mind, under one uniform
contract; origin tags survive as provenance only.

**Fold Briefing in.** Every chat's first message carries a code-rendered
`<fold-briefing>` block — today's contract and keystone, due and
recently-fired wakes with states, recent deliveries with their lines, the
last few fold-turn conclusions verbatim — gathered mechanically from the
stores on the Situation Briefing's discipline: gathering is mechanical,
interpreting is the entity's job. It rides `injectedContext` (ADR-0045),
so the prefix-cache root survives and the transcript records what the turn
saw. A later message re-injects a fresh briefing iff the fold advanced
meanwhile (new fold turn past the last briefing's stamp) — a resumed chat
is never behind its own mind.

**Report-Back out.** `report_back` widens from dialogue-only to every
chat; the dialogue ledger (one wind-down/quiet nudge, entity judges
whether anything concluded) generalizes from "the active dialogue" to "the
active chat". The briefing ends with the contract line, so milestone
deposits happen unprompted. The harness never authors a summary — what a
conversation concluded is judgment, and judgment is the entity's
(ADR-0040).

**Banner click = summons engaged.** Clicking a `notify` banner begins a
Dialogue seeded with the banner's line as the entity's first words — the
same engage contract as the voice overlay — with the correlation persisted
in the notification's `userInfo` so a click survives relaunch. A click on
a banner correlated to a live dialogue conversation reopens it; Mission
Control's transcript is never a click destination.

**`notify` lives.** Considered and rejected: removing the tool (the banner
is the only rung that survives the owner being away, respects Focus, and
backs §11's unanswered-summons fallback) and replacing it with custom
banner windows (loses Notification Center persistence, lock screen, and
Focus integration; duplicates `summon_overlay`). The felt uselessness was
the dead click and the context-blind follow-up chat — both fixed above.
Rung *choice* stays entity policy: the LOOP POLICY seed gains the presence
rule (owner present with screen unlocked → prefer `speak`; away or locked
→ banner), and the entity keeps rewriting it with wear.

## Consequences

- The chat tool consumers converge: with `report_back` in every chat,
  nothing distinguishes the interactive and dialogue resolves (ADR-0048's
  consumer table shrinks).
- Trivial chats cost one nudge line and a brief "nothing concluded"; the
  entity, not code, decides what is worth depositing.
- Briefing tokens (a few hundred) ride every chat opening and each fold
  advance; bounded by fold activity, not chat length.
- The fold hears about every conversation, so a chat that contradicts a
  running campaign (the evening-journal case) becomes visible to the next
  fold turn as a deposit instead of inferred from memory scraps.
