# ADR-0040: Companion proactive loop — an entity in a harness (wakes, turns, self-authored instructions)

- Status: Accepted (grilled with the owner, 2026-07-16)
- Date: 2026-07-16
- Relates to: map #301, ticket #307; the anchor experience (#302), tracking model (#308), persona & trust (#309), success metrics (#313), flight recorder (#326); ADR-0035 (memory), ADR-0032 (prepared checkpoint); walking skeleton #303 (`Features/Companion/CompanionHeartbeat.swift`)

## Context

The Companion needs the thing no code path provides today: proactivity. The walking skeleton proved the delivery pipe (fixed-time beats → notification/overlay → recorded outcome) and proved with lived data that a content-free cron ping earns no engagement (19 beats, 1 reply). The ticket originally framed the answer as "a deterministic loop around the model — scheduled, tracking, always checking."

**The owner explicitly superseded that framing in this grilling.** The abandoned first session on this ticket was going wrong precisely because it followed it. The corrected vision, on the record 2026-07-16:

> Jarvis is an **entity**, and the Companion feature is his **harness**. The harness exists to make the entity possible — turns, tools, memory, context, an unbreakable record of his own commitments — never to make his decisions. Not a to-do-list wrapper: **a mind that happens to live in your Mac** (the product slogan, owner-locked). End state: dedicated hardware, the biggest model available, a 24/7 continuous loop whose goal is the human's success. Slogan of behavior: **Be proactive.**

v1 approximates the continuous loop with **turns** on shared hardware — the one honest physical constraint (one GPU, one `.llm` arbiter slot). Everything below is designed so that v1's shapes are seeds of the continuous loop, not cages around it. The North Star lives as its own ticket on map #301.

## Decision

**1. The entity/harness split is the architecture.** The model decides *everything with judgment in it*: whether to speak, when, through which channel, about what, what to research, what to book for its own future. Code contributes exactly three things, none of them judgment:

- **Turns** — the entity cannot grant itself GPU time; the harness wakes it reliably.
- **Continuity** — commitments and loop state survive restarts, sleeps, crashes.
- **The record** — every decision seam emits flight-recorder events (#326); accountability is retrospective (record + weekly review), not preventive (code gates).

The rejected alternative ("code decides whether/when, model composes what") is recorded below — it is the architecture of a reminder app and was explicitly refused by the owner.

**2. Spine: a ticking evaluator with event accelerants.** One wall-clock tick (~30 s) evaluates `(now, persisted loop state, signals) → due wakes / eligibility` — a pure function, replayable over a recorder snapshot. Events (presence transition, Mac-wake, power change) force an immediate extra evaluation so reactions feel instant; they are latency optimizations, never correctness dependencies (a missed event costs ≤ one tick, not a lost morning). The skeleton already proved why armed timers lose to ticks across system sleep (`CompanionHeartbeat.swift:54-55`). The evaluator decides *nothing* but due-ness and eligibility — judgment is the entity's, at the turn it triggers.

**3. The loop lives in the app.** No LaunchAgent helper: beats are conversations on the one agent (one-interface, #308), composition needs the one `.llm` slot, memory, and the conversation store — all app-owned; a helper would be IPC into everything or a duplicate brain. Enabling the Companion **asks once** to enable launch-at-login (`SettingsManager.swift:659`, default off) so the loop survives reboots; a silent login-item flip is a trust violation. App not running ⇒ Jarvis off, visibly (menu-bar presence, recorder gap).

**4. Persistence: wakes are rows; the recorder is never load-bearing.**

- **`Wake` table** in the same SQLite DB as `Day`/`Observation`/`WorkItem` (#308's one-database call): the generalization of #309's promises to *any* self-booked future turn — content ("what I'll want to know then"), due time, class (`promise | rhythm | followup | resummons`), state machine (`booked → fired → engaged / delivered / resurfaced → delivered-unheard`), per-wake summons grant, provenance ref to the booking conversation. The entity books via a typed tool; **state transitions are written only by app code**.
- **Loop day-state**: one small record per day — beats fired and outcomes, active summons + next repeat, discretionary-fire count against the budget, pulse done/skipped. On launch the loop resumes from it; a summons interrupted by a restart re-arms instead of vanishing (the skeleton's known crude gap, fixed structurally).
- The flight recorder logs every transition but the loop **never rebuilds state from JSONL** — SQLite is state, JSONL is audit; one direction.

**5. The single correctness invariant: a wake is consumed only by a completed turn.** Everything reduces to it: restarts, crashes mid-turn, Mac asleep at due time, model failures, arbiter evictions — in every case the wake stays due and re-presents. The must-fire invariant (#309) becomes enforceable mechanics: on every launch and tick, any due unconsumed wake either runs or reaches the catch-up turn (below); none is ever dropped by code.

**6. The check-run contract: every wake is a full agent turn.** No small-model tier, no code-only "cheap check" — the check *is* the entity thinking. On wake, the harness hands the model a code-gathered **situation briefing** — time, presence/idle span, frontmost-app session, calendar (read-through, #308), contract state, its own due and upcoming wakes, recency of last interaction — and the entity decides: stay silent, glyph, notify, speak, summon the overlay, open a conversation, re-book itself, book new wakes. **Silence is a decision it takes and records, not a branch code took.** Composition happens at fire time, never in advance (the callback must be fresh; the skeleton's `composeBody` precedent).

**7. The wake fabric — Jarvis books his own day.** Code owns only two primitive wake classes plus one eligibility rule:

- **Transition wakes** (physics the entity can't foresee): day start (first sustained presence after the overnight gap — wake-linked per #302), Mac-wake, app-launch catch-up.
- **Self-booked wakes**: everything else. The morning turn ends with the entity booking its own midday pulse (calendar-aware placement is its judgment); the evening wake was booked that morning; a summons repeat is "wake me in 12 minutes" (class `resummons`). The anchor rhythm is **his standing instructions, not a cron table**.
- **Ambient turns** (owner-approved for v1): unoccasioned cognition — think, research, notice, book. Eligibility is a harness rule, not judgment: AC power + `.llm` slot free + owner not actively using the agent + spacing (≤ ~2/hour, revisable with wear). The waking analog of ADR-0035's sleep passes.
- **Empty-day safety net**: if the day-start transition fires and no wakes exist for the day (fresh install, wiped state, bug), the day-start turn itself is where the entity re-establishes its rhythm. The system can never wake up empty and stay silent all day.

**8. Full observability (owner requirement): every turn is a conversation.** Ambient, beat, promise, catch-up, sleep — each persists as an origin-tagged conversation in the one chat list, every tool call visible, rendered by the existing transcript UI (state-specific treatment is #327's). A crashed turn persists half-finished — visible, not vanished.

**9. Model and arbiter: one model, owner right-of-way, no residency machinery.**

- The Companion model setting (**default: Qwen3.6-35B-A3B** — owner call: "only with the smartest model it could work reliably"; the 4B is insufficient) becomes the app's default agent model while the Companion is enabled — interactive chats and Jarvis's turns share loaded weights: no swap cost, shared prefix-cache root.
- **The owner always wins the slot.** Ambient turns are eligible only when the slot is free; if the owner arrives mid-turn, the entity's generation yields at the next step and its turn re-presents or re-books (its call, recorded). Occasioned wakes never cancel an in-flight owner generation — they queue for the slot (seconds of delay, wake still counted on time).
- No pinning/keep-warm machinery: normal arbiter load-on-demand semantics; deliberate owner model switches swap back at the entity's next turn (~9 s prepared-checkpoint load, ADR-0032 — invisible in an unattended turn). This resolves the map's "compute/energy policy" fog item.

**10. Delivery is a typed-tool palette; escalation is policy, not code.** One tool per rung — set **menu-bar glyph** (the quietest rung, new), post notification, speak a line, **summon the voice overlay** (#328), open a conversation. The escalation ladder (#302's quiet-first → spoken summons, backoff repeats, engage-or-recorded-dismissal) and all interruption ethics (meetings, focus, quiet hours, promise-class quiet delivery per #309) live in the entity's **versioned standing instructions** — auditable because every recorder trace references the policy version it acted under (#326). Quiet hours are a sentence the owner says to Jarvis, not a code curfew.

**11. The harness keeps exactly three channel-level guarantees** (mechanics, not judgment):

1. An unanswered delivery always leaves a visible artifact — the overlay's banner fallback generalizes to every rung; no summons can evaporate silently.
2. Budget limits refuse **visibly at the tool layer** (the entity sees "budget spent"; nothing is silently swallowed).
3. Must-fire per decision 5.

**12. Self-authored standing instructions (owner directive, 2026-07-16).** No hardcoded personal defaults — the entity *learns* its human. Mechanism:

- A dedicated, versioned **instructions document** injected into the system prompt alongside memory (the memory system is the owner's stated gold standard for this shape).
- The entity edits it through a typed tool; sleep passes review it (the instructions' own consolidation loop); **the owner can read and edit it at will** — reviewable is non-negotiable.
- Every edit is a recorder event; every trace carries the instructions-version hash (#326's policy-by-version-reference, satisfied by construction).
- v1 ships **seeds, marked as seeds**: the anchor ladder as starting escalation policy, the #309 persona register, an adjustable evening-window default. The owner is explicitly "not the persistent person": rhythm facts are the entity's to learn and re-book ("Jarvis, do it in an hour" is just a wake moved), never constants.

**13. Failure semantics.**

- **Mac asleep / app closed at due time**: wakes wait as rows; on Mac-wake or launch, one **catch-up turn** receives everything overdue and the entity triages — late summons ("anchor-flavored"), fold into next beat, or recorded skip. The skeleton's hardcoded 45-minute staleness cutoff dies; staleness is judgment.
- **Model failure at turn time** (load/generation error, OOM): harness retries with backoff, wake stays due; persistent failure ends in the **generic fallback delivery** — a plain notification carrying the wake's own stateable line (every wake is announceable, #309). Generic is survivable, invented is not; never-silent-give-up holds with the brain offline. Failure is its own recorder event class.
- **Crash mid-turn**: wake unconsumed → re-presents on relaunch; the partial conversation remains in the list.

**14. The determinism boundary, restated.** Deterministic: that turns happen, that state survives, that the record is complete, that the three guarantees hold, that aggregation for the weekly review is computed by code (#326 — the model narrates, never tallies). Judgment: everything else, the entity's, at every wake. LLMs aren't deterministic — so the harness is deterministic about *delivery and memory of commitments*, and free about nothing else being code's business.

## Considered / rejected

- **Code-owned trigger evaluation with model-as-composer** ("the model never decides whether or when Jarvis speaks — only what he says"): explicitly rejected by the owner — "opposite of what I wanted"; it is a 10-years-ago app with an LLM bolted on. Recorded because the abandoned first grilling session drifted here and future sessions must not.
- **A small-model cheap-check tier**: dies from both directions — trigger due-ness needs no model, and anything with judgment deserves the real entity (runtime assumption: previous-gen-Sonnet-class, #309).
- **Pure event-driven spine / armed timers**: timers die with system sleep (skeleton evidence); missed events fail silent, violating never-silent-give-up.
- **LaunchAgent helper**: IPC into agent, arbiter, memory, conversations — or a second brain. Structurally wrong under one-interface.
- **Pre-scheduled `UNUserNotificationCenter` content**: composition must be fresh at fire time; a notification scheduled yesterday says yesterday's words.
- **Recorder as loop-state store**: makes the audit log load-bearing; #326 forbids it.
- **Code-enforced quiet hours / interruption gates**: owner's no-restricting-rules call; ethics live in versioned instructions, enforcement is the record + weekly review.

## Consequences

- The walking skeleton (`CompanionHeartbeat`) is absorbed: its tick loop and overlay/notification fallback survive as harness mechanics; its fixed schedule, hardcoded staleness, and in-memory outstanding-ping state are replaced by the `Wake` table + day-state. v0 JSONL imports at recorder cutover (#326).
- The map's "compute/energy policy" fog item is resolved (decision 9).
- `tesseract` gains: `Wake` + loop day-state tables beside #308's, the wake evaluator, the situation-briefing builder, the delivery-palette tools, `book_wake`, the instructions document + editing tool, catch-up-turn plumbing, launch-at-login ask, 35B-A3B default while Companion enabled.
- Feeds relayed: #310 (voice loop: summons/overlay rungs are entity-chosen; auto-listen/barge-in operate within turns), #327 (home surface: origin-tagged conversations, state badges, menu-bar glyph states, instructions-document review surface).
- CONTEXT.md glossary gains: **Wake, Turn, Ambient Turn, Transition Wake, Situation Briefing, Standing Instructions (self-authored), Entity/Harness split** (with #308's pending additions).

## Accepted costs

- ~10–20 full-model turns/day (beats + self-booked + ambient) on the 35B — thermally trivial on AC-gated ambient turns; battery cost of occasioned turns accepted as the product.
- Making 35B-A3B the default agent model changes the interactive default too — deliberate (one model, one mind).
- Model judgment will sometimes be wrong (a mistimed ping, an unwise silence): caught by the record and the weekly review, not prevented by gates. This is the point, not a gap: guardrails-as-crutches were rejected with the runtime assumption in #309.
- Wrong-taste risk on delivery choices is bounded by the wearing loop (#313's annoyance-in-band measure), not by code.

## v2 direction (North Star, not v1 scope)

Dedicated always-on hardware (Mac mini/Studio class), the largest local model available, the tick approaching continuous cognition; the entity setting its own goals within the owner's; rhythm fully learned, seeds long since rewritten by their owner — the entity. v1's wake fabric is deliberately the same architecture at lower duty cycle: nothing here needs to be unbuilt to get there.
