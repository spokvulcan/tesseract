# ADR-0035: Living memory — an immutable episodic layer under a first-person semantic layer, with salience decided in sleep

- Status: Accepted
- Date: 2026-07-12
- Relates to: map #314 (Memory — the living, brain-inspired memory system), tickets #319–#325; map #301 (Companion), ticket #302 (the anchor day); ADR-0034 (Proofread Pass — own co-resident MLX model), ADR-0015/0018/0019 (prefix cache), ADR-0024 (chat parts model)

## Context

Tesseract's memory today is `memories.md`: six flat lines of third-person prose in the agent's file sandbox, read only when the model volitionally calls the `read` tool (verified: `AgentPackages/personal-assistant/prompts/APPEND_SYSTEM.md:12`; a real conversation on disk opens with `{"name":"read","argumentsJSON":"{\"path\":\"memories.md\"}"}`). There is no structure, no retrieval, no lifecycle, and no guarantee recall happens at all. `CLAUDE.md` makes memory a first-class product goal — "knowing the user is first-class… persistent memory of goals, habits, and preferences is the product, not a feature" — so the gap is strategic, not cosmetic.

Three research tickets (#315 industry survey, #316 academic sweep, #317 human-memory grounding) surveyed ~200 primary sources. Their convergent findings, and the owner's grilled decisions on them (map #314, 2026-07-12), fix this design. Four alternatives were weighed and rejected:

1. **Plain agentically-edited markdown with version history** (the frontier's substrate of choice — Claude Code's `MEMORY.md`, the Anthropic memory tool, Letta's MemFS). Rejected as the *whole* answer. An LLM rewriting a memory from that memory's own previous text is Bartlett's serial-reproduction chain with a confabulation engine in it: Bergman & Roediger measured recall error rising .57 → .75 over a week, with each recall making the *next* one more distorted (.36 → .18); in the DRM paradigm, **72% of false recognitions of never-presented words received "remember" judgments — identical to the rate for real words**. Version history records the drift; it does not prevent it, because every version is a re-encoding of the previous version. What is needed is a layer that is *never re-encoded*.

2. **Usage-driven retirement — "a memory never used retires"** (the owner's original locked call, and the one thing no shipped system does). Rejected on evidence, and replaced by the owner at the final grill. "Retire the unused" is the **Law of Disuse, demolished in 1932**; three independent literatures agree that disuse is not a *cause* of forgetting (interference is), that the long tail is an external store's entire value, and that "never retrieved" is usually a fact about the *index*, not the memory. The allergy case is the proof: rare-but-critical facts are seldom "used" and no shipped system protects them.

3. **A write-time 1–10 importance judge** (the industry default). Rejected, and replaced by the owner at the final grill. Salience is assigned **retroactively** in the brain — McGaugh's modulation effects are post-training and invisible for 6–24h — so a write-time judge answers a question whose determining information has not yet arrived. It is also the single most expensive thing one could put on a turn boundary.

4. **Parametric / weights-level memory** (Titans, LoRA-as-memory). Settled as out of scope by #316: opaque, GPU-bound, and incompatible with a store the user can inspect, contest, and delete.

The constraints the codebase imposes are as decisive as the science:

- **The system prompt is the radix prefix cache's tree root.** `PrefixCacheManager` plants a type-protected `.system` checkpoint at the `system + tools` boundary, shared across every conversation (`PrefixCacheManager.swift:1020-1024`). A memory block that varied per turn *inside* the system prompt would orphan that checkpoint and force a full re-prefill on every turn of every conversation. The codebase already solved this for skills: inject as a **user message**, "never a system-prompt mutation, so the prefix cache's stable prefix is untouched" (`ChatSession.swift:668-672`).
- **The extension event system is dead code.** `ExtensionRunner` is never instantiated; `PersonalAssistantPackage.swift:20`'s `// TODO: Add .beforeAgentStart handler to inject memories into context` points at a seam that does not exist. Hooks must be wired directly.
- **The GPU lease is FIFO with no priority and no preemption** (`InferenceArbiter.swift:122-137`). Work outside a turn queues behind chat.
- **ADR-0034 established that a second co-resident MLX model is architecturally safe** — the 0.8B proofreader lives entirely outside the arbiter's slots, reads `isGPULeaseHeld` and skips rather than queueing. It is the template.

Measured this session, on this machine:

- System SQLite is **3.51.0 with FTS5, JSON1, and WAL** — a zero-dependency store.
- `MLXEmbedders` is **already vendored** in `Vendor/mlx-swift-lm` with `Qwen3-Embedding-0.6B-4bit-DWQ` in its registry. Downloaded (335 MB) and smoke-tested: loads in **0.4 s**, produces **1024-dim** vectors at **334 texts/sec** warm, and separates a paraphrase (cosine 0.70) from unrelated text (0.35–0.40).
- The owner's corpus is **66 conversations / 42 MB** of JSON at `{id, title, messages: [{type, payload}], createdAt, updatedAt}`.

## Decision

### 1. Two layers. The lower one is immutable, and no agent may edit it.

**`Episode`** — append-only, never rewritten, never deleted by any automatic process. A verbatim record of something that happened: a chat turn, a Companion interaction, a dictation take. This is the source of truth and the only thing that can ever correct a drifted belief.

**`Memory`** — the mutable, derived, **first-person** layer: the assistant's own beliefs ("I've noticed Bohdan starts with the hardest task"), each carrying `source_refs` back into the episodes it came from. The rewriter **always sees the source episodes**, never only its own prior output.

**Both layers are kept permanently.** Moscovitch et al. 2016 is explicit that a schematized memory "can regain its specificity with appropriate reminders" — the literature rejects the distil-and-discard model the industry has implicitly adopted. Retirement is **demotion in retrieval priority, never deletion**.

### 2. Memories are atomic and carry their provenance.

One memory = one claim = one provenance marker. `provenance: STATED | INFERRED` distinguishes what the user actually said from what the assistant concluded — **a safety field, not a nicety**, because a reconstructive system cannot recover the distinction later from the inside. Narrative coherence, which atomic facts alone cannot carry, is supplied by the **core tier** (§5) and by **pattern memories** (§7), not by mixing provenance inside one record.

`kind` discriminates `.belief` (a stable trait or preference), `.event` (an assertion with a temporal argument), `.pattern` (a distilled regularity), `.directive` (how the owner wants the assistant to behave). `specificity: .specific | .general` marks which retrieval pathway a record wants — episodes want separation, generalizations want completion — and **semanticization is a `.specific → .general` migration**.

**No `PromiseRecord`.** Prospective memory decomposes into a retrospective component (the content — ours) and a prospective component (noticing it is time to act — **attention, not memory**, and therefore the Companion's scheduler on map #301). Memory stores assertions with temporal arguments and stable IDs; it does not store triggers. And open loops get **no salience bonus**: the Zeigarnik effect is dead (2025 meta-analysis, 38 publications, recall ratio 0.99).

### 3. Two strength numbers, and storage strength never decreases.

Per memory:

- **`stability` (`S`, days)** — the time-constant of a **power-law** need-probability curve, `R(t) = (1 + FACTOR · t/S)^DECAY` with FSRS-6's fitted `DECAY = −0.1542`. `R` is **not** recall probability (our reads never fail) — it is **need-probability**, i.e. Anderson & Schooler's rational analysis, whose log-odds *is* ACT-R base-level activation. Two research tickets converged on this equation from different literatures.
- **`storageStrength` (`SS`)** — **monotone non-decreasing, forever**. Bjork's asymmetry, which a brain can only approximate and we can implement *literally*, because storage is cheap. `SS` is the **deletion guard**, and it is the answer to the rare-but-critical problem (the allergy mentioned once, needed in six months) that no shipped system solves.

Strengthening is gated on a **useful** use and scales *inversely* to current retrievability (FSRS's `(e^{(1−R)·w} − 1)`), so a memory revived after long dormancy gains far more than one used twice in an hour.

**Retrieved ≠ useful.** `lastUsefulUseAt` and `usefulUseCount` advance only on a graded-useful outcome; `lastSeenAt`/`seenCount` are diagnostic only. "Retrieved and ignored" is **not a lapse** — it usually indicts the retriever, so it decrements a per-cue affinity and leaves `S`, `SS` untouched.

### 4. Retire the superseded. Demote, never delete.

`tier ∈ {.core, .hot, .warm, .cold}` — and **never `.deleted`**. `.cold` means out of the default retrieval pool, reachable only by explicit `memory_search` and by the ε-slot. Three paths lead there:

1. **Superseded.** Age is not consulted. This is the primary path — how memories actually retire.
2. **Surfaced repeatedly and never once useful** (`seenCount ≥ 8 ∧ usefulUseCount == 0 ∧ SS < θ`). Note the difference from the Law of Disuse: *"never retrieved"* is a fact about the index and proves nothing, but *"retrieved eight times and it never once helped"* is evidence about the memory. This is the path that actually clears extraction noise.
3. **Never useful, and need-probability has finally decayed** (`R < 0.45 ∧ SS < θ ∧ usefulUseCount == 0`).

> **A calibration finding, recorded because it nearly became a silent bug.** This ADR originally set the path-3 threshold at `R < 0.05`. That is unreachable: a *power-law* tail does not collapse. With `DECAY = −0.1542`, need-probability at initial stability is still **0.34 after a decade**, and reaching 0.05 takes **~2.3 million years**. The rule would have been dead code, and the store would have grown without bound. A unit test caught it (`MemoryLifecycleTests.theTailNeverCollapses`). The threshold is now calibrated *against the curve* — 0.45 is reached after roughly 18 months of a memory never once being useful — and path 2 exists precisely because a time-based rule alone is, correctly, far too slow to be the only answer. The heavy tail is the design working as intended; it just means retirement cannot be keyed to a small absolute `R`.

In all three paths, a memory that was **ever** useful has `SS > 0` and is protected — forever. That is the allergy guarantee.

**The ε-exploration slot is not optional**: ~1 retrieval slot in 20 is drawn from `.warm`/`.cold` and its outcome logged. Without it the lifecycle trains on its own priors forever and the cold tail can never come back — the counterfactual (memories never retrieved) is otherwise unobservable.

**Only the owner deletes.** True deletion exists exactly once, in the Memory window, by explicit human act.

### 5. The read path: a stable working set injected as a user message, plus agentic search.

Retrieval score is **multiplicative**, because the terms are probabilities of independent failure modes:

```
score(m, cue) = relevance(m, cue)              // hybrid: cosine ⊕ FTS5/BM25
              · needProbability(t_m, S_m)      // the decay term
              · (1 + log1p(storageStrength_m))  // the entrenchment term
              · (superseded ? 0.1 : 1)          // the interference term
```

**Injection is a `<memory>` block in the first user message of a conversation** — never a system-prompt mutation (§Context). The block carries the **core tier** (promotion's concrete grant: unconditional presence — these have stopped being retrievals and become identity) plus the top-ranked retrieval for the opening turn. Later turns do not re-inject, so the conversation's cached prefix stays stable; new depth is reached through a **`memory_search` tool**, which frontier-capable local models use well and which is where the design's "assume frontier capability" call cashes out.

Retrieval **logs, and does not grade**. Every memory placed in context is written to a retrieval log with its turn; the grade is assigned later, by sleep (§6).

Because memories are only *formed* in consolidation, same-day recall is served by the episodic layer: retrieval searches **episodes as well as memories**, so something said at 10am is available at 2pm without any hot-path LLM work.

### 6. Salience is decided in sleep, and prediction error gates every rewrite.

**The hot path runs no LLM and makes no importance judgment.** On turn end (detached, never blocking the turn): append the `Episode`, embed it (~3 ms), compute **store-relative distinctiveness** (one k-NN against the store — the best-evidenced salience signal available, and the one that answers "do we already know this?"), and stage it. That is the entire per-turn cost.

One deliberate exception: an explicit **`remember` tool** the agent calls when the owner says "remember this" — an immediate, `STATED`-provenance memory. The owner's explicit lever is not a heuristic and should not wait for sleep.

Consolidation then routes each candidate by **prediction error against what memory already believed**:

| candidate | PE | path | cost |
|---|---|---|---|
| fits the store, nothing new | none | `confirmations += 1`. **The rewriter is never invoked.** | free |
| fits, adds a detail | small | in-place gist update, sources appended | one cheap call |
| new | n/a | ADD | one cheap call |
| **conflicts with a confirmed memory** | large | **the episode is already committed verbatim; mark the memory `CONTESTED`; defer.** Never overwrite inline. | deferred |

The destabilization threshold **scales with `confirmations` and age** — a well-confirmed belief demands a bigger surprise before it may be rewritten. One anomaly must not be able to rewrite a settled belief. This No-PE → no-rewrite gate is simultaneously the biologically correct rule and the largest compute saving on the path: it kills reconcile-every-turn.

### 7. Sleep: one consolidation engine, two cadences, and it may never strengthen a memory.

Sleep is **idle-opportunistic** (owner's call): it starts when the Mac has been idle, prefers the overnight window, runs the **full agent model** through the arbiter, and **yields instantly** when the owner returns. Instant yield is achieved by acquiring the GPU lease **per work item**, not for the session — so a chat turn never waits longer than one item — plus cooperative cancellation between items. Missed nights are caught up at the next idle. Justification for the nightly cadence is **the idle GPU**, not biology.

Two cadences, one engine:

- **Micro-consolidation** — short idle (~5 min), small budget: extraction only. This is the daytime, outcome-triggered pass the research demands; without it, credit assignment misses the moment the outcome lands.
- **Full sleep** — long idle / overnight: the whole work list.

The work list, in order:

1. **Grade** the day's retrievals (`decisive | used | ignored | harmful`), re-reading the turns. *This is the whole ballgame* — it is the only source of the usefulness signal the lifecycle runs on.
2. **Extract** memories from unprocessed episodes — **cluster first, then distil**. Gist emerges from co-replaying *overlapping* episodes, not from summarizing them one at a time. This is how "I've noticed Bohdan prefers X" is actually made.
3. **Reconcile** each candidate against its k-NN neighbourhood under the PE gate (§6).
4. **Separate** near-duplicates by contrastive rewrite — "say what your neighbours don't". This is the dentate-gyrus analogue: the brain's answer to interference is **orthogonalization at write time, not deletion**. Write-time contextualization is independently the cheapest measured retrieval win available (Anthropic's Contextual Retrieval: top-20 failure 5.7% → 2.9%); the contrastive, near-duplicate-gated form appears to be unshipped anywhere.
5. **Distil patterns** — streaks, conscious switches, dismissal runs. A first-class memory product (#302), not an afterthought.
6. **Lifecycle sweep** — recompute `R`, promote, demote.
7. **Journal** every mutation.

**Consolidation must never increment `S` or `SS`.** A system that reviews its own memories and strengthens them is a self-licking ice cream cone. Only the owner's actual use may strengthen a memory. Replay is of **real stored records with an LLM deriving new records from them** — never of LLM-generated pseudo-episodes, which manufacture exactly the near-miss records that most damage a retrieval store.

Every sleep mutation is journaled and owner-visible; a bad consolidation is inspectable and revertable.

### 8. Storage: SQLite — a deliberate break with the file-based grain.

Every existing store in the app is JSON or JSONL. Memory is not, and the reasons are specific: it needs **FTS5 keyword search** alongside vector search; **transactional multi-table writes** (episode + memory + sources + journal must commit together or not at all); **incremental update without whole-file rewrite** (the existing pattern already costs a 2.7 MB rewrite per turn on the largest conversation — `AgentConversationStore.swift:267-284`); and it must **grow for years**. System SQLite (3.51, FTS5 + JSON1 + WAL) adds **no dependency**. Embeddings are `BLOB` columns; at personal scale (thousands of records, not millions) **brute-force cosine over Accelerate is sub-millisecond and no ANN index is needed** — which also sidesteps the hubness pathology that afflicts ANN indices as they grow.

The database lives **inside the agent's file sandbox** (`PathSandbox.defaultRoot/memory/`), following the Companion heartbeat's precedent, so the owner can always reach it.

### 9. Inspection: browse, delete, contest — no free-text editing.

A Memory window (a suppressed-at-launch `Window` scene, per the Markdown Gallery precedent) shows what the assistant believes and why: the gist, its provenance (`STATED`/`INFERRED`), the source episodes, tier, strengths, and usage. The owner may **delete** anything (the one true-deletion path) and may **contest** a belief, which marks it `CONTESTED` and queues it for reconciliation in the next sleep. There is **no free-text belief editing**: a hand-edited belief has no provenance and no lifecycle, which would quietly falsify every claim this design makes.

### 10. Evaluation: the baseline is built first, and every mechanism must beat it.

The scoreboard exists before the mechanisms, because in the continual-learning record brain-inspired methods keep losing to *just keeping the data* (sleep-replay 48.5% vs plain rehearsal on 0.75% of the data 79.9%; EWC 0.087 vs an undefended MLP's 0.085). The baseline is **"store everything, retrieve well"**: every episode, embedding retrieval, no lifecycle, no consolidation. If the living lifecycle cannot beat it on the owner's own corpus, the lifecycle is decoration.

The published benchmarks are unusable as targets (LoCoMo's answer key is 6.4% broken and its corpus fits in context, so the *no-memory* baseline beats every memory system; scores do not transfer across harnesses; every published judge is a cloud model). So the yardstick is built here, under a **local judge**, against the owner's corpus, and it measures the three things nobody measures:

- **retirement recall-regret** — retire aggressively, then probe later for the rare-but-critical fact that should have survived;
- **promotion-predicts-usefulness** — does a promotion decision beat recency and random at forecasting future hits;
- **the sleep-consolidation differential** — is the store measurably better after a pass, with nothing load-bearing lost.

## As built (2026-07-12)

The design above was written before the code. This section records what actually shipped, where it departs from the design, and what the yardstick measured — so the ADR is read as a record, not a wish.

### What the eval found, in full

The harness (`tesseractTests/MemoryEvalHarness.swift`, +`MemoryEvalTests`, `MemoryBaselineTests`) runs on the owner's real corpus: **65 conversations, 458 records, a 5-day span**. Probes are derived mechanically from the corpus's own structure — no LLM, no hand-written key — in three families (`sessionContinuation`, `referenceBack`, `titleTopic`). Three arms rank the *same* records, differing only in the ranker: baseline (relevance alone), lifecycle (§5's score), and a **null arm that discards the cue entirely and ranks by age**.

- **The lifecycle beats the baseline everywhere** — ΔMRR +0.02 to +0.27, ΔR@5 +0.06 to +0.38, cold-start and held-out.
- **And that number must not be cited as evidence.** The null wins by a mile (MRR 0.68–1.00). Gold sits in the **newest 10% of the pool for 100% of probes in all three families** (mean gold percentile 0.000): the owner's conversations are near-independent one-offs over five days, so every answer is in the turns immediately before its question. This corpus cannot distinguish a memory system from `ORDER BY timestamp DESC` — which is precisely the flaw §10 diagnoses in LoCoMo, now measured in our own yard. **Retrieval scoring is therefore UNPROVEN**, and stays so until the corpus grows real cross-session fact reuse. The harness prints `THE NULL WINS` next to its own win rather than let the number be quoted.
- **Retirement safety is PROVEN, and it is the load-bearing result.** Run §4's policy on a clock two years on: **346 of 458 records retire, and all 97 memories that were ever useful survive — regret 0.000.** The rejected Law of Disuse ("retire the unused", what almost every shipped agent memory does) retires 361 and pushes **100% of the genuinely-needed gold out of the pool — regret 1.000**. The guard in §4 is the difference between those two numbers.
- **Promotion-predicts-usefulness is unmeasurable here**: no memory born before the cutoff is needed after it. Reported as unmeasurable rather than dressed up.
- **The sleep differential** is asserted as invariants, not a score (a score would need the corpus the eval just showed we do not have): episodes are never mutated, replay never strengthens, and no ever-useful memory is retired.

### Where the code departs from the design above

- **§6's four-way PE table is a three-way one.** Reconcile answers `SAME` (confirm, rewriter never invoked), `NEW` (add), or `REPLACES` (supersede). The "small PE ⇒ in-place gist update" row is **not built** — it is a rewrite of a belief's text, which is exactly the serial-reproduction risk §1 exists to prevent, and nothing in v1 needs it. The "conflict ⇒ mark CONTESTED" row is also not built: `CONTESTED` is now reserved for **the owner's veto**, and a model-detected conflict supersedes outright.
- **`CONTESTED` does something.** §9 promised contested beliefs get "queued for reconciliation in the next sleep". Sleep now has that phase (`reexamineContested`), and it is honourable only because of §1: the belief goes back to the immutable episodes it was drawn from, *without* the reading he rejected, and either a corrected claim supersedes it or it goes cold. A re-read that merely restates the rejected claim is dropped — he is the authority on his own life — and the successor inherits none of the vetoed belief's strength or confirmations.
- **One sleep cadence, not two.** §7's micro-consolidation/full-sleep split collapsed into a single idle-triggered run (180 s idle threshold) that does the whole work list and yields instantly on the owner's return. Batches are digested atomically (extract → reconcile → mark consolidated), so a yield loses nothing and re-reading a batch is free under the PE gate. A second cadence buys nothing until the first one is under load.
- **Steps 4 (contrastive separation) and 5 (pattern distillation) of §7's work list are not built.** Both are real, both are deferred: separation needs a store dense enough to have near-duplicates, and pattern distillation needs more than five days of history to have a pattern in it. The shipped work list is grade → re-examine → extract → reconcile → sweep, each journaled.
- **The hot path does no k-NN.** §6 has it computing "store-relative distinctiveness" per turn and staging it. It does not: `MemoryEngine.record` is an insert plus an embedding (~3 ms), and the k-NN happens in sleep where the neighbours are needed anyway. This is more faithful to "the hot path makes no importance judgment", not less.
- **`memories.md` migration is guarded on *storing*, not parsing.** The first implementation archived the file once it parsed the bullets, and an early launch archived the owner's real file while the claims went nowhere. It now archives only after the claims are in the store, and the markdown and corpus seeds are gated independently.

### Verified live, on the owner's own machine — and what that caught

Backfill imported the markdown claims and the corpus; idle detection fired; sleep ran to completion under the GPU lease, yielded instantly on return, and learned true things from his own history. The Memory window, the Companion callback (composed from memory and spoken, 37 s), and the chat injection were all driven end to end against the real store.

**Five bugs, every one of them invisible to the test suite, and two of them fatal to the whole feature.** They are listed because the pattern matters more than the fixes: each one lived in a seam *between* components, and each component's own tests passed.

1. **The model was never shown a single memory.** `sendMessage` wraps the message in a `CoreMessage` before `prepare` runs, so the unwrap in `attachMemory` matched nothing — retrieval ran, the block was built, and it was dropped on the floor between the store and the model, on every turn, silently. Nine tests covered the block's *contents*; none went through `send`. The feature looked alive from every angle except the only one that counts: asking the running app what it remembered. It said it had nothing.
2. **13.5% of the episodic layer was the app's own voice.** A skill fire puts the skill's entire body into the user message, and memory recorded it verbatim as testimony — 28 of 207 episodes were skill instructions and container paths, eight beliefs had been distilled from them, and they were quoted back to the model as "things that were actually said". `MemorySpeech` now strips the wrapper at the single door both capture and backfill go through.
3. **Sleep was eating the store it was building.** Its internal neighbour lookups marked memories `seen`, which is exactly what the third retirement path acts on — so a belief nobody had ever been shown accrued eight sightings in one night and went cold on the strength of them.
4. **A cancelled sleep could clear its successor's handle**, letting two consolidations race over the same episodes.
5. **The batch cap assumed 8 episodes per batch** when batches are per-conversation and the owner averages three — the first night read a fifth of his history and stopped.

`memories.md` was also destroyed once and restored: the migration archived the file when the claims *parsed*, not when they were *stored*. It now archives only after they are in the store, and the two seeds are gated independently.

### The sixth finding: the line itself

The first real morning beat, fired at 09:00 on the owner's machine, said:

> Morning. What's the one hard thing today, the AI agent or that pending task?

Every piece of the machine worked — it recalled, it composed, it spoke — and the *product* still failed, because "the AI agent **or** that pending task" is a hedge. Offering him two things is telling him you know neither, and #302's bar is one **specific** callback. `MemoryCallback`'s own header meanwhile claimed it "verifies the composition is grounded", which was simply untrue: it checked for `PASS`, took the first line, and measured the length.

So the promise is now kept in code. A line clears three gates or the beat falls back to its hardcoded prompt: `clean` (one sentence), `grounded` (it re-uses a distinctive stem from what was actually recalled — the generic line shares nothing but stopwords, the invented one shares nothing at all, and this runs *before* the model so no GPU is spent judging a fabrication), and `critique` (a second pass over the same evidence whose only job is to refuse; anything short of an unambiguous `KEEP` is a `PASS`). The compose prompt forbids the hedge outright, and both passes share one GPU lease.

After the fix, two live test pings, both kept: *"Ти просив адаптувати чат Tesseract під різні екрани — продовжимо?"* and *"Ти просив доопрацювати адаптивний чат агента Tesseract — чи встиг ти завершити це завдання?"* — specific, true, first-person, one thing each. **The language is the model's own choice**, drawn from the language of the recalled material; no rule constrains it, and for a bilingual owner that seems right rather than wrong. Worth watching, not worth legislating yet.

The lesson is the same one the five bugs teach, one level up: every component passed its tests, and the thing the owner would actually *read* was still wrong. The only test that finds this is looking at the sentence.

### The first outside look (2026-07-12, same day)

The system was built and landed autonomously, direct to main. Its first review — four independent passes over the full range, attacking the invariants above — confirmed the enforced ones (episodes append-only; SAME never reaches the rewriter; per-generation GPU lease with cancellation between every item; injection rides the user message and the system prompt is untouched) and found six real defects, all fixed the same day:

1. **The grading pipeline was severed in production.** `attachMemory` never passed the turn's episode id, `retrieve` only logs `if let episodeID` — so the retrieval log stayed empty forever, sleep graded nothing, and `storageStrength`/`usefulUseCount` could never move. Meanwhile every surfacing still counted as *seen*, which is the second retirement path's trigger: the lifecycle, unfed, inverts into a retire-everything machine — bug #1's class ("looked alive from every angle"), one seam over. The eval never noticed because its harness fabricates retrieval events directly. Fixed by giving a chat episode **the user message's own id** (the grain the backfill already used): the log can point at the turn before its episode is written, and re-capture becomes a no-op.
2. **Retirement lasted one night.** The sweep feeds every live memory through `sweepTier`; `shouldRetireToCold` refuses what is already cold, so cold rows fell through to the hot/warm re-derivation and came back — counted and journaled as *promotions*. Cold is now sticky; the one way back on the sweep's authority is a graded useful use.
3. **A vetoed belief could be re-tried and re-credentialed.** Contested-and-retired beliefs re-entered `reexamineContested` every night (a nightly LLM bill and a fresh chance to mint a successor), and reconcile offered contested beliefs as neighbours — a lookalike claim could *confirm* the belief the owner rejected, or supersede it with strength inherited. Cold contested beliefs are now final; reconcile sees live neighbours only.
4. **A garbled reconcile verdict confirmed a random neighbour.** The unparseable fallback was `.same(0)` under a comment claiming it "dropped the claim". There is now a real `drop` verdict: unvouched claims touch nothing.
5. **Read-modify-write across `await`s.** Every counter bump was a full-row upsert of a possibly-stale snapshot: a late seen-mark could roll back a grade's strength bump (silently breaking SS monotonicity — the one invariant with no SQL-level guard), and a contest clicked over a stale window could resurrect a superseded belief. Supersession itself was three transactions with a GPU call between — a crash in the gap loses the belief. The store now owns every read-modify-write: targeted UPDATEs (`markSeen`, `confirm`, `setTier`, guarded `contest`) and one-transaction compounds (`supersede`, `grade`), each reading the fresh row inside the actor call.
6. **"Yields instantly" was a fifteen-second poll.** `IdleMonitor`'s header promised HID return fires synchronously; there is no such event without Input Monitoring entitlements. The poll now tightens to one second while idle, and the header says what is actually true.

Smaller, same review: the Memory window's tier list sorted ascending and led with the cold tail (the exact inversion class the last pre-review commit fixed, one file over); useful grades now get the journal line §7 promised ("journal every mutation" — strengthening was the one mutation with no line); the adjudicate prompt now shows each neighbour's confirmation count, which is §6's destabilization threshold in prompt form — the numeric gate remains unbuilt, deliberately: supersession is non-destructive and owner-revertable, and a threshold constant would be tuned blind; deleted memories take their retrieval events with them; a re-captured episode no longer double-indexes FTS; the backfill reads the corpus off the main thread and archives `memories.md` only when *every* claim is stored, with re-runs deduplicated.

Two departures the review surfaced are kept and documented rather than fixed: **later turns do inject** — but only memories the conversation has not yet been told (`injectedMemoryIDs` dedupes; the block rides the newest user message, so the cached prefix is untouched) — §5's "later turns do not re-inject" holds as *no repeats*, not *first turn only*; and the agentic search tool shipped as **`recall`** (with `remember` beside it), not §5's `memory_search`.

The `memories.md` skill and seed file are gone from the personal-assistant package; its system prompt now teaches the two tools and the injected block. The old regime has no remaining callers.

### The EOS attractor (2026-07-12, the first live recall — #332)

The owner asked the agent to recall the Companion feature and got cats, Ghostty, and Celsius back — while the memory sat live and hot in the store and the injection path was surfacing it. The diagnosis was proven from the stored vectors alone before a single line changed:

**Every text shorter than the embedder's 16-token padding floor pooled a padding token's hidden state.** Three faults compounding: the embedder padded every batch to ≥16 tokens with EOS; the vendored `Qwen3Model` accepted an attention mask and silently ignored it; pooling ran maskless with the model's `.last` strategy, so it always read position `maxLength−1` — for a short text, an EOS pad at the end of a run of EOS pads, whose hidden state converges to a content-free direction. Measured on the owner's store: memories under 40 chars sat at mean pairwise cosine **0.959** to each other regardless of content; the honestly-embedded 137-char Companion memory sat at 0.19 to that cluster. A ~5-token query lands *on* the attractor, matches every short memory at ~0.9, and the true answer's ceiling is 0.25 — the most a perfect keyword match can contribute under the 0.75/0.25 fusion. The ranking the owner screenshotted is exactly what the code computed.

Four fixes, each with a gate that fails on the old code:

1. **The embed path is reference-faithful now** (`encodeAndPool`): each sequence ends with the EOS the tokenizer's own post-processor appends, the pad floor is gone, and pooling receives a *length-based* mask — never a token-value one, which would erase the very EOS that last-token pooling exists to read. Queries carry the model card's instruct prefix; documents never do. The eval confirmed the card's claim: dropping the prefix costs ~1–2pp MRR. The post-pooling layer-norm (not in the HF reference) measured as a no-op and is kept; the fusion (linear vs RRF vs max) and the 0.2 floor measured as ties — noteworthy because gold and noise relevances *overlap* (gold p50 0.509, noise p90 0.502): no score floor can separate them in this space, ranking does all the work, and the floor survives only as a degenerate-input guard.
2. **Vectors are artifacts of a scheme, and the store knows which one it holds.** An `embedding_scheme` stamp (schema v2 `meta` table); on prewarm, a mismatch wipes every vector — and the cue affinities learned from ranking them — and re-embeds everything, stamping only on completion so an interrupted pass re-runs. ~2 s for the owner's 444 records. This is also the repair path for vectors that never got written.
3. **`recall` now searches what its description always claimed.** The tool read beliefs only; a fact told in the morning exists only as an episode until sleep distills it, so the tool had a same-day blind spot (the §Consequences line "same-day recall rides the episodic layer and the `recall` tool" was aspiration, not fact). `searchEverything` returns beliefs plus dated, quoted episodes.
4. **The vendored `Qwen3Model` honors the mask it is handed** (folded into the causal mask as a key-padding term), upstreamed to ml-explore/mlx-swift-lm. For this app's right-padded causal batches the fix changes nothing — the pooling mask was the load-bearing repair — but a left-padded batch (the Hugging Face default for this model family) is corrupted without it.

The yardstick grew the probe family whose absence hid all of this: **belief recall** — `sourceEcho` (a source episode must find its distillates) and `rareTerm` (a memory's rarest terms, as a short query, must find it — the shape that failed live), mechanical like the rest, run against a copy of the owner's real store, with a cue-blind null arm. After the fix: rareTerm R@1 **0.949**, MRR 0.996 (null: 0.022); sourceEcho MRR 0.348 (null: 0.015). §10's "retrieval scoring UNPROVEN" is now half-resolved: *proven* for targeted recall, and honestly hard for episode-to-distillate echo. Permanent gates encode "we shipped the winner": if any judged variant ever dominates the shipped configuration, or the null ever catches it, the suite fails. An embedder-sanity suite (unrelated short texts must not be near-identical; a query must find the memory it names; a vector must not depend on its batch-mates) fails loudly on the old code, and a tool-truth smoke drives the registered `recall` tool end to end — everything below the model. The one hop no test covers is the model *choosing* to call the tool: covering it would mean a second resident copy of the agent model inside the test host, which is deliberately not done.

The lesson this time is not a seam — it is **calibration without ground truth**. The embedder produced unit-norm, plausibly-distributed vectors; retrieval returned confidently-ranked results; nothing crashed, nothing was empty, and cosine 0.96 between "I love cats" and "He is from Europe" sat in the store for the system's whole life. No component *could* notice, because every number was locally reasonable. The only test that finds this class is one that asserts a semantic invariant of the space itself — unrelated things must not be identical — which no amount of plumbing coverage implies.

### The voice that meant its opposite, and the doors that doubled (2026-07-12, #333)

The owner noticed the agent remembering in *his* voice — `remember` storing "I like to eat apples in the morning" — when the map's decision was the opposite: these are the **agent's** memories. The design had not changed; one write door had garbled it. This ADR names the semantic layer "first-person", meaning the *assistant's* first person ("I've noticed Bohdan starts with the hardest task"). Sleep's prompt renders that correctly — "Write about him in the third person" — and never slipped once: **117 of 117** consolidation-born beliefs on the live store were owner-third-person. The `remember` tool description carried the transplanted phrase *"in the first person"* next to a third-person example, and the model, reading "first person" from where it stands, resolved "I" to the owner: **17 of 27** remember-born beliefs landed in his voice. Same instruction, two write doors, opposite referents — and at read time the injection block ("My own memories of the person I'm talking to") makes "I use the Ghostty terminal emulator" an assertion about the *agent*.

The rule, sharpened by the owner and confirmed against the field (Letta's persona/human block split; ChatGPT's and Mem0's subject-explicit user facts): **the convention is about referents, not pronouns**. Every pronoun in a stored belief must resolve with no conversational context — "he" is always the owner, "I" is always the assistant, and "I" is not banned: agent self-memories are a legitimate category. Directives take sleep's existing canon, "He wants me to X", because a remember-born directive is always owner-ordained and that form keeps the ordainer visible. The fix is prompt-only — no mechanical guard, precisely because "I" legitimately denotes the agent — plus a contract test that pins "third person" into the tool description and keeps "first person" out. The 17 live wrong-voice beliefs were superseded offline by voice-only rewrites through the store's own grammar (supersession + journal, refs copied, scheme stamp cleared so reconcile re-embedded everything).

The same investigation closed two capture defects. **Fresh-id duplicates**: until the first outside look's fix, chat capture passed no episode id, so the `UUID()` default defeated the `INSERT OR IGNORE` dedup — and capture fires at every `turnEnd`, i.e. once per tool round, so one morning message became eleven episodes, seven of them carrying the "\n\n" fragments between tool calls as their "reply". Post-fix capture is verified clean; the 35 residual copies were pruned. **The double door**: dictating *into Tesseract's own composer* wrote the utterance twice — once at dictation commit, once at send. The rule is now **one door per testimony**: the chat door owns words sent to the agent (it attaches the reply; strictly more context), and the dictation door skips when the frontmost app at commit is Tesseract itself. Its 17 residual copies were pruned with it; consolidation had been reading each copy as an independent attestation.

The lesson is a naming one, and it belongs to the domain model: a term whose meaning depends on where the reader stands ("first person" — whose?) will eventually be read from the other side, by the one reader who executes instructions literally. The glossary now carries the convention as **Agent Voice**, defined by referents, with the deictic trap named.

### The correction had no door (2026-07-12, #333 round two)

First, a correction to the paragraph above, owed to the journal: the 17-of-27 figure was real, but the prompt caused only two of them. Fifteen of the seventeen owner-voice beliefs were born in a single second (11:50:33) with no conversation anywhere near — the `memories.md` retirement, whose backfill feeds every line of the old file through `remember` verbatim (`MemoryBackfill.swift`). The old file was written in the owner's voice, so the import carried his voice wholesale, stamped STATED, journalled "The owner asked me to remember this." The live inversions the phrase caused were two-for-two that afternoon (apples, cats) — still damning for the phrase, but the bulk was the old era's residue arriving through a door with no gate. Worth recording precisely because the first version of this section over-attributed: the journal knew better.

Then the owner found the sharper thing. On July 7 he wrote *"Very nice, Pelican. You are an artist."* — praising a pelican SVG the model had just drawn — and the live model, with the **full conversation in front of it**, read the vocative comma as a christening and answered "I'm honored to be called Pelican — maybe that's my new name. I'll add it to my memory." It did: `memories.md`, then the import laundered it into a STATED belief, and sleep independently minted the INFERRED event from the same episodes — episodes whose attached reply *endorsed* the misreading. So "give the extractor more context" was never the fix; the context was there, poisoned at the moment of utterance. What actually failed came next: when the owner corrected it, the agent's only write door was `remember` — the system prompt even taught "when they correct a fact, remember the corrected version" — so it inserted a negation *beside* the lie. Two affirmations and their negation, all live, and nothing in the store able to notice, because reconcile compares a new claim against neighbours but never neighbours against each other, and its `REPLACES <n>` verdict could retire at most one belief per claim anyway. Meanwhile the machinery built for exactly this event — the owner's veto, contest → re-read → supersede-or-retire — was reachable only from the Memory window.

Three fixes, one per failure. **The `contest` tool** gives the veto its conversational door: `recall` lines now lead with an eight-hex handle, `contest(memory, reason)` flips that belief to contested (a status change — no edit, no delete, same store guard as the window), and the rejection reason rides the journal into sleep's re-read, which now shows it beside the source episodes — necessary, because the source episodes alone are exactly the evidence that produced the wrong belief; his rejection is the only thing pointing the other way. **Multi-`REPLACES`**: the reconcile verdict takes a list ("REPLACES 1, 3"); the first target is superseded normally, every further target is retired in favour of the same successor, so one correction can falsify a stated belief and its inferred sibling in the same pass. **The reply that never arrived**: episodes are written at the turn's first `turnEnd`, and a turn that opens with a tool call has emitted only whitespace by then — so the stored "reply" was `"\n\n"` and `INSERT OR IGNORE` dropped every fuller re-capture. Capture now trims, and a re-capture of the same turn updates `meta.reply` in place, newest non-empty wins; the testimony itself never moves. The grading judge and the extractor both read that field.

The second repair pass on the live store followed the same offline grammar as the first, and owed two of its four parts to the first pass's own blind spots: the two Pelican affirmations contested (with the rejection note, so the next sleep disposes of them properly), the two voice rows the `LIKE 'I %'` filter missed superseded ("My name is Bohdan…", "The user did not give…"), nine duplicate live pairs folded — the first migration inserted its rewrites without meeting the reconcile gate, so rewrites collided with beliefs the agent had already remembered correctly — and thirty-five more double-door dictation copies pruned, invisible to the first pass because the chat composer smart-quotes what dictation typed and the owner dictates in fragments that the send concatenates; equality matching found neither, normalised-substring matching found both.

## Consequences

**Accepted costs.**

- **A third co-resident MLX model** (~335 MB, the embedder) joins the 0.8B proofreader outside the arbiter. ADR-0034 established this is safe; the process-global `MLX.Memory.cacheLimit` remains the agent `LLMActor`'s knob and memory does not touch it.
- **Memories appear after consolidation, not instantly.** Same-day recall rides the episodic layer and the `recall` tool; the explicit `remember` tool covers the owner's deliberate asks. This is the price of moving the salience decision to where the information actually exists.
- **Sleep needs idle detection, which the app does not have.** It is built here (`CGEventSource.secondsSinceLastEventType`, plus screen-lock/wake notifications) — one new Platform adapter.
- **Consolidation runs only while the app is running.** `launchAtLogin` defaults off and the Companion heartbeat already accepts this; missed nights are caught up at the next idle rather than lost.
- **Dictation content becomes memory** (owner's explicit call — "dictated content is your life too"). This is in tension with `TextInjector`'s treatment of dictated text as private (transient/concealed pasteboard). The tension is resolved in the owner's favour, behind a setting, and named here so it is not discovered later as an accident.
- **SQLite breaks the repo's file-based grain** (§8). Deliberate, argued, and confined to this subsystem.
- **`memories.md` is replaced.** The personal-assistant package's memory skill and its prose protocol are superseded by the engine; the file is migrated into the store and the skill retired.

**Follow-ups deliberately deferred.**

- **Learning the retention model from our own logs.** FSRS-6's fitted constants are a *cold-start prior on the shape*, not scripture — in FSRS's own benchmark a zero-parameter moving average beats FSRS-6 (log loss 0.337 vs 0.346) and a small learned sequence model crushes both (0.277). Once ~10⁴ graded retrieval events exist, re-fit on our own data; the optimizer is a trivial nightly job on this hardware.
- **Difficulty (`D`)** is stored but not fitted in v1 — with a 4-level grade from a noisy local judge it may simply be unidentifiable, and two numbers (`S`, `SS`) may be the honest schema.
- **Per-cue decay.** Interference theory says competition lives at the *cue*, not the memory. Conditioning need-probability on cue-cluster density is the most scientifically defensible refinement available, and it changes the schema.
- **Cross-system benchmark comparison** — ruled out of map #314's scope by the owner; the in-map bar is the owner-corpus baseline.
