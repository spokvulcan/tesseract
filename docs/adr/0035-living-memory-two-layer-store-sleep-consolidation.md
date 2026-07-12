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

## Consequences

**Accepted costs.**

- **A third co-resident MLX model** (~335 MB, the embedder) joins the 0.8B proofreader outside the arbiter. ADR-0034 established this is safe; the process-global `MLX.Memory.cacheLimit` remains the agent `LLMActor`'s knob and memory does not touch it.
- **Memories appear after consolidation, not instantly.** Same-day recall rides the episodic layer and the `memory_search` tool; the explicit `remember` tool covers the owner's deliberate asks. This is the price of moving the salience decision to where the information actually exists.
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
