---
status: accepted
---

# Recovery-cost eviction across cache tiers — Marconi adapted on-device

This records the redesign of prefix-cache admission/eviction decided in
the 2026-06-12 grilling. See `CONTEXT.md` → **Pressure-Reactive Budget**,
**Budget Floor**, **Snapshot Demotion**, **Recovery Cost**,
**Survival Gate**, **Eviction Configuration**, **AlphaTuner inversion**.

The starting point: the eviction half of Marconi (arXiv 2411.19379) was
implemented faithfully — `utility = norm(recency) + α·norm(FLOPs/bytes)`
plus an online grid-search AlphaTuner — and was structurally dead on a
Mac. α defaulted to 0 with an LRU fast path; the tuner's trigger (first
RAM eviction) never fired under the auto-sized half-of-leftover-RAM
budget; its one-shot lifecycle died with every process; grid-search ties
re-picked 0; and the SSD tier was deliberately α-blind, deferred "until
production traces show α > 0" — evidence the RAM tier could structurally
never produce. The paper's other half (judicious admission) was never
implemented at all. Marconi's assumptions — multi-tenant traffic dense
enough to tune per-process, a single cache tier where eviction is
terminal, a fixed contended budget — are all false here: single user,
short-lived processes, two tiers (RAM + SSD), and a budget whose real
constraint is the rest of the machine on unified memory.

The decisions, as one cluster:

- **Objective: turn latency, not hit rate.** Minimize P50/P95 TTFT on
  real agent traces. FLOPs-saved is the score's proxy for TTFT (on one
  GPU, prefill FLOPs avoided ≈ TTFT saved); memory footprint is **never
  a score term** — it is the constraint.

- **The constraint is a band, not a constant.** The RAM budget keeps its
  auto-sized value as a *ceiling*; OS memory-pressure events push a
  *current* budget down toward a floor and hysteresis regrows it
  (fast down, slow up). The **Budget Floor** is content-defined and
  deliberately dumb: `.system` chains plus the single
  most-recently-extended leaf — a last-resort survival set, never the
  protection mechanism. Per-partition floors and workload heuristics
  (e.g. "protect the tallest chain") were rejected: protecting the main
  agent's tall leaf from subagent churn is the eviction score's job, and
  a safety net with policy in it is a safety net with bugs.

- **Demote, don't drop.** Any RAM-tier shrink — pressure or ordinary
  evict-to-fit — first persists the victim to SSD (if not already
  backed) and then drops the RAM body; terminal drop is the fallback
  when backing is unavailable. Invariant: a demotion write never
  refreshes the ledger's `lastAccessAt` — demoted bodies are the least
  valuable, and refreshing them would invert the SSD tier's recency
  signal on every pressure event. Only hydrations and extensions
  refresh.

- **F is recovery cost, tier-aware, in seconds.** The score's F term is
  what the next hit *pays* if the body leaves the tier — not the FLOPs
  the snapshot embodies (Marconi's single-tier reading, which overstates
  the loss of an SSD-backed body by the ratio of re-prefill to
  hydration, ~10–50×). For an SSD-backed RAM body: hydration cost. For
  terminal loss (unbacked RAM drop, the SSD cut): re-prefill FLOPs.
  Both denominated in seconds via rolling measured device estimates
  (prefill FLOPs/s, hydration bytes/s) captured from real operations —
  never guessed constants. Deliberate consequence: among backed RAM
  bodies, recovery cost per byte is a constant, the density term goes
  flat, and demotion ordering degenerates to recency — **LRU in the RAM
  tier is correct by design**, and the α=0 fast path survives as its
  implementation. The α-blend only changes outcomes at terminal-loss
  sites.

- **The α-blend moves to the SSD cut.** The ledger cut adopts
  terminal-loss utility (re-prefill seconds over bytes-on-disk, chain
  totals) under the **one shared α** — separate per-site αs were
  rejected as doubling the tuning surface for a second-order gain that
  per-candidate-set normalization mostly absorbs. `.system` chains stay
  hard-protected (a scoring fluke that cuts the system prefix is a
  catastrophic tail with no upside). This breaks the circular deferral:
  the SSD tier no longer waits for RAM-tier α evidence.

- **Admission gating is the cut, run early.** The **Survival Gate**:
  an SSD write happens only if the incoming chain's terminal-loss
  utility would survive the eviction its own admission triggers —
  otherwise a demotion terminal-drops and a leaf stays RAM-only with
  supersession preserve. This *derives* Marconi's judicious-admission
  half from the eviction score instead of bolting on a second policy.
  One asymmetry: end-of-turn leaf admissions bypass the gate — the
  just-finished leaf is the highest-reuse object in the system and its
  extension write is suffix-sized (ADR-0010). The gate bites only under
  budget contention; an unfilled ledger admits everything, keeping
  cold-start behavior unchanged.

- **Tuning is persisted and continuous, and built last.** The one-shot
  first-RAM-eviction phase machine is retired. The tuner retunes a
  sliding window on terminal-eviction pressure, damps each update
  (never jumps), and persists the result per model fingerprint — on a
  single-user Mac, persistence does the work traffic volume does in the
  cloud. The shipped default α comes from offline replay of recorded
  traces (the `http-completions/` request recordings), not 0. The
  replay sandbox simulates the *ledger* (chain totals, hits, cuts) —
  simpler than the current full-radix-cache sandbox, because post-
  reframe α only matters there.

- **Sequencing and the ablation bar.** Ship order: (1) telemetry +
  trace-replay harness (baseline first, no behavior change), (2)
  pressure-reactive budget + demotion + recovery-cost reframe, (3) SSD
  cut scoring + survival gate at the seeded α, (4) the rebuilt tuner —
  only if the phase-3 ablation earns it. α's contribution is judged
  against LRU-with-demotion, not against today's pure LRU; demotion
  will deliver most of the TTFT win and the tuner must not take credit
  for it. If the ablation is noise, the honest outcome is a frozen
  seeded α and no online tuner.

Considered and rejected:

- *Literal Marconi port, re-aimed* (keep embodied-FLOPs F, retrigger
  the tuner off the SSD cut). Keeps a wrong cost model for a two-tier
  cache and every cold-start pathology.
- *Kill the tuner, ship a constant.* Honest, but α is workload-
  dependent (Marconi's own finding) and the persisted-continuous loop
  costs little once the replay harness exists; the sequencing gate
  preserves this as the fallback outcome.
- *Memory in the objective* (score bytes as a co-objective). Rejected:
  keeps the score one-dimensional and tunable; "memory-efficient" falls
  out of a tighter budget, not a different formula.

Consequences: the manifest must carry what terminal-loss scoring needs
(token offsets / chain FLOP inputs — today it stores only
`checkpointType` and `lastAccessAt`); a persisted per-fingerprint α
sidecar appears; `EvictionPolicy`'s α=0 fast path is no longer a
pathology but the designed RAM-tier common case; per-completion
telemetry (TTFT, restored offset, hit tokens, hydration ms, terminal
evictions) becomes the tuner's window food and the UI's evidence; and
the speculative-canonical-prefill tuner blind spot (ADR-0009) closes
naturally — its leaf admissions reach the ledger, which is where the
window now listens.
