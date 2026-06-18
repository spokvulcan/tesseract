---
status: accepted
---

# Recovery-cost eviction across cache tiers — Marconi adapted on-device

The eviction half of Marconi (arXiv 2411.19379) was implemented faithfully —
`utility = norm(recency) + α·norm(FLOPs/bytes)` plus an online grid-search
AlphaTuner — and was structurally dead on a Mac. Marconi's premises are all
false here: it assumes multi-tenant traffic dense enough to tune per-process, a
single cache tier where eviction is terminal, and a fixed contended budget. We
have one user, short-lived processes, two tiers (RAM + SSD), and a budget whose
real constraint is the rest of the machine on unified memory. Decided in the
2026-06-12 grilling; see `CONTEXT.md` for the named concepts.

## Decision

- **Objective: turn latency, not hit rate.** Minimize P50/P95 TTFT. FLOPs-saved
  proxies TTFT (on one GPU, prefill FLOPs avoided ≈ TTFT saved). Memory
  footprint is never a score term — it is the constraint.

- **The constraint is a band, not a constant.** The auto-sized RAM budget
  becomes a *ceiling*; OS memory-pressure pushes a *current* budget down toward
  a floor, hysteresis regrows it (fast down, slow up). The **Budget Floor** is
  content-defined and deliberately dumb — `.system` chains plus the single
  most-recently-extended leaf — a last-resort survival set, never the protection
  mechanism. Per-partition floors and workload heuristics were rejected:
  protecting a tall leaf from subagent churn is the eviction score's job, and a
  safety net with policy in it has bugs.

- **Demote, don't drop.** Any RAM-tier shrink first persists the victim to SSD
  (if not already backed), then drops the RAM body; terminal drop is the
  fallback. Invariant: a demotion write never refreshes the ledger's
  `lastAccessAt` — demoted bodies are the least valuable, and refreshing them
  would invert the SSD tier's recency signal on every pressure event. Only
  hydrations and extensions refresh.

- **F is recovery cost, tier-aware, in seconds** — what the next hit *pays* if
  the body leaves the tier, not the FLOPs the snapshot embodies (Marconi's
  single-tier reading overstates an SSD-backed body's loss by the re-prefill /
  hydration ratio, ~10–50×). SSD-backed RAM body: hydration cost. Terminal loss
  (unbacked RAM drop, the SSD cut): re-prefill FLOPs. Both denominated in
  seconds via rolling measured device estimates (prefill FLOPs/s, hydration
  bytes/s) — never guessed constants. Consequence: among backed RAM bodies
  recovery-cost-per-byte is constant, the density term goes flat, and demotion
  degenerates to recency — **LRU in the RAM tier is correct by design**, and the
  α=0 fast path is its implementation. The α-blend only changes outcomes at
  terminal-loss sites.

- **The α-blend moves to the SSD cut.** The ledger cut adopts terminal-loss
  utility (re-prefill seconds over bytes-on-disk, chain totals) under **one
  shared α** — per-site αs were rejected as doubling the tuning surface for a
  gain per-candidate-set normalization mostly absorbs. `.system` chains stay
  hard-protected. This breaks the circular deferral that kept the SSD tier
  waiting for RAM-tier α evidence the RAM tier could never produce.

- **Admission gating is the cut, run early.** The **Survival Gate**: an SSD
  write happens only if the incoming chain's terminal-loss utility would survive
  the eviction its own admission triggers; otherwise a demotion terminal-drops
  and a leaf stays RAM-only with supersession preserve. This *derives* Marconi's
  judicious-admission half from the eviction score instead of bolting on a
  second policy. End-of-turn leaf admissions bypass the gate — the just-finished
  leaf is the highest-reuse object and its extension write is suffix-sized
  (ADR-0010). The gate bites only under contention; an unfilled ledger admits
  everything, so cold-start is unchanged.

- **Tuner: seed offline, build last.** The shipped default α comes from offline
  replay of recorded traces (the `http-completions/` recordings), not 0. The
  online tuner ships only if the phase-3 ablation earns it — see Sequencing.
  Reframing the tuning signal onto the ledger also closes ADR-0009's
  speculative-canonical-prefill blind spot: those leaf admissions reach the
  ledger, which is where any tuning window now listens.

## Sequencing

Ship order: (1) telemetry + trace-replay harness (baseline, no behavior change),
(2) pressure-reactive budget + demotion + recovery-cost reframe, (3) SSD cut
scoring + survival gate at the seeded α, (4) the tuner — only if the phase-3
ablation earns it. α's contribution is judged against LRU-*with-demotion*, not
today's pure LRU; demotion delivers most of the TTFT win and the tuner must not
take credit for it. If the ablation is noise, the honest outcome is a frozen
seeded α and no online tuner.

## Considered and rejected

- *Literal Marconi port, re-aimed* (keep embodied-FLOPs F, retrigger the tuner
  off the SSD cut). Keeps a wrong cost model for a two-tier cache.
- *Kill the tuner, ship a constant.* Honest, but α is workload-dependent
  (Marconi's own finding); the sequencing gate preserves this as the fallback.
- *Memory in the objective.* Rejected — keeps the score one-dimensional;
  "memory-efficient" falls out of a tighter budget, not a different formula.

## Status note (drift, 2026-06-18)

The shipped `AlphaTuner` is **not** the persisted/continuous design this ADR's
grilling sketched. It still defaults α to 0.0 (LRU) and tunes **once per
process** via the three-phase machine `waitingForFirstEviction → bootstrapping →
tuned`; there is no per-fingerprint persistence or sidecar (the proposed
"persisted α sidecar" consequence never shipped). Evidence:
`AlphaTuner.swift` ("Tunes once per process… Continuous retuning is out of
scope."), `EvictionPolicy.swift` (`alpha: Double = 0.0`). The redesign's other
decisions (band, demotion, recovery-cost F, SSD α-blend, survival gate) are
implemented as written.
