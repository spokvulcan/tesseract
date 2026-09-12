# Leaf Lease review follow-up

Follow-up to [PR #498](https://github.com/spokvulcan/tesseract/pull/498) and its initial implementation, `e677c62f724b5eb06aacaf1d9af04b4c7764f73e`.

The supplied review identified documentation and telemetry gaps. Verification also reproduced a return bug: lookup omitted a pending-only destination, letting growth check-in attach a body over its pending ref. Exact-path validation now preserves every occupied destination, including pending refs and chain-prefix restore points. An empty structural destination's access gate is reserved before release, so a tombstoned writer still reading its former body also blocks return. Both cases leave the source lease, bytes and topology unchanged; returning after the reader finishes succeeds. Membership was already checked before insertion; the final release guard now also precedes topology mutation.

Mandatory admission cannot read or replace an actively owned body. ADR-0019 now states the ordering explicitly: check in or rewind, then admit the finished leaf. `StoreDiagnostics.leaseRefusals` reports rejected entries' lease IDs so a caller can retry. A real SSD test proves the retry persists with a one-byte pending cap, preserving the mandatory-write bypass. Production checkout and its admission ordering remain #480.

ADR-0019 also distinguishes refusing **move checkout** from falling back to copy restore. That preserves ADR-0064's eligibility rule. The tree's writer-exclusion gate is a separate safety boundary; production eligibility must still reject a pending full payload for move checkout.

Refused acquisition and return events now carry typed reasons and actual lease identities. A contender is correlated to the active request; a refusal without a lease does not invent an ID. The whole-tree eviction refusal sweep was removed because floor members were never candidates. Membership checks follow parent links instead of traversing every snapshot. Event payloads have typed fields and fixed names; body-access predicates use a typed operation and the name `blocks`.

Flush waiters suppress their normal re-pump only when a lease blocked the drain. Ordinary forced flush retains its previous behavior. A single-pass set tracks blocked bases and dependent suffixes instead of scanning the queue for each candidate. The existing 500 ms recheck remains the bounded wake delay after return, avoiding writer callbacks or retained writer references in the scalar access gate. Leased ancestors remain excluded from condemnation because the supersession walk cannot replace their bodies; the two policies must agree.

## Validation

- **890 passed, two existing skips, zero failures** across 69 focused suites.
- **One passed** isolated ownership test: 24 small real-cache success/cancel/error simulations, checking no extra body copy, accounting and release of cache objects/arrays while retired tokens remain alive.
- Strict formatting and SwiftLint passed. SwiftLint reports structural-size warnings; details and source hashes are in [validation.json](validation.json).
- Independent spec review found no remaining issues. Standards review found one stale RAM-clear comment, subsequently corrected; no remaining implementation violations or actionable design smells were reported.

The [original memory evidence](../2026-09-12/README.md) remains unchanged and describes its original revision. Its source hashes were verified against `e677c62f`. This follow-up records test outcomes rather than replacing those measurements or claiming a process-footprint improvement. No large-model run or production checkout was performed.
