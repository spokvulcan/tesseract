# PR #503 external-review verification — 2026-09-12

Follow-up to implementation commit `46c59e6b` for #480. Each claim was traced
against the current production adapters, cache state, tree accounting and
lifecycle code. Standards and spec reviews ran independently. Large-model
validation was not authorized or attempted in this follow-up.

## Findings and disposition

| External point | Verified conclusion | Result |
| --- | --- | --- |
| 1. Unload can reject cleanup and orphan a lease | Not a current production failure: the provider strongly retains its container, whose entry rethrows; unload waits for in-flight starts/completions. An independently throwing future adapter would weaken this guarantee. Permanent retention of the request cache was not established either. | Encode the existing entry guarantee as `async rethrows`; remove the swallow-and-log cleanup path. Keep `completeRequest` unable to end leases. Warm-prefill drain/cancel/resend verifies lease release and reuse. |
| 2. Direct fallback rewinds then tries to capture an empty cache | Confirmed with a zero-output warm turn through Server Completion: the final event lacked rewind source and logged `no-reusable-cache-state`. Merely skipping rewind could mislabel unproven current state with a longer canonical path. | Preserve the original leaf and conclude the direct structural fallback as rewind with its original boundary reason. Resend reuses the original offset and produces the expected output. |
| 3. Immutable leaf copies appear as checkpoints | Confirmed for the representation used by copied captures and SSD hydration. No claim about its production frequency was verified. | Add `immutableBody`; structural checkpoint/branch/chain-prefix cases retain `checkpoint` precedence. |
| 4. Trim is inside precondition | Confirmed under `-Ounchecked`; current Release uses `-O`, so this was a conditional hazard. | Execute trim before checking its result. A small compiler reproduction reports trimCalls=1 under `-O`, 0 under unchecked before the change, and 1 under unchecked after the effect is split out. |
| 5. Process-global allocation assertions can race | Confirmed. Serializing this one-test suite does not exclude allocations or releases in other suites. | Gate both allocator thresholds behind an explicit isolated-run environment flag. Physical identity, exact state and owner-release assertions still run normally. |
| 6. Maximum-advance guard is dead | Qualified: plain supported `KVCacheSimple` currently always permits the advance, but the capability call is required and runs. | Retain the guard and clarify its conservative role. Rotating/sliding-window caches remain excluded. |

The review nits were also checked. The stale traversal comment is corrected;
text-only identity eligibility and recurrent class naming use one shared
spelling; refused check-in recovery fails its invariant instead of silently
creating an empty owner. A new regression demonstrated that a zero-layer body
could be captured and admitted as a reusable leaf; both capture forms and the
server's structured-leaf admission now reject it. Pure topology/accounting
fixtures remain independent of the server's physical-cache admission guard.

The recurrent metadata correction was already part of the initial PR and
also fixes the pre-existing copy path: lengths are restored and an empty
padding placeholder means absent padding. A new copy-restore round trip
covers present and absent padding, sparse state slots, lengths, offset and
independent backing arrays.

## Validation

The final full unit target passed: **2,878 passed, 15 skipped,
0 failed** (2,893 test declarations, including parameterized tests).
This run covers all final source changes, including the deterministic warm-prefill
drain helper. The 78-test focused run and isolated 1-test memory run also passed.
Exact results are recorded in `validation.json`. `small-cache.json` and
`small-cache-diagnostics.log` preserve the explicitly isolated measurement: all
24 checkout deltas were **64 bytes**, versus **2,097,232 bytes** for copy restore.
All retired request owners released their cache references.

The Standards review has no remaining code findings; its drain-test timing
concern was fixed with an actor-isolated `Task.immediate` helper. The Spec
review has no remaining implementation findings; the unmeasured acceptance
gates below remain open. Strict Swift formatting and documentation-reference
checks pass. SwiftLint reports 11 warnings and no errors on the changed files.
The final Release build passed; its binary checksum is in `environment.json`.
The baseline implementation's [evidence](../2026-09-12/README.md) remains
historical evidence; its old full-target failure is not presented as a green
run. New evidence files are checksummed by `manifest.json`.

To enable allocator thresholds for the isolated evidence suite, use
`TEST_RUNNER_TESSERACT_ISOLATED_LEAF_CHECKOUT_EVIDENCE=1` with the documented
serial Xcode invocation and only `LeafCheckoutMemoryEvidenceTests`. Do not set
that flag for combined suites. The full target retains ownership assertions
but deliberately disables process-global allocation thresholds.

Loaded Qwen3.8/DFlash2 45k/75k/93k parity, peak/settled footprint, HTTP-tail and
production cleanup measurements remain pending the owner's approved resource
plan. These code fixes and unit evidence do not establish those performance
gates. `ActiveInferenceReserve` remains unchanged.
