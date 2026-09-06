# ADR-0063: Emitted Path Index — the ids the model fed are the truth for server-generated turns

- Status: Proposed (documents-first, ticket #472 of
  [issue #471](https://github.com/spokvulcan/tesseract/issues/471); flips
  to Accepted with as-built notes at ticket #480)
- Date: 2026-09-06
- Relates to: ADR-0062 (superseded by this ADR together with ADR-0064 once
  both are Accepted), ADR-0009 (speculative canonical prefill; stays for
  think-stripping templates at a user boundary), ADR-0033 (`LeafStorePhase`),
  ADR-0060 (render kwargs and the cache partition), ADR-0064 (Leaf Handoff),
  [issue #466](https://github.com/spokvulcan/tesseract/issues/466) (closed
  as superseded by #471)

## Context

A finished turn's leaf has been keyed on the chat template's re-render of
the turn, tokenized fresh. The model's own token choices do not survive that
round trip. It emits non-canonical BPE splits of ordinary text (`KN`+`I` for
`KNI` re-tokenizes as `K`+`NI`; `>S`+`outh` as `>`+`South`), and it writes
JSON with spaces that the template's `tojson` renders compactly. When the
re-render differs from the ids the model fed, the **Live Leaf Capture** of
ADR-0062 falls back to restoring the boundary snapshot and re-prefilling
every generated token at 5 to 7 ms each, and the client waits for that before
its terminal chunk.

Two Pi sessions on 2026-09-06 against Qwen3.8-27B with thinking preserved,
read from the request recordings, the `event=leafStore` log lines and the
diagnostics corpus: a 13.1k-token `write` turn re-prefilled 13,111 tokens
and the client waited 66 s; a 12.7k-token `write` waited 81 s; six `edit`
turns of 4.5k to 7.9k tokens waited 42 to 54 s each; six smaller `edit`
turns 1.9 to 6.6 s. In total 216 s of post-EOS wait in one session, none of
it in a class a render fix can remove.

The ADR-0062 comparison earned its keep before it was retired. Decoding the
ids it logged on its first day found two breakers of the **Append-Stable
Render**, neither in this repository: the vendored streaming detokenizer
re-emitted the base character of any grapheme cluster that grew across a
token boundary (`🏳️‍🌈` reached the client as `🏳🏳️🏳️‍🏳️‍🌈`), fixed in the
mlx-swift-lm fork and filed upstream as
[mlx-swift-lm #613](https://github.com/ml-explore/mlx-swift-lm/pull/613);
and swift-jinja's `tojson` escaped `/` as `\/` and non-ASCII as `\uXXXX`, so
a re-rendered `edit` call never matched the emitted text, fixed in the
swift-transformers fork and filed as
[swift-jinja #72](https://github.com/huggingface/swift-jinja/pull/72). The
same comparison surfaced the vendored tool-call processor dropping the text
before a tool call
([mlx-swift-lm #610](https://github.com/ml-explore/mlx-swift-lm/pull/610)).
With all three fixed and in the running build, every fallback in the table
above remained: the model's JSON spacing inside array-typed tool arguments,
its parameter order (issue #466), and the BPE re-splits are properties of
what the model emitted, which a parse-and-render round trip cannot
reproduce.

The owner's product rule: with a preserve-thinking model, a coding-agent
session never prefills a token the model already produced. Nothing about the
solution may rest on trusting the client; an edited, compacted or reordered
history must render canonically and reuse whatever prefix the tree has,
exactly as today.

## Decision

1. **The Emitted Path is the truth for server-generated turns.** The token
   path a conversation prefix actually took through the model on this
   server, the prompt ids as fed plus the generated ids, ending with the
   canonical end-of-turn id, is what the next request feeds for every
   assistant turn this server generated. Canonical re-encoding of such a
   turn is no longer used to build a prompt or a leaf key.

2. **Whole-prefix index key.** Per model fingerprint, the **Emitted Path
   Index** maps the SHA-256 of the template's rendered bytes, from the start
   of the render through a server-generated assistant message's end-of-turn
   marker, to that prefix's Emitted Path. Keying on the whole rendered prefix
   scopes every entry by its preceding history for free: the same text after
   a different history is a different key. The template's own normalization
   (trim on reasoning, `tojson` on object and array parameters) is the
   accepted normalization; an echo the template renders byte-identically is
   the same message, anything else misses.

3. **Registration** happens at the end of every stored turn, before anything
   else in the Leaf Store: one Jinja render of the stored conversation to
   bytes, no tokenization; the hash of the bytes up to the last end-of-turn
   marker is registered against the request's **Cache Key Path** plus the
   recorded generated ids. If the model stopped on a token other than the
   end-of-turn id, or was cut by the token limit, the canonical end-of-turn
   id is appended so the path matches what the template renders; the leaf is
   captured at the cache's own offset, so an unfed speculative bonus token or
   an appended end-of-turn id is prefilled by the next request. Registration
   requires all of: the request's **Cache Key Space** is the identity
   (text-only; an image-bearing request's Cache Key Path carries digest
   pseudo-tokens that are not model input), the recorder fed at least one id,
   the live cache's offset lies inside the live path, the turn was not
   intervened, and the fidelity check of decision 9 passed. A turn that fails
   any of these registers nothing and logs the reason.

4. **Resolve inside the Conversation Render module**, for every spelling of
   render-and-encode (the request edge, the planner's last-user render, the
   leaf store's continuation and base renders, the agent edge, the admission
   builder's probe renders): render to bytes; compute the running hash with a
   snapshot at every end-of-turn marker in one pass; look up from the last
   marker backwards; on the deepest hit take the entry's path and canonically
   encode only the bytes after that marker; concatenate. The suffix begins
   immediately after a special token, a hard pretoken boundary for byte-level
   BPE, so the standalone suffix encode equals the in-context one. The
   Render+Token Cache keeps its exactness contract because it never holds an
   emitted id: it serves the canonical encode of suffixes and of whole
   renders on a miss. The resolve consults the index for text-only requests
   only; an image-bearing request never resolves through it and keeps today's
   prepare, boundary and copy paths.

5. **Single seam.** The token list the resolve returns is both the model
   input and the Cache Key Space input, as it is today at the request edge.
   No path may build tokens for a request outside the resolve.

6. **Same-key policy: last writer wins.** A same-parent regeneration with
   identical text and a different split overwrites the earlier entry. The
   earlier branch's next request then resolves to the later branch's path: if
   that leaf is resident it is an immediate hit on the later branch's state,
   which matches the resolved input exactly while the earlier generation's
   own split is lost; otherwise the request costs one canonical miss. Either
   outcome is consistent, never a wrong prompt; what is given up is the exact
   identity of the earlier generation. A counter and a log line record each
   overwrite with both path lengths.

7. **Retention.** In memory, bounded by a byte budget of ids (32 MB by
   default), least-recently-used eviction, cleared on model unload or
   fingerprint change. Not tied to KV eviction: the index is cheap provenance
   and outlives the leaf it points at. The entry format (fingerprint, hash,
   path length, ids) is fixed now so persistence beside the SSD manifest can
   be added later without migration.

8. **The miss path is always safe.** Anything not indexed renders canonically
   and reuses whatever prefix the radix tree has: the first turn of a
   session, client-authored text, edited or compacted history, stripped
   reasoning, a think-stripping template at a user boundary, a restart. A
   miss is never a wrong prompt.

9. **Fidelity check gates registration.** Because the stored path is now the
   emitted path by construction, a token comparison can no longer detect a
   response-conversion bug. At registration the server detokenizes the
   emitted ids and compares the text with the canonical render under the
   template's normalization: thinking and content equal after trim, object
   and array parameters equal as parsed JSON. That normalization is the only
   accepted difference. Beyond it the mapping is rejected: nothing is
   registered for the turn, a warning-level diagnostics event and a counter
   record the mismatch, and the leaf is still stored live under the emitted
   path (the state matches those ids by construction, and the tree reuses
   whatever prefix the next canonical render shares). The next request
   renders canonically and re-prefills from the first differing token, so a
   conversion bug costs a visible re-prefill instead of silently mapping text
   the client saw onto tokens it did not. Exact cache-key equality proves
   that state matches the selected tokens; only this check proves that those
   tokens represent the request. The offline Canonical-Echo Fidelity harness
   learns the index so its verdicts stay meaningful.

10. **Leaf Store fast path.** For any finished turn the template renders
    verbatim for the next request (every turn under the **Preserve-Thinking
    Render**; every tool-stretch turn under the Qwen3-family templates), the
    Leaf Store reduces to: register the index entry, capture the leaf at the
    live cache's offset, admit it under the live path. No canonical
    render-and-tokenize, no continuation probes, no boundary snapshot lookup,
    no live-versus-render comparison. The leaf ends exactly at the live
    cache's offset; the next request prefills the few glue tokens with its
    new message. The structural guards of ADR-0062's decision survive as the
    fast path's eligibility (intervened turn, non-identity key space, no fed
    ids, cache offset outside the live path), each still logged with its
    reason; only the per-token comparison and its divergence logging are
    deleted.

11. **The boundary path narrows.** The boundary restore-and-re-prefill and the
    speculative seed of ADR-0009 remain only for a think-stripping template at
    a new-user-message boundary and for guard failures. Taking the fast path
    on a turn the template later strips is still safe: the stripped render
    misses the index, encodes canonically, and the tree matches up to the
    strip point. Intervened turns (a thinking-safeguard continuation swapped
    the generation) keep the boundary path in this change; their frequency is
    counted before deciding whether they need the fast path.

12. **Render kwargs and the partition are unchanged.** A reasoning-effort
    change on Qwen3.8 re-prefills from token 0 as ADR-0060 specifies, because
    the first system block's bytes change and no index entry can match them.
    A kwarg that changes only the generation prompt (`enable_thinking` on the
    Qwen3-family templates) keeps selecting its own cache partition through
    the template-context digest, so such a flip still misses the whole
    history; whether it should is a separate decision. The index is keyed
    per model fingerprint on rendered bytes and is partition-agnostic, so
    that decision needs no index change.

13. **Telemetry.** The `leafStore` event carries the leaf source and the
    boundary reason. New events: index registration (prefix length, path
    length) or its skip reason, index resolve (indexed prefix length, suffix
    tokens encoded, or miss reason), same-key overwrite, fidelity mismatch.
    The trace corpus gains optional fields for the same without a schema
    bump.

The change lands as a dark launch first (the index registers only turns the
ADR-0062 comparison already proves equal, and a shadow comparison proves on
real traffic that the resolved path equals the canonical encode) and as the
fast path second, so the special-token boundary claim of decision 4 is
verified in production before anything depends on it.

## Consequences

- The re-prefill classes in the table above vanish: identical text with a
  different split, JSON spacing, parameter order. The wait after the last
  token becomes tens of milliseconds at any context length below 20k and
  grows only with the suffix written to SSD (ADR-0064 removes the remaining
  capture copy).
- The **Append-Stable Render** is no longer load-bearing. It remains a
  property some templates have; the cache no longer depends on it for zero
  re-prefill, and the Live Leaf Capture is by construction for an indexed
  turn.
- The ADR-0062 comparison's diagnostic value moves to the text level: the
  fidelity check is the bug detector for the response conversion, and a
  mismatch is visible as a re-prefill rather than hidden as a hit.
- A restart costs at most one canonical miss per conversation and never a
  wrong state; losing the in-memory index is safe.
- The same-parent, same-text, different-split regeneration loses the exact
  identity of the earlier generation (decision 6). Accepted and counted.
- Image-bearing requests, quantized-KV partitions, the MTP path and the
  Batch Engine keep today's paths. Think-stripping templates at a user
  boundary keep ADR-0009's answer.
- Follow-ups filed from #471: persisting the index beside the SSD manifest;
  an optional lineage field for clients that can carry one; profiling the
  fixed warm-turn prefill overhead; the partitioning of generation-prompt-
  only kwargs.

## Rejected alternatives

- **A per-message key.** It cannot tell two generations with identical text
  apart, and every entry would need its own history scoping. The whole-prefix
  key scopes by history for free and leaves only the same-parent,
  same-text, different-split case ambiguous, which decision 6 resolves.
- **A lineage or session field in the HTTP API.** A protocol change for
  every client, and unnecessary: the whole-prefix hash reconstructs the
  lineage from what any OpenAI-compatible client already echoes. Kept as an
  optional follow-up for clients that can carry one.
- **Accommodating the model on the render side** (`tojson` separators,
  spacing, parameter order). Cannot cover BPE re-splits of ordinary text,
  and each accommodation chases one model's habits into a vendor template
  the repository never edits. Issue #466 is closed as superseded.
- **Keeping the ADR-0062 comparison as the gate.** It compares the emitted
  path against a canonical reference the product rule inverts for
  server-generated turns; with the index the two are equal by construction
  and there is nothing to compare.
- **Registering despite a fidelity mismatch** (the first draft of #471). It
  would map the client's text onto tokens the client never received and hide
  the very bug class the comparison used to catch. Rejected in the
  2026-09-06 review.
- **Mid-turn checkpoints to bound the fallback re-prefill.** They shrink a
  cost the index removes.
