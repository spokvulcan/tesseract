# ADR-0070: Request Keying derives every per-request fact once; the Generation Prompt is measured from the template

- Status: Accepted (spec: #562)
- Date: 2026-09-24; accepted 2026-09-25, with the amendment below
- Relates to: ADR-0033 (phases return values; "extend a value, not thread a
  parameter"), ADR-0060 (template capability by introspection, never model
  name), ADR-0063 (the Conversation Render module and its probe-measured
  end-of-turn marker), ADR-0016 (Request Keying runs inside the Model
  Session), ADR-0069 (the claim's check-out reads text-only), ticket #473
  (the planner's generation-prompt measure had stayed outside Conversation
  Render)

## Context

A cache-aware completion reads a handful of facts about its request in many
places: whether generation starts inside a `<think>` block, where the
last-message boundary sits, whether the request is text-only, the prefill
step. Each read site derived its own copy from raw inputs, and the copies
drifted.

- Thinking start was a load-time guess (`ModelIdentity` finds `<think>`
  anywhere after `add_generation_prompt` in the template text) combined with
  the request's render context at four call sites. The Prefill Planner and
  the replay harness read the raw guess. #460 and #508 fixed one site each.
- The planner finds the last-message boundary by subtracting a hand-spelled
  ChatML string. Under `enable_thinking: false` the Qwen3.5/3.8 templates end
  in a closed `<think>\n\n</think>\n\n` block, which matched neither
  spelling, so those requests lost the boundary until #563 added a third
  spelling for it. The spellings are still ChatML-only and still start from
  the load-time guess, so a non-ChatML template never gets the boundary and
  a template that thinks only when asked gets the wrong spelling.
- The guess cannot see polarity: the Qwen3.5-0.8B template (thinking off by
  default) reads as "starts thinking".
- The leaf-store mode keyed a stop turn's canonical user leaf on thinking
  start alone. A thinking-off turn carries a closed, empty think block that
  a think-stripping template drops from history, so its live leaf could not
  be hit by the next user turn (found in #563).
- "Text-only" had four spellings. Two used the request's own images, so a
  text-class instance that dropped a request's images (#439: text-only by
  construction) was still refused a Leaf Handoff and a moving capture.
- The replay harness already measured the generation prompt from probe
  renders (`GenerationPromptProbe`): a sixth derivation of the same fact.

The pure deciders were tested. The bugs lived in which input each call site
fed them, inside an 860-line closure no test isolates.

## Decision

1. **The Generation Prompt is measured, never spelled, in two stages.**
   The Conversation Render module renders a one-message probe, without
   tools, with and without `add_generation_prompt` under the render
   context. It measures only when the render without the prompt is a byte
   prefix of the render with it and no token merges across the append
   point (the end-of-turn marker's hard-boundary check); an empty
   difference is measured and means the template appends nothing. The
   probe is memoized per model fingerprint and render-context digest, or
   per tokenizer instance for a load without a fingerprint. Each request
   then checks the probe against the tokens it fed, and only the checked
   value reaches consumers. The last-message boundary is the key path
   minus the checked tokens.
2. **The think block has four states.** The prompt opens a think block
   (its last think-open tag, as the stream parser spells it, has no close
   after it), closes an empty one (`enable_thinking: false`), has none, or
   is unknown: the probe failed, or the request did not feed what it
   measured. Only an open block starts the parser inside. A stop turn takes
   the canonical user leaf whenever its prompt carries a think block, open
   or closed, under a think-stripping render, because the template drops
   either from history once a new user message arrives; with none it takes
   the direct leaf. Unknown has its own answer at every consumer: the
   parser starts outside, a stop turn takes the canonical user leaf
   (correct for any template), MTP does not engage, no last-message
   boundary is placed, and the request logs the reason. Every catalog model
   must measure, so no shipped model reaches the unknown state.
3. **Request Keying yields a Keyed Request** carrying the identities and
   every per-request fact, derived once: the Generation Prompt, text-only
   by instance truth, the prefill step, the decode parameters, the SSD
   gate and the tool-call format. An Unkeyed Completion carries the same
   facts and no key space or render, as a separate case, not an optional.
4. **Consumers take the fact, never its ingredients.** No parameter named
   `promptStartsThinking` survives downstream of Request Keying. The fact
   values are constructible only by their one derivation. The generation
   value, the Leaf Store phase and the Speculative Canonical Prefill seed
   take the Keyed Request whole instead of copied fields.
5. **The code that tokenized the prompt reports its Generation Prompt.**
   Server Completion and the Raw Generation Start both hand it to the
   stream loop on the start handle. The load-time guess is deleted; the
   chat view's spinner reads the canonical context's measurement at load.
6. **Text-only means no image reached the model.** Check-out, capture,
   engagement and transient-boundary gating all read that one fact.

## Considered options

- **Fix each call site as it is found.** #508 did this for the MTP
  predictor, and #563 for the planner and the replay harness, adding the
  closed-block spelling. Each fixes its site and leaves the class: the
  strings stay ChatML-only, and a string table plus a flag are still two
  facts that must agree. ADR-0060 rules out capability by family.
- **Measure from each request's own renders.** Exact even for a template
  whose prompt depends on the conversation, but it costs a trim-recovery on
  every request. No shipped template's prompt depends on anything but
  `add_generation_prompt` and `enable_thinking`, and the per-request check
  sends a request that breaks that assumption to the unknown state.
- **Keep the load-time guess as a fallback** for the unknown state. It keeps
  a second derivation alive for a path the catalog gate keeps shipped
  models off, and it is wrong for thinking-off requests and for templates
  that think only when asked, which is where a fallback would matter.
- **Treat unknown as "outside" everywhere.** The stream parser emits text
  outside a think block as it arrives and cannot retract it; only text still
  buffered when a stray `</think>` arrives is reclassified. And "outside"
  selects the direct leaf mode, which is not a display-only choice.
- **Seal consumers further** (the claim and the speculative-arm policies
  take a key space instead of a Bool). This churns pure policy tables shared
  with the agent path for little gain once the Bool has one source.

## Consequences

- The planner's hand-spelled generation prompts, three since #563, are
  deleted. A template that thinks only when asked, or is not ChatML-shaped,
  gets its last-message boundary too.
- A thinking-off stop turn under a think-stripping render stores the
  canonical user leaf, so the next user turn hits it. It no longer takes
  the MTP arm, which forfeits the boundary snapshot that leaf is built
  from; a reusable leaf is worth more than a faster cold decode.
- A dropped-image request on a text-class instance takes the leaf by
  handoff and captures by move, as any text request does. The
  instance-truth test that pinned a copy restore changes.
- The planner's generation-prompt measure moves inside the Conversation
  Render module; only the agent hand-off suffix remains a plain-text encode
  outside it.
- On a template whose Generation Prompt is unknown, reasoning streams to
  the client as content until `</think>`. The cache stays correct, because
  the leaf mode and the fidelity check follow the same state; the cost is
  display and speed, and the catalog gate keeps it off shipped models.
- A render context the probe has seen costs no renders; a new one costs two
  one-message renders once per model load.
- Tests build facts by measuring through a fake tokenizer or by running
  Request Keying on the toy Model Session; a test can no longer configure a
  template and a flag that disagree.

## Amendment (2026-09-25): as built

- The probe memo keys on the tokenizer's type as well as the fingerprint,
  so two kinds of tokenizer handed one fingerprint (test doubles) never
  share a probe. Without a fingerprint a value-type tokenizer has no
  instance to key on and is measured on every ask; production always has
  a fingerprint.
- The facts also record whether the request defined tools, which the MTP
  prediction reads.
- The planner takes the Keyed Request. Its arithmetic stays callable from
  a Generation Prompt, key space and render, as an internal seam for the
  tokenizer-only planner tests and the CPU bench.
- An unknown Generation Prompt is logged on the request as a
  `generationPrompt` skip with its reason, and warned about once per
  model, render context and reason. The `leafStore` event reports the
  prompt's state on every request (`opens`, `closed`, `none` or
  `unknown(<reason>)`).
- The chat view's spinner reads the canonical context's Generation
  Prompt, measured when the model loads, which also warms the probe for
  that context.
