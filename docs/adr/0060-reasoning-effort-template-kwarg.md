# ADR-0060: Reasoning effort as a template kwarg; thinking-safeguard budget split

- Status: Accepted
- Date: 2026-08-24
- Relates to: issue #98 / PRD #94 (template render kwargs, Preserve-Thinking
  Render), ADR-0033 (completion phase map), the thinking-loop safeguard
  (`ThinkingRepetitionDetector`, no prior ADR)

## Context

Qwen3.8-27B natively supports `reasoning_effort` — but not the way OpenAI
serves it. The chat template consumes it as a Jinja kwarg
(`reasoning_effort|default('xhigh')`, allowed values `low`/`medium`/`xhigh`)
and injects plain prose into the request's **first system block**: `xhigh`
and `low` each inject an instruction sentence, `medium` injects nothing, and
an unknown value raises inside the template. Three consequences follow:

- A different level is a different prefix **from token 0** — changing effort
  mid-conversation is a full re-prefill by construction, and effort must fold
  into the template-context digest / cache partition like every render kwarg.
- The template's default is `xhigh`, so "omit the kwarg" is byte-identical to
  today's renders — existing clients and their cache partitions survive.
- Validation must happen before render, or the Jinja exception 500s.

Tesseract had no effort plumbing: the wire field was decoded and dropped, the
render-kwarg channel (`TemplateRenderFlag`) was a Bool-only one-case enum,
and Pi — the motivating client — never even sent the field (its
`thinkingFormat: "qwen-chat-template"` compat collapses the level to
`chat_template_kwargs.enable_thinking`, which Tesseract also dropped).

Meanwhile the only thinking-length control was the thinking-loop safeguard's
budget trigger (`maxThinkingChars = 16_384`, hardcoded): one of four triggers
in `ThinkingRepetitionDetector`, the other three being repetition heuristics.
On an effort-native model a hard budget fights the native mechanism — a
legitimate `xhigh` think can exceed 4.5K tokens.

Two adjacent defects surfaced during design and are fixed here: (a) the
template-default polarity heuristic did not recognize Qwen3.8's
`preserve_thinking is undefined or … is true` shape, recording
preserve-by-default as strip-by-default — so the server emitted a spurious
`preserve_thinking: true` on every request, fragmenting the cache partition
off the canonical digest for a render the template produced anyway; (b) the
resolve rule "absent from app-enabled set = desired false" would have
force-emitted `enable_thinking` on templates whose default is off the moment
the flag was modeled.

## Decision

1. **Effort is a first-class render-kwarg.** `TemplateRenderContext` carries
   `reasoningEffort: ReasoningEffort?` (`low`/`medium`/`xhigh`) beside the
   boolean flags: emitted only when it differs from the template's own
   default (parsed from the `|default('…')` expression), folded into the
   digest as a JSON string, merged into `additionalContext`. Capability is
   template introspection (`ModelIdentity.declaresReasoningEffort`), never
   model name — the same rule as the boolean flags.

2. **Wire contract.** `/v1/chat/completions` accepts the union of the OpenAI
   and Qwen vocabularies — `minimal→low`, `low→low`, `medium→medium`,
   `high→xhigh`, `xhigh→xhigh` — as the top-level `reasoning_effort` field
   and inside `chat_template_kwargs` (the native channel wins, matching the
   request-kwargs-first precedence of the flags). Unknown values, including
   `"none"`, 400 before the lease; thinking-off is expressed as
   `chat_template_kwargs {"enable_thinking": false}` instead. A valid level
   sent to a non-declaring model is ignored with a log line — OpenAI
   unsupported-parameter semantics, never an error.

3. **`enable_thinking` is modeled** as the second `TemplateRenderFlag`. The
   resolve rule changes from an app-enabled *set* to an app-desired
   *dictionary*: a flag with no app stance follows the template default and
   is emitted only on an explicit request value. An emitted
   `enable_thinking: false` also swaps the generation prompt to a closed,
   empty think block, so `startsInsideThinkBlock` now derives from
   `promptStartsThinking && !renderContext.disablesThinking`.

4. **Safeguard budget split by trigger purpose.** The three repetition
   triggers stay always-on for every model — they catch loops, not effort.
   The budget trigger becomes: a fixed hidden anti-runaway ceiling of
   65_536 chars (≈18K thinking tokens) for effort-native models, and the
   settings-exposed legacy cutoff (`thinkingBudgetCutoffEnabled` /
   `thinkingBudgetCutoffChars`, default on / 16_384) for everything else.
   The budget threshold no longer waits out the repetition grace period —
   it is an absolute limit, and a cutoff configured below the grace must
   still cut. Per-request `thinking_safeguard` vendor-extension overrides
   stay authoritative over the policy.

5. **Agent side.** A global Reasoning Effort setting
   (`agentReasoningEffort`, default Automatic = inject nothing) rides
   `AgentGenerateParameters` to the internal routing edge, where the loaded
   model's identity resolves emission and the native budget ceiling — the
   agent's render context stays canonical-plus-effort, so today's renders
   are byte-identical under Automatic. The polarity fix restores the
   canonical digest for Qwen3.8 HTTP traffic (a one-time partition
   migration: old fragmented-digest partitions age out via Stale-Partition
   GC).

6. **Pi.** `~/.pi/agent/models.json` drops `compat.thinkingFormat` (falling
   back to Pi's `"openai"` format, which emits top-level `reasoning_effort`).
   No `thinkingLevelMap` is pinned: Pi clamps its `xhigh` level to `high` on
   the wire, which the server's union vocabulary maps straight back to native
   `xhigh` — the map would be redundant. Pi then stops sending
   `preserve_thinking: true` — inert, because the per-model app setting
   (default on) resolves the same render.

## Consequences

- Pi's thinking-level picker becomes real: each level renders a distinct
  system block, each level is its own cache partition, and switching levels
  mid-session re-prefills from token 0 — correct and expected, not a cache
  bug.
- Omitted effort stays the model's heavy default (`xhigh`); "Automatic"
  everywhere means zero byte change at ship.
- Qwen3.8 HTTP conversations migrate partitions once (spurious
  `preserve_thinking: true` no longer emitted); the stranded partitions age
  out via Stale-Partition GC.
- An effort-native model can think ≈4× longer than the old hard cap before
  any intervention; the repetition triggers remain the loop net at every
  length.
- A future effort-native template is a zero-code addition (introspection),
  and its default level is parsed, not assumed.

## Rejected

- **Mapping request effort onto safeguard budgets for non-native models** —
  conflates two mechanisms; the legacy cutoff stays server-configured.
- **Gating the whole safeguard on capability** — would strip loop
  protection from the primary model; only the budget trigger splits.
- **`reasoning_effort: "none"` → `enable_thinking: false`** — thinking-off
  drags the sampling-defaults question (the card recommends different
  temperature/top_p for non-thinking mode) into scope; rejected value,
  clear error, native kwarg instead.
- **A server-side default-effort setting** — omitted means the template
  default, exactly OpenAI semantics; a default knob can layer on later
  without wire changes.
