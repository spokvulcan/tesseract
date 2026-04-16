# OpenCode Integration — Research & Setup

**Date:** 2026-04-16
**Status:** Research complete. Shipping with hardcoded `opencode.json` for now. Plugin path documented for later.

## 1. Goal

Let OpenCode use Tesseract's local HTTP server (`http://127.0.0.1:8321/v1`) as an OpenAI-compatible provider without manually enumerating every model in `opencode.json` each time the Tesseract catalog changes.

## 2. Findings

### 2.1 OpenCode does not auto-discover from `/v1/models`

OpenCode's custom-provider catalog is fully **static** in `opencode.json`. The provider's `models` object is a hardcoded map; keys must match exactly what the provider's API accepts in `request.model`. OpenCode does **not** call the provider's `/v1/models` endpoint at runtime.

An upstream RFE tracks this: [anomalyco/opencode#6231 — Auto-discover models from OpenAI-compatible provider endpoints](https://github.com/anomalyco/opencode/issues/6231). Open as of OpenCode 1.0.58+.

### 2.2 Catalog source: models.dev

OpenCode pulls its curated model catalog from `https://models.dev`, caches it to `models.json`, and refreshes every hour. This catalog drives the `provider: {}` object for _known_ providers (OpenAI, Anthropic, Bedrock, etc.). Custom providers like Tesseract are not served from models.dev — they live entirely in user config.

Submitting Tesseract to models.dev is the **wrong approach**: the catalog there is for commercial cloud APIs, not per-device local providers whose model list depends on what the user downloaded.

### 2.3 Per-model fields OpenCode understands

From [opencode.ai/docs/models](https://opencode.ai/docs/models/) and [deepwiki](https://deepwiki.com/sst/opencode/3.3-provider-and-model-configuration):

| Field | Purpose |
|---|---|
| `name` | UI display label |
| `tool_call` | Boolean — advertises function-calling support |
| `reasoning` | Boolean — advertises reasoning/thinking tokens |
| `attachment` | Boolean — advertises multimodal attachment (image) support |
| `limit.context` | Max input token capacity |
| `limit.output` | Max output/generation tokens |
| `cost` | Pricing metadata (not relevant locally) |
| `release_date` | Metadata |
| `temperature` | Default temperature |
| `options` | Provider-specific params (e.g. `reasoningEffort`, `thinking.budgetTokens`) |
| `modalities.input`/`.output` | `"text"`, `"image"`, `"embedding"` |

Provider-level fields:

| Field | Purpose |
|---|---|
| `npm` | `@ai-sdk/openai-compatible` for `/v1/chat/completions`; `@ai-sdk/openai` for `/v1/responses` |
| `name` | UI display label |
| `options.baseURL` | API endpoint (should end at `/v1`) |
| `options.apiKey` | Optional, supports `{env:VAR}` |
| `options.headers` | Custom HTTP headers |

### 2.4 Tesseract's current `/v1/models` response

Handler: `DependencyContainer.swift:389`. Payload (`OpenAI.ModelObject`):

```json
{
  "id": "qwen3.5-4b-paro",
  "object": "model",
  "type": "llm",
  "owned_by": "tesseract",
  "max_context_length": 131072,
  "loaded_context_length": 131072,  // only when state == "loaded"
  "state": "loaded" | "available"
}
```

Sufficient for client sanity checks and for a future plugin that would populate `limit.context` dynamically. Missing fields worth adding before building a plugin: `display_name`, `family`, `capabilities: {tool_call, reasoning, vision, attachment}`, `suggested_max_output_tokens`, `default_temperature`.

## 3. Options Evaluated

### Option A — Hardcoded `opencode.json` (chosen for now)

Status-quo approach: enumerate each Tesseract model by hand in `~/.config/opencode/opencode.json`. What the screenshot in the research session was already showing.

**Pros:** zero moving parts, works today.
**Cons:** manual churn whenever the Tesseract catalog changes (new PARO size, new base model).

### Option B — OpenCode plugin (`config` hook)

Write a tiny plugin that runs at OpenCode startup, calls `GET http://127.0.0.1:8321/v1/models`, and merges each returned model into `config.provider.tesseract.models`. This is exactly the pattern used by [`agustif/opencode-lmstudio`](https://github.com/agustif/opencode-lmstudio).

Reference plugin skeleton (`~/.config/opencode/plugin/tesseract.ts`):

```typescript
import type { Plugin } from "@opencode-ai/plugin"

const BASE = "http://127.0.0.1:8321/v1"

export const TesseractPlugin: Plugin = async () => ({
  config: async (config) => {
    let body: { data: Array<{ id: string; max_context_length?: number; state?: string }> }
    try {
      const r = await fetch(`${BASE}/models`, { signal: AbortSignal.timeout(800) })
      if (!r.ok) return
      body = await r.json()
    } catch {
      return // server off — fall back to static config
    }

    config.provider ??= {}
    config.provider.tesseract ??= {
      npm: "@ai-sdk/openai-compatible",
      name: "Tesseract",
      options: { baseURL: BASE },
      models: {},
    }
    const p = config.provider.tesseract
    p.models ??= {}

    for (const m of body.data) {
      if (p.models[m.id]) continue // user-authored override wins
      p.models[m.id] = {
        name: pretty(m.id),
        tool_call: true,
        reasoning: m.id.includes("paro"),
        attachment: false,
        limit: { context: m.max_context_length ?? 131072, output: 16384 },
      }
    }
  },
})

function pretty(id: string): string {
  return id.replace(/-/g, " ").replace(/\b\w/g, c => c.toUpperCase()) + " (Tesseract)"
}
```

Wired via:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": ["file:///Users/owl/.config/opencode/plugin/tesseract.ts"]
}
```

**Pros:** zero-config after install; catalog stays live; survives upstream changes.
**Cons:** new artifact to maintain; requires Bun/Node at OpenCode install time. String-sniffing model ids (`id.includes("paro")`) is brittle until `/v1/models` advertises capabilities.

### Option C — Wait for upstream auto-discovery

If [anomalyco/opencode#6231](https://github.com/anomalyco/opencode/issues/6231) merges, the plugin becomes unnecessary — OpenCode itself will call `/v1/models` for any `@ai-sdk/openai-compatible` provider and merge results.

**Pros:** zero custom code eventually.
**Cons:** unmerged today; no timeline.

### Rejected — models.dev submission

Public commercial registry; Tesseract catalog is device-specific. Wrong fit.

### Rejected — `.well-known/opencode`

Used for organization-wide config distribution, not local provider discovery.

### Rejected — `ctrl+a` Connect provider (TUI)

Triggers `opencode auth login` → API-key storage only for models.dev providers. Not applicable here.

## 4. Chosen Path (current release)

**Option A — hardcoded `opencode.json`.** Current config at `~/.config/opencode/opencode.json` enumerates:

- `qwen3.5-4b-paro`
- `qwen3.5-9b-paro`
- `qwen3.5-27b-paro` *(added 2026-04-16)*
- `qwen3.5-4b`

All pointed at `http://127.0.0.1:8321/v1` via `@ai-sdk/openai-compatible`, with `limit.context: 131072` and `limit.output: 16384`.

**Maintenance rule:** when `ModelDefinition.agentModels` in `tesseract/Features/Models/ModelDefinition.swift` gains or removes a model, mirror the change in `~/.config/opencode/opencode.json` under `provider.tesseract.models`.

## 4.1 Output-token sizing (Qwen3.5 family)

Source: official Qwen HuggingFace cards for [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B).

| Spec | Value |
|---|---|
| Native context | **262,144** tokens (256K), extensible to ~1,010,000 via YaRN |
| Tesseract-advertised context | **262,144** tokens (256K) — equal to Qwen's native `max_position_embeddings`, verified against the downloaded `mlx-community/Qwen3.5-4B-MLX-8bit` `config.json` |
| Recommended output (general) | **32,768** tokens |
| Recommended output (complex, e.g. coding competitions) | **81,920** tokens |

OpenCode sends `max_completion_tokens` equal to the provider's `limit.output`. The initial 16,384 value was well below Qwen's 32K/82K guidance, which is why long coding turns were cut off mid-response.

**Decision:** set `limit.context: 262144` and `limit.output: 262144` for every Tesseract model — both equal to the Qwen3.5 native context window (`max_position_embeddings = 262144` from the model's `config.json`). Rationale:

- The true cap is the context window, not an artificial output budget. With `output == context`, the binding constraint becomes `context - prompt_tokens`, which is always correct.
- No risk of overallocation: Tesseract's generation loop (`CompletionHandler.swift:378, 521`) stops at `generationTokenCount >= params.maxTokens`, and the KV cache is bounded by the context window regardless.
- Avoids having to update `limit.output` separately from `limit.context` when we adjust the advertised window.

### Tesseract app-side changes (2026-04-16)

Previously the app hardcoded `131_072` for the advertised window and `120_000` for the agent-chat compaction threshold — both below the model's native capacity. Bumped everywhere to `262_144` (Qwen3.5 native `max_position_embeddings`):

| Location | Before | After |
|---|---|---|
| `AgentGenerateParameters.maxTokens` default | `131_072` | `262_144` |
| `DependencyContainer.swift:411` `max_context_length` | `131_072` | `262_144` |
| `DependencyContainer.swift:412` `loaded_context_length` | `131_072` | `262_144` |
| `DependencyContainer.swift:134` compaction `contextWindow` | `120_000` | `262_144` |
| `AgentFactory.swift:77` compaction `contextWindow` | `120_000` | `262_144` |
| `BackgroundAgentFactory.swift:121` compaction `contextWindow` | `120_000` | `262_144` |
| `AgentCoordinator.swift:118` default `contextWindow` | `120_000` | `262_144` |
| `CompactionSettings.standard` header comment | references "120K context window models" | references "Qwen3.5-family models (262,144 native context)" |
| `CompletionHandler.swift:271` | Honors request `max_completion_tokens` / `max_tokens` verbatim; no server-side cap | unchanged (no cap was ever imposed) |
| Benchmark suites | Per-round caps (e.g. `maxTokensPerRound` in `BenchmarkConfig.swift`) | unchanged — scoped to benchmarks, no leakage to production |
| TTS `TTSParameters.maxTokens = 4096` | — | unchanged (unrelated — speech synthesis) |

When a client (e.g. OpenCode) omits `max_tokens`, generation falls through to the 262,144 default in `AgentGenerateParameters`, effectively "generate until context runs out."

### Follow-ups to consider

- **Reserve sizing.** `CompactionSettings.standard.reserveTokens = 16_384` was originally sized against a 120K window. It still gives 245K of pre-compaction room on a 262K window, which is fine, but if long-form generations from the *agent chat* (not the HTTP server, which bypasses compaction) start getting truncated near the tail, raise `reserveTokens` toward 81,920 to match Qwen's "complex problems" recommendation.
- **Memory.** At 262K, KV cache and prefix-cache budgets grow ~2× versus the old 128K setting. Watch `LLMActor.Defaults.prefixCacheMemoryBudgetBytes` and real-world unified-memory usage on smaller Macs; may need to cap per-request effective context if OOMs surface on 36 GB systems.
- **YaRN.** If we ever need >262K, switch to rope `factor` scaling (`factor=2.0` → 524K, `factor=4.0` → 1 M) per the Qwen YaRN config block documented on the HF card. `max_position_embeddings` in the checkpoint stays at 262144; the runtime rope config does the extension.

## 5. Future Work (ordered)

1. **Enrich `/v1/models` response.** Extend `OpenAI.ModelObject` in `tesseract/Features/Server/Models/OpenAITypes.swift:285` with:
   - `display_name`
   - `family` (`"qwen3.5-paro"`, `"qwen3.5"`, …)
   - `capabilities: { tool_call, reasoning, vision, attachment }`
   - `suggested_max_output_tokens`
   - `default_temperature`

   Source the values from `ModelDefinition` rather than string-sniffing the id. Update `docs/HTTP_SERVER_SPEC.md` §4.2.

2. **Ship `opencode-tesseract` plugin** (Option B above). Live in `Extras/opencode/` with a one-line install script. Drops the manual-update rule in §4.

3. **Re-evaluate when #6231 merges.** If upstream OpenCode auto-discovers, delete the plugin and keep only the config block (or drop the config block entirely).

## Sources

- [OpenCode — Providers](https://opencode.ai/docs/providers/)
- [OpenCode — Models](https://opencode.ai/docs/models/)
- [OpenCode — Config](https://opencode.ai/docs/config/)
- [anomalyco/opencode#6231 — Auto-discover `/v1/models`](https://github.com/anomalyco/opencode/issues/6231)
- [DeepWiki — Provider and Model Configuration](https://deepwiki.com/sst/opencode/3.3-provider-and-model-configuration)
- [agustif/opencode-lmstudio — plugin pattern](https://github.com/agustif/opencode-lmstudio)
- [tobrun — Local LLM with OpenCode](https://tobrun.github.io/blog/add-openai-compatible-endpoint-to-opencode/)
