# Allocation inventory: production capture and validation

Evidence for [#506](https://github.com/spokvulcan/tesseract/issues/506), under
[#505](https://github.com/spokvulcan/tesseract/issues/505). The
[full source inventory and interpretation](../../../docs/research/2026-09-12-local-inference-allocation-inventory.md)
separate logical bytes, shared owners, physical observations and open gaps.
Source baseline is `b1f0dca32ec9baead778221a8001fe4aa59e1850`; every capture
archives its exact instrumentation diff, binary hash, model-file hashes,
runner and helper. The successful campaign used one instrumented build;
earlier attempts used earlier instrumentation. This is not an optimization A/B test.

## Attempts and successful coverage

| Directory | Outcome |
| --- | --- |
| [startup-attempt](startup-attempt/outcome.json) | No inference sent. Initial harness waited for automatic model loading; app loads on first completion instead. Corrected before later attempts. |
| [production](production/outcome.json) | Cold load stopped at warning pressure. Zero completed requests; largest OS footprint sample 29,245,784,552 bytes. No load-phase probes existed in this binary. |
| [production-revised](production-revised/outcome.json) | Added load-phase probes, allowed warning pressure and raised footprint trigger from 28 to 32 GiB. Stopped at 512 MiB additional swap; largest observed growth 739,639,296 bytes (sampled triggers can overshoot). Target/draft loaded; stacking had not finished. |
| [production-final](production-final/outcome.json) | Same instrumented binary; 2 GiB swap-growth allowance. All seven scenarios completed in 171.809 s, with six successful responses and one intentional disconnect. No resource stop. All 641 OS samples had normal pressure and swap below the starting baseline. |

The maintainer explicitly requested production models and, after the first
warning stop, requested continuing on this 48 GiB Mac because no larger host
was available. Allowances were revised as documented; model precision and
active inference settings were not reduced. The final run's differing VM state,
file-cache state and swap baseline prevent attributing its success solely to
the changed allowance or claiming a memory reduction across attempts.

Final workload: fresh process/model load and synthetic prefix cache miss,
warm continuation, growth, disconnect after 3 seconds, resend, then two short
warm continuations. Largest prompt 12,219 tokens; max output 128 tokens,
temperature 0. DFlash2 engaged on all completed responses; unquantized KV,
MTP not loaded, vision off. Existing SSD metadata remained in place; no private
conversation was replayed or copied into this evidence bundle.

Measured highlights:

- Loading sample peak: 31,871,681,024 bytes (29.683 GiB); final idle footprint:
  18,964,311,104 bytes (17.662 GiB).
- Cancel/resend restored 7,136 valid rows with 306 MiB of unused full-attention
  array extent. This is capacity, not a proven physical savings amount.
- Client disconnect at 3.008 s; server cancellation telemetry at 25.958 s.
  Transport/handler/prefill boundaries need further attribution.
- One full and five extension SSD payloads committed. Full payload
  387,121,152 bytes plus separate 387,153,971-byte encoded container.
- 40 focused tests passed; Release build and format/docs checks passed.

## Files and reproduction

Each attempt has `environment.json`, `model-files.json`, `instrumentation.patch`,
`os-samples.json`, `diagnostics.jsonl`, `outcome.json` and a derived `summary.json`.
The final capture also has `request-attempts.json` and `requests.json`, recording
synthetic request/response hashes, usage, IDs and timing. `diagnostics.jsonl`
contains scalar fields only. The cancelled request has no usage response;
its prompt/cache counts come from its lookup event.

`validation/` contains the complete compressed focused-test log, its hash and
command, a readable pass summary and the Release build result. SHA256SUMS covers
all files in this evidence directory except itself. Regenerate summaries with:

```sh
python3 benchmarks/allocation-inventory/2026-09-12/summarize.py
```

The preserved final runner is the exact executed code. The current repository
runner prints a plan unless explicitly passed `--run`; an execution requires
one available local Release binary, downloaded production checkpoints, normal
starting memory pressure, an unused port and no other running Tesseract app.
Use a new output directory; the runner refuses to overwrite evidence.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 scripts/allocation_inventory_probe.py \
  --app '/path/to/Tesseract Agent.app/Contents/MacOS/Tesseract Agent' \
  --output benchmarks/allocation-inventory/NEW-CAPTURE \
  --allow-pressure-warning --footprint-stop-gib 32 \
  --swap-growth-stop-gib 2 --run
```

That command reproduces the final plan, not permission for unlimited retries
or larger contexts. Default flags instead stop at warning pressure, 28 GiB
footprint and 512 MiB additional swap. The runner samples at 250 ms and stops
at 180 s per HTTP response / 900 s overall, with a separate 60 s release wait.
It cancels its transport on a breach and
terminates only the child it launched. Sampling ends at the trigger; the later
termination window is not a measured peak. SIGTERM exit code -15 in a successful
outcome is deliberate end-of-campaign cleanup, not an app crash.

OS footprint, resident/compressed memory, system-wide swap and MLX active/cache
are overlapping accounting views. The request observer starts after model load;
loading has separate allocation events and OS samples. Existing request facts
carry freshness tags; `afterRelease` and five-second settling do not prove SSD
drain or an empty cache. These small-context observations do not close loaded
bitwise parity, explicit hydration/unload, long-context, alternate-layout or
whole-application leak gates.
