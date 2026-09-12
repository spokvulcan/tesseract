# Paired capture results

Values read **copy control → handoff**. GB is decimal. These are single observations; see the protocol and sampling limits in [README.md](README.md).

[baseline.json](baseline.json) retains exact summary values. Full traces are in the [evidence archive](https://github.com/spokvulcan/tesseract/releases/download/evidence-478-2026-09-08/capture-handoff-2026-09-08.zip), under `capture-handoff-2026-09-08/diagnostics/<mode>-<case>.jsonl`; request IDs below identify the corresponding runs.

| Case | Prompt / output tokens (copy → handoff) | Capture active Δ GB | Capture ms | Post-generation ms | Response seconds |
| --- | --- | --- | --- | --- | --- |
| short-cold | 2749 / 16 → 2749 / 16 | 0.336 → 0.000 | 5.288 → 0.207 | 43.177 → 38.069 | 26.622 → 23.173 |
| 46k-cold | 46154 / 128 → 46154 / 128 | 3.161 → -0.009 | 156.685 → 1.805 | 176.762 → 9.361 | 273.943 → 261.944 |
| 46k-warm | 46299 / 128 → 46299 / 128 | 3.170 → -0.028 | 2287.015 → 1.850 | 2651.606 → 192.294 | 7.942 → 9.823 |
| 46k-resend | 46444 / 128 → 46444 / 128 | 3.188 → -0.019 | 167.775 → 1.897 | 217.051 → 34.070 | 4.838 → 5.533 |
| 77k-restart | 76714 / 128 → 76714 / 128 | 5.163 → -0.028 | 242.508 → 2.449 | 292.761 → 40.553 | 522.163 → 521.244 |
| 77k-warm | 76859 / 110 → 76859 / 110 | 5.199 → 0.000 | 302.618 → 0.222 | 668.552 → 102.280 | 17.311 → 19.334 |
| 93k-matched | 92777 / 128 → 92777 / 128 | 6.243 → 0.000 | 1118.323 → 0.204 | 1153.306 → 17.399 | 385.931 → — |

The short copy seed overlapped compilation; it establishes cache history and is excluded from causal before/after performance claims. All successful handoff rows report zero request-owned cache layers after capture.

| Case | Sampled peak active GB | Sampled peak footprint GB | After-release active GB | After-release footprint GB |
| --- | --- | --- | --- | --- |
| short-cold | 18.014 → 17.828 | 19.200 → 19.603 | 17.341 → 17.344 | 18.932 → 18.685 |
| 46k-cold | 26.899 → 25.375 | 29.900 → 27.885 | 20.193 → 20.535 | 29.696 → 29.979 |
| 46k-warm | 29.948 → 27.558 | 33.867 → 36.272 | 20.202 → 20.548 | 23.152 → 25.159 |
| 46k-resend | 29.985 → 27.583 | 31.475 → 31.537 | 20.212 → 20.558 | 23.415 → 25.316 |
| 77k-restart | 40.079 → 37.547 | 40.809 → 38.946 | 24.372 → 22.209 | 28.178 → 31.107 |
| 77k-warm | 37.956 → 33.244 | 39.033 → 41.821 | 21.871 → 22.042 | 25.133 → 37.861 |
| 93k-matched | 43.952 → 41.006 | 44.563 → 41.497 | 26.094 → 26.099 | 37.247 → 39.657 |

After-release values are fresh scalar samples approximately one second after request return. They are not idle or SSD-drain guarantees. Peaks are sampled lower bounds; process-lifetime peaks remain separately available in the archived `runs.json`.

| Handoff case | Restored snapshot GB | Prefill checkpoints GB | New leaf GB | Payload mode / GB | Tree GB at release | Pending bytes GB / queued count at release |
| --- | --- | --- | --- | --- | --- | --- |
| short-cold | 0.000 | 0.667 | 0.335 | full / 0.335 | 0.668 | 0.668 / 1 |
| 46k-cold | 0.333 | 3.178 | 3.187 | full / 3.187 | 3.855 | 3.187 / 0 |
| 46k-warm | 3.187 | 3.188 | 3.197 | extension / 0.163 | 3.865 | 0.163 / 0 |
| 46k-resend | 3.197 | 3.197 | 3.206 | extension / 0.163 | 3.874 | 0.163 / 0 |
| 77k-restart | 0.333 | 12.677 | 5.190 | full / 5.190 | 5.523 | 8.369 / 1 |
| 77k-warm | 5.190 | 5.191 | 5.198 | extension / 0.162 | 5.198 | 5.352 / 1 |
| 93k-matched | 3.179 | 11.609 | 6.243 | full / 6.243 | 9.421 | 6.243 / 0 |

Component counts are **not additive**. Tree/pending facts were measured at `releasingRequest` and carried into afterRelease. Pending bytes include the active writer charge; queue count excludes the active writer. Full payloads can alias immutable leaf arrays; extension payloads detach attention suffixes and recurrent state.

| Case | Copy request ID | Handoff request ID | Request hash matches | Response hash matches | Cached prompt tokens (copy → handoff) |
| --- | --- | --- | --- | --- | --- |
| short-cold | A20BE1F0-450C-45FF-B096-BC3B0A5A1157 | 804AFAEC-7FF6-4ACF-AB9F-DD60ADA03E20 | yes | yes | 0 → 0 |
| 46k-cold | CA8F4B5B-1487-4AEA-A12D-12E8975A122C | 4B9C3B76-4934-43AB-9A21-8D4E9E7E81C5 | yes | yes | 2735 → 2735 |
| 46k-warm | C0221A7E-6C4F-4249-8C81-EA7D844CFBE6 | 34A5430C-D57B-4512-B7C8-1DDBAD4A33EE | yes | yes | 46282 → 46282 |
| 46k-resend | FDC755FF-7F47-4FFD-A5E3-156DCA2BA3A4 | A4ADD4FE-769C-485E-B024-94A6D3908BC6 | yes | yes | 46427 → 46427 |
| 77k-restart | 03BAE730-1402-41E2-BD5D-2C982D8FE224 | 5A091F08-D4F8-4C74-90A3-6979D45BB031 | yes | yes | 2735 → 2735 |
| 77k-warm | 7E7C48E5-7FAB-4B8C-9E7A-11AF16E65F18 | 08C55D68-02B4-44AC-88D3-48A844065584 | yes | no | 76842 → 76842 |
| 93k-matched | C1046C03-3DB3-443C-8A65-404E0BC7EDAA | 31026874-06BB-43D7-B70B-CC246CE9CD26 | yes | unavailable | 46155 → 46155 |

Response hashes include generated tool-call IDs; an ID mismatch alone is not evidence of different model output. The separate loaded-model gate compares raw continuation logit bytes.

The matched 93k handoff timeline was recovered across diagnostics rotation with all 392 memory samples continuous. Its client wall time and response hash are unavailable; prompt/cached and generation counts come from lookup and emittedPathRegister diagnostics. See README.md for recovery provenance.

## Cancellation and resend

| Mode | Cancel request ID | Signal origin | Signal → terminal ms | After-release active GB | After-release footprint GB |
| --- | --- | --- | --- | --- | --- |
| copy-46k-cancel | 4E12D3E1-0434-49C5-9719-4D2E4BC6B812 | caller | 99.358 | 20.202 | 23.348 |
| handoff-46k-cancel | E4623A44-AE27-4EB7-ABD4-01B31BB860DE | caller | 154.449 | 20.548 | 25.244 |

Neither cancelled request captures a leaf. The following resend uses the identical request hash and is included above. One cancel/resend cycle does not establish behavior for every cancellation phase or reproduce the historical minute-long cleanup incident.
