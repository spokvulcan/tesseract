## Proposed changes

`KVCacheSimple` and `QuantizedKVCache` currently grow in fixed 256-row increments. Repeated appends therefore concatenate and copy the cache many times. This adds an optional `KVCache.reserveCapacity(_:)` request for a known total prompt length and doubles allocation increments from 256 rows to a 4096-row cap for subsequent growth.

Reservation is applied on the next update, when tensor shapes are known, without changing the offset or logical state. Known prompt reservations round using the initial granule. A pending reservation survives `copy()` and conversion between simple and quantized caches until its rows are written. CacheList forwards the request; other cache types keep their existing allocation policy. `state`, `metaState`, trim and prompt-cache serialization retain their existing formats and logical contents. The public simple-cache `step` continues to set the initial granule.

The new `CacheCapacityTests` covers chunked-prompt reservation, doubling and the cap, restore, trim/copy/persistence, dynamic quantization, and nested CacheList forwarding. With a synthetic single-row decode, capacities advance through 256, 768, 1792, 3840, 7936 and 12032 rows. After the cap, increments remain 4096 rows; growth is not logarithmic indefinitely. This is allocation-capacity evidence, not a production throughput or footprint claim.

Validation on vanilla upstream `c6446cf` (still upstream `main` as of 2026-09-21) with upstream dependency pins: the six new `CacheCapacityTests` and the existing empty-prompt-cache legacy-format, quantized-copy and empty-simple-to-quantized metadata tests pass under `xcodebuild test`, and `pre-commit run --all-files` passes. No integration tests or loaded-model campaign were run for this patch. The same commit has been carried in the downstream app's vendor pin since 2026-09-19 (spokvulcan/tesseract#533), where the app's prefix-cache suites run against it.

## Checklist

- [ ] I have read the [CONTRIBUTING](https://github.com/ml-explore/mlx-swift-lm/blob/main/CONTRIBUTING.md) document
- [x] I have run `pre-commit run --all-files` to format my code / installed pre-commit prior to committing changes
- [x] I have added tests that prove my fix is effective or that my feature works
- [x] I have updated the necessary documentation (if needed)

## AI usage

- [ ] I have read this PR description in full and approve it as my own, and it accurately describes the code changes.
- AI usage disclosure: OpenAI Codex implemented the patch, wrote the tests and the first draft of this description and performed independent Standards and Spec reviews; Claude Code re-ran the test and formatting checks on 2026-09-21 and revised this description. The human contributor reviews the code and description and completes the unchecked attestations before submission.
