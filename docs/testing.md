# Testing

Tests use the Swift `Testing` framework (not XCTest), in `tesseractTests/`. Run
before committing changes to server, caching, or agent engine code.

## Unit / integration suites

```bash
# Server + agent suites (recommended for fast, focused runs):
xcodebuild test -project tesseract.xcodeproj -scheme tesseract -destination 'platform=macOS' \
  -only-testing:tesseractTests/HTTPPrefixCacheSpikeTests \
  -only-testing:tesseractTests/HTTPPrefixCacheSessionReplayTests \
  -only-testing:tesseractTests/CompletionHandlerTests \
  -only-testing:tesseractTests/MessageConverterTests \
  -only-testing:tesseractTests/OpenAITypesTests \
  -only-testing:tesseractTests/AgentEngineToolSpecTests \
  -only-testing:tesseractTests/EditToolTests

# Prefix cache suites (radix tree + hybrid snapshot + stable prefix detector):
xcodebuild test -project tesseract.xcodeproj -scheme tesseract -destination 'platform=macOS' \
  -only-testing:tesseractTests/HybridCacheSnapshotTests \
  -only-testing:tesseractTests/TokenRadixTreeTests \
  -only-testing:tesseractTests/StablePrefixDetectorTests \
  -only-testing:tesseractTests/PrefixCacheManagerTests \
  -only-testing:tesseractTests/PrefixCacheIntegrationTests \
  -only-testing:tesseractTests/CheckpointCaptureTests \
  -only-testing:tesseractTests/StablePrefixDetectorNonDeterminismTests \
  -only-testing:tesseractTests/JinjaNonDeterminismReproTests

# Run all tests:
xcodebuild test -project tesseract.xcodeproj -scheme tesseract -destination 'platform=macOS' \
  -only-testing:tesseractTests
```

## Loaded-model verification

Not unit tests — these run against a real model.

```bash
scripts/dev.sh prefix-cache-e2e          # PrefixCacheE2ERunner — TTFT/output equivalence proxy
scripts/dev.sh hybrid-cache-correctness  # HybridCacheCorrectnessRunner — bitwise logit + state equivalence

TESSERACT_ENABLE_DECODE_RATE_TESTS=1 \
  xcodebuild test -project tesseract.xcodeproj -scheme tesseract -destination 'platform=macOS' \
  -only-testing:tesseractTests/TriAttentionMoEDecodeRateTests
```

Both exit non-zero on any failed check. Run before releases and after any change
to `LLMActor`, `PrefixCacheManager`, `HybridCacheSnapshot`, or
`StablePrefixDetector`. The correctness runner is the stronger gate (bitwise
tensor comparison via raw `ModelContainer.perform` access); the e2e runner
exercises the full HTTP path and is the right shape for catching pipeline
regressions the correctness runner can't see.
