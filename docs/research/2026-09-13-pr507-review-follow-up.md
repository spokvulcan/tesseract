# PR #507 review follow-up

The supplied review was checked against commit `79e771c7` and the current CI
workflow. The changes below preserve the published benchmark captures and their
source provenance. Production measurements in the original reports describe
those captures; this follow-up does not claim another production-model run.

| Review finding | Disposition |
| --- | --- |
| CI build/tests skipped | Confirmed. The workflow requires a `run-build` label event, not just the label's presence. Request a run after pushing the follow-up. |
| Scratch directory survives failure | Confirmed with a queued SSD write. The benchmark now captures the operation's result, awaits the writer flush on both success and failure, removes its directory, then returns or rethrows. A synchronous `defer` alone cannot await that flush. Removal errors also fail the gate. Forced process termination cannot execute Swift cleanup. |
| Write-error behavior undocumented | Added to the PR description: throwing `FileHandle.write(contentsOf:)` routes write-time errors through existing disk-full classification and eviction/retry handling. |
| Cancelled startup sends no response | Added the same best-effort 503 cancellation response used before lease acquisition. The review's shutdown example needs qualification: `HTTPServer.stopAndDrain` already cancels each transport before cancelling its handler. A new driver test covers parent cancellation while the client lifecycle remains connected and verifies late-handle drain. It does not assert HTTP wire bytes. |
| Evidence obscures source changes | Added a source-first review guide and a separate follow-up commit. Published history and immutable captures stay together so the source-to-evidence links remain usable. |
| Mixed cache-clear APIs | Replaced the two adjacent deprecated aliases with `Memory.clearCache()`. |
| String-based cache-fact reset | Replaced the implicit zero-count sentinel with `markCacheReleased`; the rewind caller and byte-fact regression use that operation explicitly. |
| Eager allocation facts | The `recordAllocation` argument is now a nonescaping autoclosure, evaluated only after the enabled check. A regression verifies disabled calls do not evaluate it. Facts prepared earlier by a caller remain that caller's cost. |
| Python issues | Future inventory captures save `app.log` inside the output directory. Clarified the oversized rusage buffer comment. Parenthesized the diagnostic rotation condition and reused one stat result; existing Python precedence already produced the intended condition. |
| Testing docs and domain language | Removed campaign narrative from the run instructions and linked the existing research reports. “Bounded cache parity” describes a validation mode, with its scope documented in `docs/testing.md`; it does not introduce a product/domain concept requiring a `CONTEXT.md` entry. |

## Validation

The new scratch-store test first ran against success-only cleanup. Failure and
cancellation both left the directory present and the queued descriptor
uncommitted. After the fix, all three cases passed, including the empty pending
queue, committed descriptor and absent directory assertions.

The focused Debug invocation passed **51 test functions in seven suites**:

```bash
xcodebuild test -project tesseract.xcodeproj -scheme tesseract \
  -destination 'platform=macOS' -skipPackagePluginValidation \
  -parallel-testing-enabled NO \
  -only-testing:tesseractTests/BoundedCacheParityTests \
  -only-testing:tesseractTests/RequestMemoryTelemetryTests \
  -only-testing:tesseractTests/CompletionDeliveryTests \
  -only-testing:tesseractTests/CompletionHandlerTests \
  -only-testing:tesseractTests/StreamLifecycleDriverTests \
  -only-testing:tesseractTests/ServerCompletionDrainTests \
  -only-testing:tesseractTests/CacheStateBytesTests
```

Changed Swift files passed formatting and SwiftLint (existing warnings remain).
Both Python runners passed syntax and dry-plan checks without launching a model
or creating an evidence directory. The OS probe smoke check populated process
resident/footprint counters and read pressure/swap successfully. GitHub build and
test results are separate from these local checks and should be read on the PR.
