# overlay-focus-hang-lab — RUNBOOK

The runtime harness for the 2026-09-15 freeze: after the first committed
take, the dictation overlay swapped its proofreading dots for the beat pill
with two buttons, and the main thread never returned from layout. Sampling
showed SwiftUI's `FocusBridge.updateDefaultKeyViewLoop` spinning inside
`NSHostingView.layout()`. This lab reproduces that with no app code: a
borderless non-activating `NSPanel` hosting a SwiftUI view that switches to
content containing buttons one second after launch, plus a watchdog thread
that reports whether the main thread still serves blocks afterwards.

## Running

```sh
swift run --package-path tools/overlay-focus-hang-lab overlay-focus-hang-lab [flags]
```

Exit code 0 means the main thread stayed responsive; 2 means it hung (the
process is then killed by the watchdog, so the run always finishes in about
seven seconds). No flags reproduces the overlay as shipped before the fix.

| flag | what it changes |
|---|---|
| `--unfocusable` | the app's mitigation: `.focusable(false)` on the buttons (`overlayAffordance()`) |
| `--tap-gesture` | images with tap gestures instead of buttons (also green) |
| `--no-buttons` | no focusable control at all (green) |
| `--buttons-from-start` | buttons present at first layout (green) |
| `--plain-window` | a titled `NSWindow` instead of the panel (still hangs: the panel is not the trigger) |
| `--hosting-controller` | `NSHostingController` as content view controller |
| `--no-glass`, `--no-animation`, `--bordered`, `--one-button`, `--no-spacer` | single-variable minimisation knobs; the hang is timing-sensitive, so treat their results per OS build |
| `--no-autorecalc`, `--cannot-become-key` | window-level knobs that do not help |

Measured on macOS 27.0 (26A428), Xcode 27.0: no flags hangs 5/5 runs,
`--unfocusable` and `--tap-gesture` pass 3/3. Re-run without flags after an
OS update to learn whether the mitigation is still needed.
