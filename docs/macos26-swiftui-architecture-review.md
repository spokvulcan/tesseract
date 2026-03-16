# Tesseract architecture review for Swift 6.2 / SwiftUI on macOS 26

Research date: 2026-03-16  
Scope: current code in this repository, compared against official Apple and Swift guidance for macOS 26 on Apple Silicon only.

## Executive summary

Tesseract is already aligned with several important macOS 26 and Apple Silicon best practices:

- The app is explicitly Apple-Silicon-only (`Config/AppleSilicon.xcconfig` sets `ARCHS = arm64`).
- The project target is already on `MACOSX_DEPLOYMENT_TARGET = 26.0` and `SWIFT_VERSION = 6.2`.
- The codebase uses a feature-first folder structure.
- Heavy ML work is isolated behind actors (`WhisperActor`, `LLMActor`, `TTSActor`) while UI-facing orchestration stays on `@MainActor`.
- The main app UI uses `NavigationSplitView`, which remains the right high-level container for a Mac sidebar app.
- The on-device ML stack is well matched to Apple Silicon: WhisperKit/Core ML for ASR, MLX for local LLM/TTS, and explicit model unload paths to manage unified memory pressure.

The biggest architectural gaps against current SwiftUI / Swift 6.2 guidance are not in the ML stack. They are in app structure and UI data flow:

1. The app still uses a mixed Observation + `ObservableObject` + Combine + singleton model, and the migration scope is larger than the first draft suggested.
2. `WindowGroup` is being used like a singleton window, and the app currently carries multiple workarounds to suppress normal multi-window behavior.
3. `DependencyContainer` is doing too much orchestration, especially inside agent bootstrap, rather than acting purely as a composition root.
4. The environment injection surface is too broad, but tightening it should happen after the Observation migration, not before.
5. The single app target is large, but extracting `Agent` into a separate package is premature until the observation/data-flow cleanup reveals real boundaries.

My conclusion: the current architecture is good enough to keep shipping, but the next phase should focus on three practical changes in this order: use a true single-window scene, migrate the non-Agent app state to Observation, and pull bootstrap/platform concerns out of the container. It should not add a separate Settings window, an Agent Swift package, or UI automation yet.

## What the current project looks like

### Project baseline

- `tesseract.xcodeproj` is configured for `Swift 6.2` and `macOS 26.0`.
- `Config/AppleSilicon.xcconfig` forces `arm64` only.
- App entry is split between `tesseract/App/TesseractApp.swift` and `tesseract/App/AppDelegate.swift`.
- There is a single main app target plus vendor packages; there is no internal Swift package or framework boundary for the app’s own code.

### Primary files inspected

The review is based mainly on these implementation files:

- `tesseract/App/TesseractApp.swift`
- `tesseract/App/AppDelegate.swift`
- `tesseract/App/DependencyContainer.swift`
- `tesseract/Core/ViewModifiers.swift`
- `tesseract/Core/Audio/AudioCaptureEngine.swift`
- `tesseract/Core/TextInjection/HotkeyManager.swift`
- `tesseract/Core/Permissions/PermissionsManager.swift`
- `tesseract/Features/Dictation/DictationCoordinator.swift`
- `tesseract/Features/Speech/SpeechCoordinator.swift`
- `tesseract/Features/Speech/SpeechEngine.swift`
- `tesseract/Features/Transcription/TranscriptionEngine.swift`
- `tesseract/Features/Transcription/TranscriptionHistory.swift`
- `tesseract/Features/Agent/AgentCoordinator.swift`
- `tesseract/Features/Agent/AgentEngine.swift`
- `tesseract/Features/Agent/Core/Agent.swift`
- `tesseract/Features/Settings/SettingsManager.swift`
- `tesseract/Features/Settings/SettingsView.swift`
- `tesseract/Features/Dictation/Views/ContentView.swift`
- `tesseract/Features/Dictation/Views/MainWindowView.swift`
- `tesseract/Features/Agent/Views/AgentContentView.swift`

### Source layout snapshot

Swift file counts in the app target:

- `App`: 8 files
- `Core`: 12 files
- `Features`: 109 files
- `Models`: 15 files

Feature distribution:

- `Agent`: 66 files
- `Dictation`: 15 files
- `Speech`: 13 files
- `Transcription`: 4 files
- `Settings`: 3 files
- `ImageGen`: 4 files
- `Models`: 4 files

Observation from the codebase: the Agent feature is now the architectural center of gravity. It represents about 61% of all files under `Features/`.

### Runtime architecture

The app uses a clear composition root:

- `DependencyContainer` wires the full graph.
- Coordinators (`DictationCoordinator`, `SpeechCoordinator`, `AgentCoordinator`) orchestrate user-facing flows.
- Engines (`TranscriptionEngine`, `SpeechEngine`, `AgentEngine`) wrap inference runtimes.
- Actors (`WhisperActor`, `TTSActor`, `LLMActor`, `ContextManager`) isolate non-Sendable or heavy mutable state.

This is a reasonable architecture for a macOS app that mixes SwiftUI, AppKit, global hotkeys, floating panels, and on-device inference.

Where the architecture is becoming strained:

- `DependencyContainer.swift` is 398 lines and injects a broad environment graph.
- `AgentCoordinator.swift` is 985 lines.
- `SettingsView.swift` is 396 lines.
- `SettingsManager.shared` is referenced broadly across views and AppKit controllers.
- The project currently contains 20 `ObservableObject` types and 7 `@Observable` types, which indicates a partial migration rather than a settled model.

### Verified Observation migration scope

Follow-up inspection shows the migration scope is substantial:

| Pattern | Verified count |
| --- | ---: |
| `ObservableObject` types | 20 |
| `@Observable` types | 7 |
| `@Published` properties | 47 across 18 files |
| `@EnvironmentObject` declarations in views | 30 |
| `SettingsManager.shared` direct accesses | 21 across 13 files |

This matters because the Agent feature has already largely crossed to the new model, while most of the rest of the app has not.

## Official best-practice baseline

### Swift 6.2

The latest official Swift feature release I found is Swift 6.2. Swift.org describes it as a release focused on productivity, concurrency, and performance. It specifically calls out:

- approachable concurrency
- optional default main-actor isolation for executable targets
- the new `@concurrent` attribute
- a modern typed `NotificationCenter` API
- transactional `Observations` async sequences for observable state

Sources:

- [Swift 6.2 Released](https://www.swift.org/blog/swift-6.2-released/)
- [What’s new in Swift](https://developer.apple.com/swift/whats-new/)

### Observation and SwiftUI data flow

Apple’s Observation guidance is now very clear:

- use `@Observable` for model types
- use `@State` when the view owns the model
- use `@Environment` when the model belongs in app-wide/shared environment
- use `@Bindable` when the view only needs bindings into an observable model
- remove `ObservableObject`, `@Published`, and many older object wrappers when migrating

Apple also explicitly says this model improves performance because SwiftUI tracks property access more precisely than the older object-wide invalidation model.

For non-view code, Swift 6.2 also provides an official `Observations` async-sequence API and `withObservationTracking`. These are the relevant modern replacement patterns for some of the repo’s current Combine `.sink` usage outside SwiftUI views.

Sources:

- [Discover Observation in SwiftUI](https://developer.apple.com/videos/play/wwdc2023/10149/)
- [Observations](https://developer.apple.com/documentation/observation/observations)
- [withObservationTracking(_:onChange:)](https://developer.apple.com/documentation/observation/withobservationtracking(_:onchange:))

### SwiftUI and concurrency on macOS 26

Apple’s current SwiftUI concurrency guidance matters directly for this project:

- SwiftUI treats the main actor as the default for app and view code.
- Some rendering-related APIs may call your code off the main thread.
- `Shape`, `Layout`, `visualEffect`, and geometry-related closures can be evaluated away from the main actor.
- SwiftUI’s concurrency annotations reflect runtime behavior, not just syntax convenience.

For a Mac app with animations, overlays, async inference, and custom rendering, this means actor isolation must stay explicit and intentional.

Sources:

- [Explore concurrency in SwiftUI](https://developer.apple.com/videos/play/wwdc2025/266/)
- [What’s new in Swift](https://developer.apple.com/swift/whats-new/)

### Scene architecture and macOS conventions

Apple’s scene guidance still matters on macOS 26:

- `WindowGroup` is for window families and supports multiple instances on macOS.
- `Settings` is the dedicated scene for app preferences and automatically wires the standard Preferences command and window treatment.
- SwiftUI’s scene model now includes `Window`, `WindowGroup`, `Settings`, `MenuBarExtra`, and other specialized scene types.

Sources:

- [Scene](https://developer.apple.com/documentation/swiftui/scene)
- [App essentials in SwiftUI](https://developer.apple.com/videos/play/wwdc2020/10037/)
- [Bringing multiple windows to your SwiftUI app](https://developer.apple.com/documentation/SwiftUI/bringing-multiple-windows-to-your-swiftui-app)
- [MenuBarExtra](https://developer.apple.com/documentation/swiftui/menubarextra)

### SwiftUI 26 platform direction

For macOS 26 specifically, Apple’s current guidance emphasizes:

- letting navigation containers adopt the new design naturally
- using `NavigationSplitView` for split navigation
- taking advantage of improved list and scrolling performance on macOS
- using the new SwiftUI performance instrument in Instruments 26
- allowing SwiftUI window and toolbar styling APIs to do more of the window work

Sources:

- [What’s new in SwiftUI](https://developer.apple.com/videos/play/wwdc2025/256/)
- [Build a SwiftUI app with the new design](https://developer.apple.com/videos/play/wwdc2025/323/)
- [Optimize SwiftUI performance with Instruments](https://developer.apple.com/videos/play/wwdc2025/306/)
- [Customizing window styles and state-restoration behavior in macOS](https://developer.apple.com/documentation/SwiftUI/Customizing-window-styles-and-state-restoration-behavior-in-macOS)

### Apple Silicon ML guidance

Apple’s current ML guidance strongly supports the choices this project already made:

- Core ML is optimized for on-device performance by leveraging Apple Silicon and minimizing memory footprint and power consumption.
- MLX is built for Apple Silicon, can run on CPU or GPU, and benefits from unified memory.

Sources:

- [Core ML overview](https://developer.apple.com/machine-learning/core-ml/)
- [Get started with MLX for Apple silicon](https://developer.apple.com/videos/play/wwdc2025/315/)
- [What’s new in Machine Learning & AI](https://developer.apple.com/wwdc25/guides/machine-learning/)

## Comparison: current project vs current best practices

### 1. Scene architecture and windowing

Current state:

- The app uses `WindowGroup(id: "main")` for its main window.
- It removes the “New Window” command manually.
- It also manually deduplicates windows on launch and during reopen flows.
- Settings are intentionally routed into the main split-view UI.

Assessment:

- `NavigationSplitView` itself is aligned.
- The scene model around it is only partially aligned.

Why this matters:

- Apple’s scene model expects `WindowGroup` to support multiple windows on macOS.
- The current code is effectively fighting `WindowGroup` semantics to preserve a single main window.
- That creates extra AppDelegate logic, reopen logic, and launch dedup logic that would be unnecessary in a more scene-native design.

Recommendation:

- If the product intent is one main app window, use `Window(id:)` rather than `WindowGroup(id:)`.
- In this repo, that is not a conceptual tweak; it is the correct scene type for the intended UX.
- Keep settings in the main-window sidebar unless the product specifically wants a separate Preferences window. Do not do both.
- Keep AppKit only where SwiftUI still does not cover the requirement: event taps, panels, special overlay windows, and menu-bar edge cases.

Concrete repo implication:

- `Window("Tesseract", id: "main")` should let the app delete most of the current duplicate-window workarounds:
  - `WindowOpenerView` dedup logic
  - `.handlesExternalEvents(matching: Set<String>())`
  - `CommandGroup(replacing: .newItem) {}`
  - reopen race-avoidance logic
  - part of the tracked main-window bookkeeping
- `WindowOpenerView` itself likely stays, but in a smaller form: it still provides the `openWindow(id:)` bridge from SwiftUI into `AppDelegate`.
- With `Window`, macOS preserves the single window instance more naturally across close/reopen, so the existing `hasSetup` guard in `DependencyContainer` remains useful and should not be removed casually.

### 2. Observation and UI data flow

Current state:

- `AgentCoordinator`, `OverlayState`, and some notch/overlay models already use `@Observable`.
- Much of the rest of the app still uses `ObservableObject`, `@Published`, `@EnvironmentObject`, `@ObservedObject`, and Combine publishers.
- `SettingsManager` is a global singleton and is read directly by many views and controllers.

Assessment:

- Partially aligned, but behind the current SwiftUI direction.

Why this matters:

- Apple’s current guidance is Observation-first.
- The mixed model increases cognitive load: some state invalidates precisely, some invalidates coarsely, and some changes are routed through `NotificationCenter` or `UserDefaults.didChangeNotification`.
- The singleton settings model also weakens testability and feature isolation.

Recommendation:

- Treat Observation migration as a first-class architectural cleanup task.
- Start with `SettingsManager`, because it drives the singleton pattern and the `UserDefaults.didChangeNotification` chains in `DependencyContainer`.
- After that, migrate low-friction types first: `SpeechCoordinator`, `SpeechEngine`, `TranscriptionEngine`, and `AgentEngine`.
- Then migrate the dictation/audio stack as a coordinated change: `DictationCoordinator`, `AudioCaptureEngine`, and `TranscriptionHistory`, along with their app-layer consumers.
- Migrate `AudioCaptureEngine` with care because it feeds AppKit consumers; keep an explicit bridge for overlay/panel consumers rather than forcing everything through SwiftUI observation.
- Leave lower-traffic engines such as image generation and download management for later.
- Replace `@EnvironmentObject` with typed `@Environment(SomeType.self)` where possible.
- Use `@Bindable` in editing views rather than passing shared singleton objects deeply.

Inference from the repo: this is the single highest-leverage cleanup because it simplifies view code, reduces broad invalidation, and should remove both `UserDefaults.didChangeNotification` observation chains from the container.

#### Hard blocker: `SettingsManager` cannot keep `@AppStorage`

This is not just a design caveat. It is a hard migration blocker.

- `SettingsManager.swift` currently contains 29 `@AppStorage` properties.
- Local verification against the installed toolchain (`Apple Swift version 6.2.4`) shows that combining `@Observable` and `@AppStorage` fails to type-check.
- The compiler error is an invalid redeclaration of the synthesized backing property (for example, `_flag`).

Implication:

- Migrating `SettingsManager` to Observation requires converting those 29 settings from `@AppStorage`-wrapped properties to plain stored `var`s with explicit `UserDefaults` load/save behavior.
- This is the dominant cost of the `SettingsManager` migration and should be planned as a single coordinated change, not a small macro-only refactor.
- The SwiftUI side of that migration also requires `@Bindable` in settings-heavy views after the environment model changes.
- The current feature views already contain 20 direct `$settings...` binding sites that will need that new binding pattern.

Practical migration shape:

- initialize stored settings from `UserDefaults` in `init`
- write back in `didSet` or dedicated persistence helpers
- keep side effects such as launch-at-login and dock-policy updates where they already belong

SwiftUI binding shape after the migration:

```swift
// Before
@ObservedObject private var settings = SettingsManager.shared

// body uses bindings like:
Toggle("Play sounds", isOn: $settings.playSounds)

// After
@Environment(SettingsManager.self) private var settings

var body: some View {
    @Bindable var settings = settings

    Toggle("Play sounds", isOn: $settings.playSounds)
}
```

That `@Bindable var settings = settings` rebinding inside `body` is the key change for the 20 current `$settings...` binding sites.

Caveat:

- `@AppStorage` automatically reflects external `UserDefaults` mutations; manual storage will not unless a re-sync path is added.
- For this app, that appears to be a low-priority concern because settings are primarily mutated from inside the app.

#### Current Combine seam that blocks piecemeal dictation/audio migration

The dictation/audio migration is not as independent as the first draft implied.

The hardest seam is in the overlay panel APIs:

- `OverlayPanelController.setup()` accepts `Published<DictationState>.Publisher` and `Published<Float>.Publisher`
- `FullScreenBorderPanelController.setup()` accepts the same typed publisher inputs

Additional Combine consumers in the current app layer:

| Source type | Property | Consumer | Friction |
| --- | --- | --- | --- |
| `DictationCoordinator` | `$state` | `OverlayPanelController` | High |
| `DictationCoordinator` | `$state` | `FullScreenBorderPanelController` | High |
| `DictationCoordinator` | `$state` | `AppDelegate` | Medium |
| `AudioCaptureEngine` | `$audioLevel` | `OverlayPanelController` | High |
| `AudioCaptureEngine` | `$audioLevel` | `FullScreenBorderPanelController` | High |
| `AudioCaptureEngine` | `$audioLevel` | `DependencyContainer` | Medium |
| `TranscriptionHistory` | `$entries` | `MenuBarManager` | Medium |

This changes the migration order:

- low-friction Observation migrations should happen first
- dictation/audio should then migrate as a coordinated change that also rewrites the panel-controller setup APIs
- the panel controllers should stop depending on `Published<T>.Publisher` and instead depend on observable state or a dedicated bridge layer

### 3. Concurrency and actor isolation

Current state:

- Strong use of `@MainActor` for app/coordinator state.
- Good use of actors around WhisperKit, MLX LLM, TTS, and context compaction.
- Real-time audio thread state is isolated using locks and explicit `@unchecked Sendable` wrappers.
- Async streaming is modeled with `AsyncStream` / `AsyncThrowingStream`.

Assessment:

- Strongly aligned.

What is especially good:

- The code isolates the unsafe or non-Sendable edges instead of spreading locks throughout UI logic.
- Inference engines stay behind actors.
- The app deliberately unloads large models to manage unified memory pressure on Apple Silicon.

Remaining opportunities:

- There are still many `DispatchQueue.main.async` / `asyncAfter` calls across AppKit bridges and overlay code.
- Swift 6.2’s newer isolation defaults and concurrency configurations are not yet visibly reflected in project settings or architecture docs.
- The app still uses notification-publisher patterns in places where Swift 6.2’s newer typed notification APIs could gradually replace stringly notifications.

Recommendation:

- Keep the actor-based ML boundary exactly in spirit.
- Do not prioritize replacing the overlay/AppKit timing hacks just for style. Many of those delayed main-thread hops are pragmatic UI timing code, not the highest-value debt in this repo.
- Use official Swift 6.2 Observation APIs in non-view code where they fit:
  - `Observations { ... }` for async sequence observation
  - `withObservationTracking` for synchronous dependency capture
- Adopt typed notification patterns for new code paths.
- Document which closures are intentionally `Sendable` because SwiftUI may run them off-main.

Future-facing note:

- Foundation now exposes `NotificationCenter.MainActorMessage`.
- That is a better long-term direction for app-local messages such as onboarding display than custom stringly notifications, but it is not urgent for this migration.

### 4. Dependency management and modularity

Current state:

- `DependencyContainer` is a clear composition root and that is good.
- Constructor injection is used inside coordinators and engines.
- But the container also exposes the entire graph broadly to the environment.
- `injectDependencies(from:)` currently injects 14 container-owned objects into the view hierarchy.
- The `agent` lazy property contains a large multi-step bootstrap sequence rather than simple dependency wiring.
- `DependencyContainer.setup()` still owns Combine-based settings observation and notification plumbing.
- The app is still a single internal module.

Assessment:

- Good local structure, but the next cleanup is inside the app target, not across target boundaries.

Why this matters:

- A single target is still acceptable for an MVP-ish app.
- It becomes less acceptable when one feature starts dominating complexity.
- Tesseract is now beyond that point, mainly because of Agent.

Recommendation:

- Keep `DependencyContainer`, but stop letting it become the only place where orchestration lives.
- Extract agent bootstrap into an `AgentFactory` or `makeAgent()` path so the container goes back to wiring dependencies rather than executing an initialization script.
- Let the Observation migration remove the two `UserDefaults.didChangeNotification` chains from the container rather than preserving them behind new abstractions.
- Defer extracting `Features/Agent` into a Swift package or separate target until the Observation migration is complete and the true dependency boundaries are clearer.
- Minimize direct use of `SettingsManager.shared` in view files so more features can be initialized with explicit dependencies.
- Do not narrow `injectDependencies` until after the main Observation migration; changing the dependency shape and the observation model at the same time would create unnecessary risk.

My view: `DependencyContainer` should remain the composition root, but not the place where bootstrap logic accumulates.

AppKit-side singleton removal is straightforward because these consumers are already container-owned or container-wired:

| Consumer | Current pattern | Likely migration path |
| --- | --- | --- |
| `MenuBarManager` | inline `SettingsManager.shared` access | pass `SettingsManager` via init or setup |
| `AppDelegate` | `SettingsManager.shared.showInDock` | read through stored `container.settingsManager` |
| `OverlayPanelController` | private stored `SettingsManager.shared` | inject via init |
| `FullScreenBorderPanelController` | private stored `SettingsManager.shared` | inject via init |

This means removing the settings singleton from AppKit code is a mechanical cleanup, not an architectural unknown.

### 5. AppKit bridging

Current state:

- The app uses AppDelegate, `NSStatusItem`, custom panels, overlay controllers, event taps, and text injection bridges.

Assessment:

- This is appropriate for the product.

Why:

- Global hotkeys, always-on-top overlays, accessibility-driven text injection, and notch/panel behavior are exactly the kinds of features that still justify AppKit.
- Apple’s current SwiftUI guidance is not “avoid AppKit”; it is “use SwiftUI first, bridge where the platform requires it”.

Recommendation:

- Keep AppKit bridges, but isolate them more explicitly as platform adapters.
- A dedicated `Platform/` or `AppKit/` group would make the architecture easier to reason about than having these pieces spread across `App/` and `Core/`.
- This is a good structural cleanup to do before any package extraction.

### 6. Apple Silicon ML stack

Current state:

- The app is arm64-only.
- ASR uses Core ML-backed WhisperKit.
- LLM and TTS use MLX.
- The app actively unloads competing models to control memory residency.

Assessment:

- Strongly aligned.

This is one of the best parts of the current architecture:

- It matches Apple’s platform direction for on-device AI.
- It is coherent with the product promise of offline inference.
- It is explicitly designed around unified memory pressure, which matters more than generic abstraction purity here.

Recommendation:

- Keep this architecture.
- Invest in profiling and performance observability rather than reworking the ML layer structurally.
- Use Instruments 26, the SwiftUI instrument, and CPU/hang profiling as a regular part of release hardening.

### 7. Testing and verification

Current state:

- Unit tests use Swift Testing, which is current and correct.
- The single unit test file is very large (`tesseractTests.swift` is 1253 lines).
- UI tests are effectively absent (`tesseractUITests.swift` only states they were intentionally removed).

Assessment:

- Core test technology is aligned.
- Test organization and UI coverage are not.

Recommendation:

- Split the large test file into feature-focused Swift Testing suites.
- Prioritize coordinator and engine unit tests over UI automation.
- The Observation migration should make state-machine testing cheaper and more valuable, especially for `DictationCoordinator`, `SpeechCoordinator`, and related flows.
- Defer UI automation until the app’s observable state surface is cleaner and the highest-value coordinator tests exist.

## Priority roadmap

### Phase 1: Ship faster

1. Replace `WindowGroup` with `Window` for the main scene.
2. Migrate `SettingsManager` to an Observation-first model and remove the singleton-driven notification plumbing.
3. Extract agent bootstrap from `DependencyContainer`.

### Phase 2: Modernize

4. Migrate low-friction Observation types first:
   - `SpeechCoordinator`
   - `SpeechEngine`
   - `TranscriptionEngine`
   - `AgentEngine`
5. Rewrite the overlay-panel setup APIs so they no longer require `Published<T>.Publisher`.
   - Both panel controllers already own `OverlayState` (`@Observable`) internally.
   - The natural target is for the controllers to accept the coordinator/engine reference or another observation-friendly input, then push values into `OverlayState` internally instead of receiving typed Combine publishers from outside.
6. Then migrate the dictation/audio stack together:
   - `DictationCoordinator`
   - `AudioCaptureEngine`
   - `TranscriptionHistory`
   - their app-layer consumers in `AppDelegate`, `DependencyContainer`, and `MenuBarManager`
7. After the migrations, narrow environment injection by feature scope instead of injecting the full graph everywhere.
8. Group AppKit-heavy code under a clearer `Platform/` or `AppKit/` group.

### Keep as-is for now

9. Apple-Silicon-only target strategy
10. Actor-based ML boundaries
11. `NavigationSplitView` as the primary main-window container
12. AppKit bridges for hotkeys, panels, overlays, and text injection
13. Most `DispatchQueue.main.asyncAfter` timing code in overlay/AppKit paths

### Do not do yet

14. Extract `Agent` into a separate Swift package
15. Add a separate SwiftUI `Settings` scene
16. Replace all AppKit timing/dispatch code with structured concurrency
17. Invest in UI automation before coordinator-level unit coverage improves

## Bottom line

The current Tesseract architecture is not outdated, but it is transitional.

It already has the right foundations for macOS 26 on Apple Silicon:

- Swift 6.2
- actor-isolated ML
- SwiftUI-first UI
- AppKit only where platform behavior requires it

The next phase should not be “invent a new architecture”. It should be:

- use the right single-window scene type
- finish the Observation migration outside the Agent feature
- move bootstrap and platform code into clearer local boundaries

If those changes happen, the codebase will be much closer to current SwiftUI best practices without paying the cost of premature package extraction, a second Settings window, or early UI automation.

## Local verification notes

These points were verified directly against the local Swift 6.2.4 toolchain, not inferred from commentary:

- `@Observable` plus `@AppStorage` fails to type-check with a synthesized backing-storage collision.
- `Observations` is available in the current toolchain and type-checks as expected.
- `NotificationCenter.MainActorMessage` exists in the current Foundation toolchain.
- `withContinuousObservationTracking` is not present in the local Swift 6.2.4 toolchain, so this review does not depend on it.

## Sources

- [Swift 6.2 Released](https://www.swift.org/blog/swift-6.2-released/)
- [What’s new in Swift](https://developer.apple.com/swift/whats-new/)
- [Discover Observation in SwiftUI](https://developer.apple.com/videos/play/wwdc2023/10149/)
- [Window](https://developer.apple.com/documentation/swiftui/window)
- [AppStorage](https://developer.apple.com/documentation/swiftui/appstorage)
- [Observations](https://developer.apple.com/documentation/observation/observations)
- [withObservationTracking(_:onChange:)](https://developer.apple.com/documentation/observation/withobservationtracking(_:onchange:))
- [Explore concurrency in SwiftUI](https://developer.apple.com/videos/play/wwdc2025/266/)
- [What’s new in SwiftUI](https://developer.apple.com/videos/play/wwdc2025/256/)
- [Build a SwiftUI app with the new design](https://developer.apple.com/videos/play/wwdc2025/323/)
- [Optimize SwiftUI performance with Instruments](https://developer.apple.com/videos/play/wwdc2025/306/)
- [NotificationCenter.MainActorMessage](https://developer.apple.com/documentation/foundation/notificationcenter/mainactormessage)
- [Scene](https://developer.apple.com/documentation/swiftui/scene)
- [MenuBarExtra](https://developer.apple.com/documentation/swiftui/menubarextra)
- [Bringing multiple windows to your SwiftUI app](https://developer.apple.com/documentation/SwiftUI/bringing-multiple-windows-to-your-swiftui-app)
- [App essentials in SwiftUI](https://developer.apple.com/videos/play/wwdc2020/10037/)
- [Customizing window styles and state-restoration behavior in macOS](https://developer.apple.com/documentation/SwiftUI/Customizing-window-styles-and-state-restoration-behavior-in-macOS)
- [Core ML overview](https://developer.apple.com/machine-learning/core-ml/)
- [Get started with MLX for Apple silicon](https://developer.apple.com/videos/play/wwdc2025/315/)
- [What’s new in Machine Learning & AI](https://developer.apple.com/wwdc25/guides/machine-learning/)
