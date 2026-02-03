# Repository Guidelines

## Project Structure & Module Organization
- `whisper-on-device/` contains the app source code, organized by feature and core services.
  - `App/` app entry points and dependency setup
  - `Core/` audio capture, text injection, permissions
  - `Features/` dictation, transcription, settings
  - `Models/` shared data types
- `Assets.xcassets/` holds app images and asset catalogs.
- `whisper-on-deviceTests/` and `whisper-on-deviceUITests/` contain unit and UI tests.
- `whisper-on-device.xcodeproj` is the Xcode project.

## Build, Test, and Development Commands
- `xcodebuild build -project whisper-on-device.xcodeproj -scheme whisper-on-device` builds the app from the command line.
- `xcodebuild test -project whisper-on-device.xcodeproj -scheme whisper-on-device` runs tests (currently optional during MVP development).
- `open whisper-on-device.xcodeproj` opens the project in Xcode.

## Build Verification Requirement
- Always run `xcodebuild build -project whisper-on-device.xcodeproj -scheme whisper-on-device` after any code change.
- If the build fails, fix the errors before replying. Do not leave build errors unresolved.

## Coding Style & Naming Conventions
- Swift 6.2 + SwiftUI; follow Xcode’s default formatting (4-space indentation).
- Type names use UpperCamelCase; properties and functions use lowerCamelCase.
- File names match their primary type (e.g., `SettingsView.swift`). SwiftUI views typically end with `View`.
- No enforced formatter or linter; keep changes consistent with nearby code.

## Testing Guidelines
- Tests use Swift’s `Testing` framework, not XCTest.
- Place tests in `whisper-on-deviceTests/` and UI tests in `whisper-on-deviceUITests/`.
- Name files `*Tests.swift` and keep test names descriptive and scenario-based.
- During the MVP phase, prioritize successful builds; run tests when changing core logic.

## Commit & Pull Request Guidelines
- Commit messages in this repo are short, imperative, and sentence case (e.g., “Refine dictation UI”).
- PRs should include a brief summary, testing status (commands run or “not run”), and screenshots or recordings for UI changes.
- Link related issues when applicable.

## Agent-Specific Instructions
- See `CLAUDE.md` for architecture notes, build guidance, and AI agent expectations.
