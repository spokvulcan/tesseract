# Repository Guidelines

## Project Structure & Module Organization
- `tesseract/` contains the app source code, organized by feature and core services.
  - `App/` app entry points and dependency setup
  - `Core/` audio capture, text injection, permissions
  - `Features/` dictation, transcription, settings
  - `Models/` shared data types
- `Assets.xcassets/` holds app images and asset catalogs.
- `tesseractTests/` and `tesseractUITests/` contain unit and UI tests.
- `tesseract.xcodeproj` is the Xcode project.

## Build, Test, and Development Commands
- `xcodebuild build -project tesseract.xcodeproj -scheme tesseract` builds the app from the command line.
- `xcodebuild test -project tesseract.xcodeproj -scheme tesseract` runs tests (currently optional during MVP development).
- `open tesseract.xcodeproj` opens the project in Xcode.

## Build Verification Requirement
- Always run `xcodebuild build -project tesseract.xcodeproj -scheme tesseract` after any code change.
- If the build fails, fix the errors before replying. Do not leave build errors unresolved.

## Coding Style & Naming Conventions
- Swift 6.2 + SwiftUI; follow Xcode's default formatting (4-space indentation).
- Type names use UpperCamelCase; properties and functions use lowerCamelCase.
- File names match their primary type (e.g., `SettingsView.swift`). SwiftUI views typically end with `View`.
- No enforced formatter or linter; keep changes consistent with nearby code.

## Testing Guidelines
- Tests use Swift's `Testing` framework, not XCTest.
- Place tests in `tesseractTests/` and UI tests in `tesseractUITests/`.
- Name files `*Tests.swift` and keep test names descriptive and scenario-based.
- During the MVP phase, prioritize successful builds; run tests when changing core logic.

## Commit & Pull Request Guidelines
- Commit messages in this repo are short, imperative, and sentence case (e.g., “Refine dictation UI”).
- PRs should include a brief summary, testing status (commands run or “not run”), and screenshots or recordings for UI changes.
- Link related issues when applicable.

## Agent-Specific Instructions
- See `CLAUDE.md` for architecture notes, build guidance, and AI agent expectations.
