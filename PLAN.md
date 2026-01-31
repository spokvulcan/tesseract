# Whisper-on-Device MVP Implementation Plan

## Overview

A privacy-focused, offline voice-to-text dictation application for macOS using native Swift/SwiftUI. The app captures audio via push-to-talk, transcribes locally using WhisperKit, and injects text into any focused application.

### Key Decisions
- **Platform**: macOS only (Apple Silicon M1/M2 with 16GB RAM)
- **Framework**: Native Swift/SwiftUI
- **ASR Engine**: WhisperKit (CoreML-based, 9x faster than whisper.cpp on Apple Silicon)
- **VAD**: Silero VAD for speech segmentation
- **LLM Cleanup**: Deferred to v1.1
- **UI Style**: Full app with dock icon + menu bar presence

### Target Metrics
| Metric | Target |
|--------|--------|
| End-to-end latency | <700ms |
| Word error rate | <5% |
| Idle memory | <200MB |
| Active memory | <4GB |

---

## Phase 1: Project Foundation (Tasks 1-6)

### Task 1: Configure Xcode Project Settings
**Description**: Set up the project with proper capabilities, entitlements, and build settings for a sandboxed macOS app.

**Subtasks**:
1. Set deployment target to macOS 13.0 (Ventura) for WhisperKit compatibility
2. Add required capabilities in Signing & Capabilities:
   - App Sandbox (enabled)
   - Hardened Runtime (enabled)
3. Configure entitlements file with:
   - `com.apple.security.device.audio-input` = YES (microphone access)
   - `com.apple.security.accessibility` = YES (for text injection, requires non-sandbox or special handling)
4. Set build settings:
   - Enable Swift 5.9+
   - Set optimization level to `-O` for Release
   - Enable whole module optimization
5. Create Info.plist entries:
   - `NSMicrophoneUsageDescription`: "Used to capture voice for transcription"
   - `NSAppleEventsUsageDescription`: "Used to inject transcribed text"

**Output**: Properly configured Xcode project ready for development.

---

### Task 2: Create App Architecture and Entry Point
**Description**: Establish the core app architecture with proper lifecycle management for a menu bar + dock app hybrid.

**Subtasks**:
1. Create `WhisperOnDeviceApp.swift` main entry point:
   ```swift
   @main
   struct WhisperOnDeviceApp: App {
       @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

       var body: some Scene {
           Settings {
               SettingsView()
           }
       }
   }
   ```
2. Create `AppDelegate.swift` for:
   - Menu bar status item management
   - Global hotkey registration
   - App lifecycle events
3. Create folder structure:
   ```
   WhisperOnDevice/
   ├── App/
   │   ├── WhisperOnDeviceApp.swift
   │   └── AppDelegate.swift
   ├── Views/
   │   ├── MainWindow/
   │   ├── MenuBar/
   │   └── Settings/
   ├── Services/
   │   ├── Audio/
   │   ├── Transcription/
   │   └── TextInjection/
   ├── Models/
   ├── Utilities/
   └── Resources/
   ```
4. Implement basic window management (main window + settings window)

**Output**: Clean app architecture with proper entry point and folder organization.

---

### Task 3: Implement Menu Bar Status Item
**Description**: Create a persistent menu bar icon that shows recording status and provides quick access to app functions.

**Subtasks**:
1. Create `MenuBarManager.swift` as an ObservableObject:
   ```swift
   class MenuBarManager: ObservableObject {
       private var statusItem: NSStatusItem?
       @Published var isRecording = false

       func setupMenuBar() {
           statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
           // Configure icon and menu
       }
   }
   ```
2. Create SF Symbol icons for states:
   - Idle: `waveform` (gray)
   - Recording: `waveform` (red, animated)
   - Processing: `waveform` (orange)
3. Build context menu with items:
   - "Start/Stop Dictation" (with hotkey hint)
   - Separator
   - "Open Main Window"
   - "Settings..."
   - Separator
   - "Quit"
4. Implement click behavior:
   - Left-click: Toggle recording
   - Right-click: Show context menu

**Output**: Functional menu bar presence with visual feedback.

---

### Task 4: Create Main Window UI
**Description**: Design and implement the main application window with dictation controls and transcription display.

**Subtasks**:
1. Create `MainWindowView.swift` with SwiftUI:
   - Recording button (large, prominent)
   - Transcription text display area (scrollable)
   - Status indicator (idle/recording/processing)
   - Waveform visualizer (placeholder for now)
2. Create `RecordingButtonView.swift`:
   - Large circular button with microphone icon
   - Visual states: idle (gray), recording (red pulse), processing (orange)
   - Animation for recording state
3. Create `TranscriptionDisplayView.swift`:
   - Shows recent transcriptions
   - Copy-to-clipboard button per entry
   - Timestamp for each transcription
4. Implement window configuration:
   - Minimum size: 400x300
   - Default size: 500x400
   - Remember window position/size

**Output**: Complete main window UI matching app functionality.

---

### Task 5: Create Settings Window UI
**Description**: Build the settings interface for configuring microphone, hotkey, model, and behavior options.

**Subtasks**:
1. Create `SettingsView.swift` with TabView:
   - General tab
   - Audio tab
   - Model tab
   - Advanced tab
2. Implement General settings tab:
   - Launch at login toggle
   - Show in dock toggle
   - Show in menu bar toggle
   - Default behavior after transcription (copy/paste/both)
3. Implement Audio settings tab:
   - Microphone picker (dropdown of available devices)
   - Input level meter (real-time visualization)
   - Test recording button
4. Implement Model settings tab:
   - Model selection dropdown (tiny, base, small, medium)
   - Model status (downloaded/downloading/not downloaded)
   - Download button per model
   - Disk space usage display
5. Implement Advanced settings tab:
   - Global hotkey configuration
   - VAD sensitivity slider
   - Silence threshold duration
   - Language selection (or auto-detect)

**Output**: Complete settings UI with all configuration options.

---

### Task 6: Implement Settings Persistence
**Description**: Create a settings manager that persists user preferences using UserDefaults with proper observation.

**Subtasks**:
1. Create `SettingsManager.swift`:
   ```swift
   class SettingsManager: ObservableObject {
       static let shared = SettingsManager()

       @AppStorage("selectedMicrophone") var selectedMicrophone: String = ""
       @AppStorage("globalHotkey") var globalHotkey: String = "F5"
       @AppStorage("selectedModel") var selectedModel: String = "base"
       @AppStorage("launchAtLogin") var launchAtLogin: Bool = false
       @AppStorage("showInDock") var showInDock: Bool = true
       @AppStorage("autoInsertText") var autoInsertText: Bool = true
       // ... more settings
   }
   ```
2. Implement launch-at-login functionality using SMAppService (macOS 13+)
3. Implement dock icon visibility toggle using `NSApp.setActivationPolicy()`
4. Create settings migration system for future updates
5. Add reset-to-defaults functionality

**Output**: Persistent settings system with UserDefaults backing.

---

## Phase 2: Audio Capture Pipeline (Tasks 7-11)

### Task 7: Implement Audio Device Manager
**Description**: Create a service to enumerate, monitor, and manage audio input devices.

**Subtasks**:
1. Create `AudioDeviceManager.swift`:
   ```swift
   class AudioDeviceManager: ObservableObject {
       @Published var availableDevices: [AudioDevice] = []
       @Published var selectedDevice: AudioDevice?

       func refreshDevices()
       func selectDevice(_ device: AudioDevice)
   }
   ```
2. Create `AudioDevice` model:
   ```swift
   struct AudioDevice: Identifiable, Hashable {
       let id: AudioDeviceID
       let name: String
       let uid: String
       let isDefault: Bool
   }
   ```
3. Implement device enumeration using CoreAudio:
   - Query `kAudioHardwarePropertyDevices`
   - Filter for input devices only
   - Get device names and UIDs
4. Implement device change monitoring:
   - Listen to `kAudioHardwarePropertyDevices` changes
   - Listen to default device changes
   - Publish updates to SwiftUI views
5. Handle device disconnection gracefully (fall back to default)

**Output**: Robust audio device management with hot-plug support.

---

### Task 8: Implement Audio Capture Engine
**Description**: Create the core audio capture system using AVFoundation for recording microphone input.

**Subtasks**:
1. Create `AudioCaptureEngine.swift`:
   ```swift
   class AudioCaptureEngine: ObservableObject {
       private var audioEngine: AVAudioEngine?
       @Published var isCapturing = false
       @Published var audioLevel: Float = 0

       func startCapture() throws
       func stopCapture() -> AVAudioPCMBuffer?
   }
   ```
2. Configure AVAudioEngine:
   - Input node from selected device
   - Sample rate: 16kHz (WhisperKit requirement)
   - Format: Float32, mono
   - Install tap on input node for level metering
3. Implement circular buffer for audio storage:
   - Store last N seconds of audio
   - Thread-safe access
   - Efficient memory management
4. Implement real-time audio level metering:
   - Calculate RMS from buffer
   - Convert to dB scale
   - Publish for UI visualization
5. Handle audio session interruptions (calls, system sounds)

**Output**: Reliable audio capture with level monitoring.

---

### Task 9: Implement Voice Activity Detection (VAD)
**Description**: Integrate Silero VAD to detect speech segments and determine when the user has finished speaking.

**Subtasks**:
1. Add Silero VAD model to project:
   - Download `silero_vad.onnx` (~1.8MB)
   - Add to app bundle in Resources
2. Create `VADEngine.swift`:
   ```swift
   class VADEngine {
       private var ortSession: ORTSession?

       func loadModel() throws
       func detectSpeech(audioBuffer: [Float]) -> Bool
       func reset()
   }
   ```
3. Integrate ONNX Runtime for Swift:
   - Add onnxruntime-swift package
   - Configure for CPU inference (efficient for VAD)
4. Implement speech detection logic:
   - Process 30ms audio chunks
   - Maintain speech probability state
   - Detect speech start/end with hysteresis
5. Configure VAD parameters:
   - Speech threshold: 0.5 (adjustable)
   - Silence duration for end-of-speech: 500ms
   - Minimum speech duration: 100ms

**Output**: Working VAD that accurately detects speech boundaries.

---

### Task 10: Implement Recording Session Manager
**Description**: Create a coordinator that manages the complete recording session from start to transcription.

**Subtasks**:
1. Create `RecordingSession.swift`:
   ```swift
   class RecordingSession: ObservableObject {
       @Published var state: RecordingState = .idle

       enum RecordingState {
           case idle
           case listening      // Hotkey pressed, waiting for speech
           case recording      // Speech detected, capturing
           case processing     // Transcribing
       }

       func start()
       func stop() -> AudioData?
   }
   ```
2. Implement state machine logic:
   - Idle → Listening (on hotkey press)
   - Listening → Recording (on speech detected)
   - Recording → Processing (on speech end or hotkey release)
   - Processing → Idle (on transcription complete)
3. Coordinate AudioCaptureEngine and VADEngine:
   - Start capture when session starts
   - Feed audio to VAD continuously
   - Collect audio buffer when speech detected
   - Trim silence from start/end of recording
4. Implement timeout handling:
   - Max recording duration: 60 seconds
   - No speech timeout: 5 seconds
5. Add recording visualization data (waveform samples)

**Output**: Complete recording session coordination.

---

### Task 11: Implement Audio Format Conversion
**Description**: Create utilities to convert captured audio to the format required by WhisperKit.

**Subtasks**:
1. Create `AudioConverter.swift`:
   ```swift
   struct AudioConverter {
       static func convertToWhisperFormat(_ buffer: AVAudioPCMBuffer) -> [Float]
       static func resample(_ samples: [Float], from: Double, to: Double) -> [Float]
   }
   ```
2. Implement resampling if needed:
   - WhisperKit expects 16kHz mono Float32
   - Use vDSP for efficient conversion
3. Implement audio normalization:
   - Normalize to [-1, 1] range
   - Apply soft clipping if needed
4. Create audio file export for debugging:
   - Save to WAV format
   - Include metadata (timestamp, duration)
5. Add audio validation:
   - Check for clipping
   - Verify sample rate
   - Ensure sufficient audio length

**Output**: Reliable audio format conversion for WhisperKit.

---

## Phase 3: WhisperKit Integration (Tasks 12-17)

### Task 12: Add WhisperKit Dependency
**Description**: Integrate WhisperKit package and configure for optimal performance on Apple Silicon.

**Subtasks**:
1. Add WhisperKit Swift Package:
   - URL: `https://github.com/argmaxinc/WhisperKit`
   - Version: Latest stable (1.x)
2. Configure build settings for CoreML:
   - Enable Neural Engine support
   - Set compute units to `.cpuAndNeuralEngine`
3. Verify package resolves correctly:
   - Check for any dependency conflicts
   - Ensure minimum macOS version compatibility
4. Create test to verify WhisperKit loads:
   ```swift
   func testWhisperKitLoads() async throws {
       let whisper = try await WhisperKit()
       XCTAssertNotNil(whisper)
   }
   ```
5. Document memory requirements per model size

**Output**: WhisperKit integrated and verified.

---

### Task 13: Implement Model Download Manager
**Description**: Create a system to download, cache, and manage Whisper model files.

**Subtasks**:
1. Create `ModelManager.swift`:
   ```swift
   class ModelManager: ObservableObject {
       @Published var availableModels: [WhisperModel] = []
       @Published var downloadProgress: [String: Double] = [:]

       func downloadModel(_ model: WhisperModel) async throws
       func deleteModel(_ model: WhisperModel) throws
       func getLocalModelPath(_ model: WhisperModel) -> URL?
   }
   ```
2. Define supported models:
   ```swift
   enum WhisperModel: String, CaseIterable {
       case tiny = "openai_whisper-tiny"
       case base = "openai_whisper-base"
       case small = "openai_whisper-small"
       case medium = "openai_whisper-medium"

       var sizeGB: Double { ... }
       var recommendedRAM: Int { ... }
   }
   ```
3. Implement download with progress:
   - Use WhisperKit's built-in download functionality
   - Track and publish download progress
   - Handle download cancellation
4. Store models in Application Support directory:
   - `~/Library/Application Support/WhisperOnDevice/Models/`
   - Verify file integrity after download
5. Implement automatic model cleanup:
   - Remove old/unused models option
   - Show disk space usage

**Output**: Complete model management with download progress.

---

### Task 14: Implement Transcription Engine
**Description**: Create the core transcription service that interfaces with WhisperKit.

**Subtasks**:
1. Create `TranscriptionEngine.swift`:
   ```swift
   actor TranscriptionEngine {
       private var whisperKit: WhisperKit?

       func loadModel(_ model: WhisperModel) async throws
       func transcribe(_ audio: [Float]) async throws -> TranscriptionResult
       func unloadModel()
   }
   ```
2. Implement model loading with memory management:
   - Load model asynchronously
   - Track memory usage
   - Support model hot-swapping
3. Configure transcription options:
   ```swift
   let options = DecodingOptions(
       language: "en",          // or nil for auto-detect
       task: .transcribe,
       temperatureFallbackCount: 3,
       sampleLength: 224,
       usePrefillPrompt: false
   )
   ```
4. Implement transcription result handling:
   ```swift
   struct TranscriptionResult {
       let text: String
       let segments: [TranscriptionSegment]
       let language: String
       let processingTime: TimeInterval
   }
   ```
5. Add error handling:
   - Model not loaded
   - Audio too short
   - Transcription timeout
   - Memory pressure

**Output**: Working transcription engine with WhisperKit.

---

### Task 15: Implement Transcription Post-Processing
**Description**: Clean up raw transcription output with basic formatting (punctuation, capitalization).

**Subtasks**:
1. Create `TranscriptionPostProcessor.swift`:
   ```swift
   struct TranscriptionPostProcessor {
       func process(_ text: String) -> String
   }
   ```
2. Implement basic cleanup:
   - Trim whitespace
   - Remove duplicate spaces
   - Fix common Whisper artifacts (repeated words, hallucinations)
3. Implement sentence capitalization:
   - Capitalize after periods, question marks, exclamation points
   - Capitalize "I" when standalone
4. Implement basic punctuation normalization:
   - Ensure space after punctuation
   - Remove space before punctuation
5. Handle edge cases:
   - Empty transcriptions
   - Single-word transcriptions
   - Non-English text (pass through unchanged)

**Note**: Advanced LLM-based cleanup deferred to v1.1.

**Output**: Basic transcription cleanup without LLM.

---

### Task 16: Implement Transcription History
**Description**: Store and manage transcription history for review and re-use.

**Subtasks**:
1. Create `TranscriptionHistory.swift`:
   ```swift
   class TranscriptionHistory: ObservableObject {
       @Published var entries: [TranscriptionEntry] = []

       func add(_ entry: TranscriptionEntry)
       func delete(_ entry: TranscriptionEntry)
       func clear()
   }
   ```
2. Create `TranscriptionEntry` model:
   ```swift
   struct TranscriptionEntry: Identifiable, Codable {
       let id: UUID
       let text: String
       let timestamp: Date
       let duration: TimeInterval
       let model: String
   }
   ```
3. Implement persistence:
   - Store in JSON file in Application Support
   - Load on app launch
   - Save on each new entry
4. Implement entry limits:
   - Keep last 100 entries (configurable)
   - Auto-prune old entries
5. Add search functionality:
   - Full-text search in entries
   - Filter by date range

**Output**: Persistent transcription history system.

---

### Task 17: Performance Optimization for Transcription
**Description**: Optimize the transcription pipeline to achieve <700ms latency target.

**Subtasks**:
1. Implement model preloading:
   - Load model at app startup
   - Keep model in memory while app is active
2. Profile transcription latency:
   - Measure audio capture → transcription start
   - Measure transcription duration
   - Measure post-processing duration
3. Optimize audio pipeline:
   - Minimize buffer copies
   - Use contiguous memory
   - Pre-allocate buffers
4. Configure WhisperKit for speed:
   - Use appropriate compute units
   - Tune decoding parameters
   - Consider using smaller models for latency
5. Add latency reporting:
   - Log timing breakdown
   - Surface in advanced settings for debugging

**Output**: Optimized pipeline meeting latency targets.

---

## Phase 4: Text Injection System (Tasks 18-22)

### Task 18: Implement Accessibility Permission Handler
**Description**: Request and verify accessibility permissions required for text injection.

**Subtasks**:
1. Create `AccessibilityManager.swift`:
   ```swift
   class AccessibilityManager: ObservableObject {
       @Published var hasPermission = false

       func checkPermission() -> Bool
       func requestPermission()
   }
   ```
2. Implement permission checking:
   ```swift
   func checkPermission() -> Bool {
       return AXIsProcessTrusted()
   }
   ```
3. Implement permission prompting:
   ```swift
   func requestPermission() {
       let options = [kAXTrustedCheckOptionPrompt.takeRetainedValue(): true]
       AXIsProcessTrustedWithOptions(options as CFDictionary)
   }
   ```
4. Create permission onboarding UI:
   - Explain why permission is needed
   - Button to open System Preferences
   - Visual confirmation when granted
5. Monitor permission changes:
   - Poll for permission status
   - Update UI when granted/revoked

**Output**: Proper accessibility permission handling.

---

### Task 19: Implement Clipboard-Based Text Injection
**Description**: Create the primary text injection method using clipboard paste simulation.

**Subtasks**:
1. Create `TextInjector.swift`:
   ```swift
   class TextInjector {
       func inject(_ text: String) throws
       private func copyToClipboard(_ text: String)
       private func simulatePaste()
   }
   ```
2. Implement clipboard operations:
   ```swift
   func copyToClipboard(_ text: String) {
       let pasteboard = NSPasteboard.general
       pasteboard.clearContents()
       pasteboard.setString(text, forType: .string)
   }
   ```
3. Implement paste simulation using CGEvent:
   ```swift
   func simulatePaste() {
       let source = CGEventSource(stateID: .hidSystemState)

       // Key down: Cmd+V
       let keyDown = CGEvent(keyboardEventSource: source, virtualKey: 0x09, keyDown: true)
       keyDown?.flags = .maskCommand
       keyDown?.post(tap: .cghidEventTap)

       // Key up: Cmd+V
       let keyUp = CGEvent(keyboardEventSource: source, virtualKey: 0x09, keyDown: false)
       keyUp?.flags = .maskCommand
       keyUp?.post(tap: .cghidEventTap)
   }
   ```
4. Implement clipboard restoration:
   - Save original clipboard contents
   - Restore after paste (optional, based on settings)
5. Add timing delays if needed:
   - Small delay between copy and paste
   - Handle slow applications

**Output**: Reliable clipboard-based text injection.

---

### Task 20: Implement Global Hotkey System
**Description**: Create a system for capturing global keyboard shortcuts even when app is not focused.

**Subtasks**:
1. Create `HotkeyManager.swift`:
   ```swift
   class HotkeyManager: ObservableObject {
       private var eventTap: CFMachPort?
       @Published var currentHotkey: KeyCombo = .f5

       var onHotkeyDown: (() -> Void)?
       var onHotkeyUp: (() -> Void)?

       func startListening()
       func stopListening()
   }
   ```
2. Implement CGEventTap for global key monitoring:
   ```swift
   let eventMask = (1 << CGEventType.keyDown.rawValue) | (1 << CGEventType.keyUp.rawValue)
   eventTap = CGEvent.tapCreate(
       tap: .cgSessionEventTap,
       place: .headInsertEventTap,
       options: .defaultTap,
       eventsOfInterest: CGEventMask(eventMask),
       callback: hotkeyCallback,
       userInfo: pointer
   )
   ```
3. Create `KeyCombo` model:
   ```swift
   struct KeyCombo: Codable, Equatable {
       let keyCode: UInt16
       let modifiers: NSEvent.ModifierFlags

       static let f5 = KeyCombo(keyCode: 0x60, modifiers: [])
       static let optionSpace = KeyCombo(keyCode: 0x31, modifiers: .option)
   }
   ```
4. Implement hotkey recording for settings:
   - UI to capture new hotkey
   - Validate hotkey isn't system-reserved
   - Save to settings
5. Handle conflicts:
   - Detect if hotkey is used by other apps
   - Warn user of potential conflicts

**Output**: Working global hotkey system.

---

### Task 21: Implement Push-to-Talk Flow
**Description**: Create the complete push-to-talk user experience from hotkey press to text injection.

**Subtasks**:
1. Create `DictationCoordinator.swift`:
   ```swift
   class DictationCoordinator: ObservableObject {
       @Published var state: DictationState = .idle

       func onHotkeyDown()
       func onHotkeyUp()
   }
   ```
2. Implement push-to-talk mode:
   - Hotkey down: Start recording immediately
   - Hotkey up: Stop recording, process, inject
3. Implement push-to-toggle mode (alternative):
   - First press: Start recording
   - Second press: Stop recording
4. Coordinate all subsystems:
   - Audio capture
   - VAD processing
   - Transcription
   - Post-processing
   - Text injection
5. Implement audio feedback:
   - Optional sound on recording start
   - Optional sound on transcription complete
   - Optional sound on error

**Output**: Complete push-to-talk dictation flow.

---

### Task 22: Implement Error Handling and User Feedback
**Description**: Create comprehensive error handling with clear user feedback for all failure modes.

**Subtasks**:
1. Create `DictationError` enum:
   ```swift
   enum DictationError: LocalizedError {
       case microphonePermissionDenied
       case accessibilityPermissionDenied
       case modelNotLoaded
       case audioCaptureFailed
       case transcriptionFailed
       case textInjectionFailed
       case noSpeechDetected
   }
   ```
2. Implement error display:
   - Show error in menu bar tooltip
   - Show notification for critical errors
   - Log errors for debugging
3. Implement recovery suggestions:
   - Guide user to grant permissions
   - Suggest downloading model
   - Offer retry option
4. Add status feedback:
   - Visual indicator during recording
   - Visual indicator during processing
   - Success/failure indication
5. Implement notification system:
   - Use UserNotifications framework
   - Request notification permission
   - Show transcription success (optional)

**Output**: Robust error handling with clear user feedback.

---

## Phase 5: Polish and Testing (Tasks 23-27)

### Task 23: Implement Memory Management
**Description**: Optimize memory usage and handle memory pressure gracefully.

**Subtasks**:
1. Monitor memory usage:
   ```swift
   func getMemoryUsage() -> UInt64 {
       var info = mach_task_basic_info()
       var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
       let result = withUnsafeMutablePointer(to: &info) {
           $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
               task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
           }
       }
       return info.resident_size
   }
   ```
2. Implement memory pressure handling:
   - Subscribe to `NSProcessInfo.memoryPressureNotification`
   - Unload model on critical pressure
   - Clear caches on warning
3. Implement lazy model loading:
   - Load model on first dictation
   - Unload after configurable idle time
4. Profile and optimize:
   - Use Instruments to find leaks
   - Optimize buffer allocations
   - Reduce peak memory usage
5. Add memory usage display in settings (advanced)

**Output**: Memory-efficient app that handles pressure gracefully.

---

### Task 24: Implement First-Run Onboarding
**Description**: Create a smooth onboarding experience for new users.

**Subtasks**:
1. Create `OnboardingView.swift`:
   - Welcome screen with app overview
   - Permission request screens
   - Model download screen
   - Hotkey configuration screen
   - Ready-to-use confirmation
2. Implement permission request flow:
   - Microphone permission (step 1)
   - Accessibility permission (step 2)
   - Explain each permission clearly
3. Implement model selection/download:
   - Recommend model based on RAM
   - Show download progress
   - Allow skip (will prompt later)
4. Implement hotkey setup:
   - Show default hotkey
   - Allow customization
   - Test hotkey works
5. Track onboarding completion:
   - Store completion state
   - Allow re-running from settings

**Output**: Polished onboarding experience.

---

### Task 25: Implement App Icon and Visual Polish
**Description**: Create app icon and finalize visual design.

**Subtasks**:
1. Create app icon:
   - Design 1024x1024 base icon
   - Generate all required sizes (16, 32, 64, 128, 256, 512)
   - Create iconset for Xcode
2. Create menu bar icons:
   - Template images for light/dark mode
   - Recording state variations
   - Proper sizing (18x18 @1x, 36x36 @2x)
3. Finalize UI theme:
   - Ensure proper dark mode support
   - Consistent spacing and typography
   - Accessibility compliance (VoiceOver)
4. Add visual animations:
   - Recording pulse animation
   - Processing spinner
   - Success checkmark
5. Review on multiple displays:
   - Retina displays
   - External monitors
   - Different resolutions

**Output**: Polished visual design and app icon.

---

### Task 26: Implement Comprehensive Testing
**Description**: Create tests for critical functionality.

**Subtasks**:
1. Unit tests for audio processing:
   - Audio format conversion
   - Resampling accuracy
   - Buffer management
2. Unit tests for transcription post-processing:
   - Capitalization
   - Punctuation normalization
   - Edge cases
3. Integration tests:
   - Recording session flow
   - Model loading/unloading
   - Settings persistence
4. UI tests:
   - Main window navigation
   - Settings changes
   - Onboarding flow
5. Performance tests:
   - Transcription latency benchmarks
   - Memory usage benchmarks
   - Startup time benchmarks

**Output**: Test suite covering critical paths.

---

### Task 27: Prepare for Distribution
**Description**: Prepare the app for distribution outside the App Store (notarization).

**Subtasks**:
1. Configure code signing:
   - Developer ID certificate
   - Proper entitlements
   - Hardened runtime
2. Implement notarization workflow:
   - Archive build
   - Submit to Apple notarization service
   - Staple ticket to app
3. Create DMG installer:
   - App icon in DMG
   - Applications folder shortcut
   - Background image
4. Create update mechanism:
   - Integrate Sparkle framework
   - Configure appcast URL
   - Sign updates
5. Test distribution:
   - Fresh install on clean system
   - Gatekeeper acceptance
   - Permissions flow

**Output**: Distributable, notarized application.

---

## Appendix A: File Structure

```
WhisperOnDevice/
├── WhisperOnDevice.xcodeproj/
├── WhisperOnDevice/
│   ├── App/
│   │   ├── WhisperOnDeviceApp.swift
│   │   └── AppDelegate.swift
│   ├── Views/
│   │   ├── MainWindow/
│   │   │   ├── MainWindowView.swift
│   │   │   ├── RecordingButtonView.swift
│   │   │   └── TranscriptionDisplayView.swift
│   │   ├── MenuBar/
│   │   │   └── MenuBarManager.swift
│   │   ├── Settings/
│   │   │   ├── SettingsView.swift
│   │   │   ├── GeneralSettingsTab.swift
│   │   │   ├── AudioSettingsTab.swift
│   │   │   ├── ModelSettingsTab.swift
│   │   │   └── AdvancedSettingsTab.swift
│   │   └── Onboarding/
│   │       └── OnboardingView.swift
│   ├── Services/
│   │   ├── Audio/
│   │   │   ├── AudioDeviceManager.swift
│   │   │   ├── AudioCaptureEngine.swift
│   │   │   ├── AudioConverter.swift
│   │   │   └── VADEngine.swift
│   │   ├── Transcription/
│   │   │   ├── TranscriptionEngine.swift
│   │   │   ├── TranscriptionPostProcessor.swift
│   │   │   ├── TranscriptionHistory.swift
│   │   │   └── ModelManager.swift
│   │   └── TextInjection/
│   │       ├── TextInjector.swift
│   │       ├── AccessibilityManager.swift
│   │       └── HotkeyManager.swift
│   ├── Coordination/
│   │   ├── DictationCoordinator.swift
│   │   └── RecordingSession.swift
│   ├── Models/
│   │   ├── AudioDevice.swift
│   │   ├── WhisperModel.swift
│   │   ├── TranscriptionEntry.swift
│   │   ├── TranscriptionResult.swift
│   │   ├── KeyCombo.swift
│   │   └── DictationError.swift
│   ├── Utilities/
│   │   └── SettingsManager.swift
│   ├── Resources/
│   │   ├── Assets.xcassets/
│   │   ├── silero_vad.onnx
│   │   └── Sounds/
│   └── Info.plist
├── WhisperOnDeviceTests/
└── WhisperOnDeviceUITests/
```

---

## Appendix B: Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| WhisperKit | 1.x | Speech-to-text transcription |
| onnxruntime-swift | Latest | Silero VAD inference |
| Sparkle | 2.x | Auto-updates (optional) |

---

## Appendix C: Required Permissions

| Permission | Reason | When Requested |
|------------|--------|----------------|
| Microphone | Audio capture for transcription | Onboarding |
| Accessibility | Global hotkey and text injection | Onboarding |
| Notifications | Success/error feedback | Onboarding (optional) |

---

## Summary

This plan consists of **27 tasks** organized into **5 phases**:

1. **Phase 1: Project Foundation** (Tasks 1-6) - Basic app structure, UI, settings
2. **Phase 2: Audio Capture Pipeline** (Tasks 7-11) - Microphone, VAD, recording
3. **Phase 3: WhisperKit Integration** (Tasks 12-17) - Model management, transcription
4. **Phase 4: Text Injection System** (Tasks 18-22) - Permissions, hotkeys, injection
5. **Phase 5: Polish and Testing** (Tasks 23-27) - Memory, onboarding, distribution

Each task is designed to be completable independently while building toward the full MVP. The estimated scope is appropriate for a single developer over 8-12 weeks, with a functional dictation app achievable by the end of Phase 4.
