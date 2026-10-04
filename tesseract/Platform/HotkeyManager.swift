//
//  HotkeyManager.swift
//  tesseract
//

import Foundation
import AppKit
import Combine
import Carbon.HIToolbox

struct HotkeyRegistration {
    let id: String
    let combo: KeyCombo
    let onDown: () -> Void
    let onUp: (() -> Void)?
    /// A one-key hotkey was spoiled: another key joined while it was held.
    var onCancel: (() -> Void)?
    /// Shift was tapped while this hotkey was held (`ModifierTapDetector`):
    /// the dictation hotkey's "keep this take in the Lens".
    var onShiftTap: (() -> Void)?
}

@MainActor
final class HotkeyManager: ObservableObject {
    /// The dictation registration's id — dictation registers through the one
    /// `registerHotkey` API like every other hotkey (audit #285 item 7; the
    /// former `currentHotkey`/`onHotkeyDown` mirror was a second source of
    /// truth for the press that starts everything).
    static let dictationHotkeyID = "dictation"
    /// The fix registration's id (⌃⌥Space by default): reopens the last take
    /// in the **Lens** to fix a word.
    static let fixLastTakeHotkeyID = "fixLastTake"

    @Published private(set) var isListening = false
    @Published private(set) var isUsingEventTap = false

    /// The owner's key presses and clicks, counted from both delivery
    /// paths: every key-down that is not Tesseract's own
    /// (`SyntheticKeyEvents`), not an auto-repeat and not a hotkey's own key,
    /// and every mouse-button press (a click can move the caret or select
    /// text). A fix put back in an app (`InAppReplacer`) compares it across
    /// the fix, so it never selects or backspaces over text the owner typed,
    /// or moved away from, since the paste. Input into Tesseract's own
    /// windows counts too; the caller allows for it. Deliberately not
    /// `@Published`: it changes on every keystroke system-wide.
    private(set) var inputCount = 0

    /// The dictation registration's current combo — the gate read App
    /// Bindings uses to skip no-op re-binds. Falls back to the default combo
    /// until the registration lands at setup.
    var currentDictationHotkey: KeyCombo {
        registrations[Self.dictationHotkeyID]?.combo ?? .optionSpace
    }

    private var registrations: [String: HotkeyRegistration] = [:] {
        didSet {
            bindingsSnapshot = registrations.filter { !$0.value.combo.isSingleModifier }
                .mapValues(\.combo)
            singleModifierDetectors = registrations.compactMapValues {
                ModifierKeyDetector(combo: $0.combo)
            }
            shiftTapIDs = registrations.values.filter { $0.onShiftTap != nil }.map(\.id).sorted()
        }
    }

    /// The registrations that want a Shift tap while held, prebuilt like
    /// `bindingsSnapshot` so the hot path never filters `registrations`.
    private var shiftTapIDs: [String] = []

    /// The one Shift-tap detector, fed from both delivery paths while any
    /// registration wants taps.
    private var shiftTapDetector = ModifierTapDetector()

    /// One detector per one-key registration (`combo.isSingleModifier`), fed
    /// from both delivery paths.
    private var singleModifierDetectors: [String: ModifierKeyDetector] = [:]

    /// The pending recording's way out, for the recorder's Cancel button.
    private var cancelPendingRecording: (() -> Void)?

    /// Prebuilt `id → combo` view of `registrations`, rebuilt on every
    /// (un)register/update so the per-keystroke hot path hands the matcher an
    /// existing dictionary instead of allocating one per key event
    /// system-wide (audit #285 item 7).
    private var bindingsSnapshot: [String: KeyCombo] = [:]

    /// The one fire-or-not decision, shared by both delivery paths. Both the
    /// tap callback and the NSEvent fallback normalize their event and fold
    /// it through this matcher; only delivery timing differs per path.
    private var matcher = HotkeyMatcher()

    /// Chord state for double-Command registrations (`combo.isDoubleCommand`).
    /// Fed raw flag words from both delivery paths; fires `onDown` once per
    /// chord (one-shot — these registrations have no held state, no `onUp`).
    private var doubleCommandDetector = DoubleCommandDetector()

    private var eventTap: CFMachPort?
    private var runLoopSource: CFRunLoopSource?
    /// The listen-only tap that counts clicks (`inputCount`).
    private var clickTap: CFMachPort?
    private var clickRunLoopSource: CFRunLoopSource?

    // Fallback monitors for when Accessibility permission is denied
    private var globalMonitor: Any?
    private var localMonitor: Any?

    init() {}

    deinit {
        MainActor.assumeIsolated {
            stopListening()
        }
    }

    // MARK: - Multi-Hotkey Registration

    func registerHotkey(
        id: String, combo: KeyCombo, onDown: @escaping () -> Void, onUp: (() -> Void)? = nil,
        onCancel: (() -> Void)? = nil, onShiftTap: (() -> Void)? = nil
    ) {
        registrations[id] = HotkeyRegistration(
            id: id, combo: combo, onDown: onDown, onUp: onUp, onCancel: onCancel,
            onShiftTap: onShiftTap)
    }

    func unregisterHotkey(id: String) {
        registrations.removeValue(forKey: id)
        matcher.forget(id: id)
    }

    func updateRegisteredHotkey(id: String, combo: KeyCombo) {
        guard var reg = registrations[id] else { return }
        reg = HotkeyRegistration(
            id: id, combo: combo, onDown: reg.onDown, onUp: reg.onUp, onCancel: reg.onCancel,
            onShiftTap: reg.onShiftTap)
        registrations[id] = reg
        matcher.forget(id: id)
    }

    // MARK: - Listening

    func startListening() {
        guard !isListening else { return }

        // Try CGEventTap first (requires Accessibility permission)
        if AXIsProcessTrusted() {
            startEventTap()
        } else {
            // Fall back to NSEvent monitors (cannot suppress events)
            startNSEventMonitors()
        }

        isListening = true
    }

    func stopListening() {
        stopEventTap()
        stopNSEventMonitors()

        isListening = false
        matcher.reset()
        for id in singleModifierDetectors.keys { singleModifierDetectors[id]?.reset() }
        shiftTapDetector.reset()
        isUsingEventTap = false
    }

    // MARK: - CGEventTap Implementation

    private func startEventTap() {
        let eventMask =
            (1 << CGEventType.keyDown.rawValue) | (1 << CGEventType.keyUp.rawValue)
            | (1 << CGEventType.flagsChanged.rawValue)

        // Store self pointer for callback
        let refcon = Unmanaged.passUnretained(self).toOpaque()

        eventTap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .defaultTap,  // Enables suppression (return nil to suppress)
            eventsOfInterest: CGEventMask(eventMask),
            callback: { _, type, event, refcon -> Unmanaged<CGEvent>? in
                guard let refcon = refcon else { return Unmanaged.passUnretained(event) }

                let manager = Unmanaged<HotkeyManager>.fromOpaque(refcon).takeUnretainedValue()

                // Handle tap being disabled by the system
                if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
                    if let tap = manager.eventTap {
                        CGEvent.tapEnable(tap: tap, enable: true)
                    }
                    return Unmanaged.passUnretained(event)
                }

                // Tesseract's own keys (a paste, a copy, a fix's backspaces)
                // are not the owner typing: never a hotkey, never counted.
                if SyntheticKeyEvents.isOurs(event) {
                    return Unmanaged.passUnretained(event)
                }

                let keyCode = UInt16(event.getIntegerValueField(.keyboardEventKeycode))
                let flags = event.flags
                let isRepeat =
                    type == .keyDown
                    && event.getIntegerValueField(.keyboardEventAutorepeat) != 0

                // Convert CGEventFlags to NSEvent.ModifierFlags
                var modifiers: NSEvent.ModifierFlags = []
                if flags.contains(.maskCommand) { modifiers.insert(.command) }
                if flags.contains(.maskAlternate) { modifiers.insert(.option) }
                if flags.contains(.maskControl) { modifiers.insert(.control) }
                if flags.contains(.maskShift) { modifiers.insert(.shift) }
                if flags.contains(.maskSecondaryFn) { modifiers.insert(.function) }

                let kind: HotkeyMatcher.EventKind
                switch type {
                case .flagsChanged:
                    manager.handleDoubleCommandFlags(rawFlags: flags.rawValue)
                    manager.handleSingleModifierFlags(keyCode: keyCode, rawFlags: flags.rawValue)
                    manager.handleShiftTapFlags(keyCode: keyCode, rawFlags: flags.rawValue)
                    kind = .flagsChanged
                case .keyUp:
                    kind = .keyUp
                default:
                    manager.handleSingleModifierKeyDown()
                    manager.handleShiftTapKeyDown(isRepeat: isRepeat)
                    kind = .keyDown
                }

                let verdict = manager.matcher.handle(
                    kind, keyCode: keyCode, modifiers: modifiers, isRepeat: isRepeat,
                    bindings: manager.bindingsSnapshot)

                // Deliver on the next main-queue turn so the tap callback
                // stays fast; the matcher state is already settled.
                manager.deliver(verdict.fires, deferred: true)
                manager.countKeyDown(kind, isRepeat: isRepeat, verdict: verdict)

                // Suppress matched key events (flagsChanged always passes).
                if verdict.suppressKeyEvent {
                    return nil
                }

                return Unmanaged.passUnretained(event)
            },
            userInfo: refcon
        )

        guard let eventTap = eventTap else {
            // Failed to create event tap, fall back to NSEvent monitors
            startNSEventMonitors()
            return
        }

        // The tap deliberately runs on the MAIN run loop (audit #285 item 7,
        // decided): every fire targets a @MainActor consumer anyway, so a
        // dedicated tap thread would only move the queuing point without
        // shortening felt latency — while adding a thread-confinement story
        // for the matcher. The callback itself stays O(bindings) with zero
        // allocation (prebuilt `bindingsSnapshot`, deferred delivery), and
        // the `tapDisabledByTimeout` re-enable above covers a stalled turn.
        runLoopSource = CFMachPortCreateRunLoopSource(nil, eventTap, 0)
        CFRunLoopAddSource(CFRunLoopGetMain(), runLoopSource, .commonModes)
        CGEvent.tapEnable(tap: eventTap, enable: true)

        startClickTap()
        isUsingEventTap = true
    }

    /// Clicks count as the owner's input (they can move the caret), seen by
    /// a second, listen-only tap: unlike the key tap it never holds an
    /// event, so a busy main thread can never delay a click.
    private func startClickTap() {
        let mask =
            (1 << CGEventType.leftMouseDown.rawValue) | (1 << CGEventType.rightMouseDown.rawValue)
            | (1 << CGEventType.otherMouseDown.rawValue)
        clickTap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .tailAppendEventTap,
            options: .listenOnly,
            eventsOfInterest: CGEventMask(mask),
            callback: { _, type, event, refcon -> Unmanaged<CGEvent>? in
                guard let refcon else { return Unmanaged.passUnretained(event) }
                let manager = Unmanaged<HotkeyManager>.fromOpaque(refcon).takeUnretainedValue()
                if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
                    if let tap = manager.clickTap { CGEvent.tapEnable(tap: tap, enable: true) }
                } else {
                    manager.inputCount &+= 1
                }
                return Unmanaged.passUnretained(event)
            },
            userInfo: Unmanaged.passUnretained(self).toOpaque()
        )
        guard let clickTap else { return }
        clickRunLoopSource = CFMachPortCreateRunLoopSource(nil, clickTap, 0)
        CFRunLoopAddSource(CFRunLoopGetMain(), clickRunLoopSource, .commonModes)
        CGEvent.tapEnable(tap: clickTap, enable: true)
    }

    private func stopEventTap() {
        if let eventTap = eventTap {
            CGEvent.tapEnable(tap: eventTap, enable: false)
            CFMachPortInvalidate(eventTap)
            self.eventTap = nil
        }
        if let clickTap {
            CGEvent.tapEnable(tap: clickTap, enable: false)
            CFMachPortInvalidate(clickTap)
            self.clickTap = nil
        }
        if let clickRunLoopSource {
            CFRunLoopRemoveSource(CFRunLoopGetMain(), clickRunLoopSource, .commonModes)
            self.clickRunLoopSource = nil
        }

        if let runLoopSource = runLoopSource {
            CFRunLoopRemoveSource(CFRunLoopGetMain(), runLoopSource, .commonModes)
            self.runLoopSource = nil
        }
    }

    // MARK: - NSEvent Fallback (No Suppression)

    private func startNSEventMonitors() {
        // Global monitor for when app is not focused
        globalMonitor = NSEvent.addGlobalMonitorForEvents(
            matching: [
                .keyDown, .keyUp, .flagsChanged, .leftMouseDown, .rightMouseDown, .otherMouseDown,
            ]
        ) { [weak self] event in
            Task { @MainActor in
                self?.handleKeyEvent(event)
            }
        }

        // Local monitor for when app is focused
        localMonitor = NSEvent.addLocalMonitorForEvents(
            matching: [
                .keyDown, .keyUp, .flagsChanged, .leftMouseDown, .rightMouseDown, .otherMouseDown,
            ]
        ) { [weak self] event in
            Task { @MainActor in
                self?.handleKeyEvent(event)
            }
            return event
        }

        isUsingEventTap = false
    }

    private func stopNSEventMonitors() {
        if let monitor = globalMonitor {
            NSEvent.removeMonitor(monitor)
            globalMonitor = nil
        }

        if let monitor = localMonitor {
            NSEvent.removeMonitor(monitor)
            localMonitor = nil
        }
    }

    private func handleKeyEvent(_ event: NSEvent) {
        // Tesseract's own keys are not the owner typing (see the tap).
        if let cgEvent = event.cgEvent, SyntheticKeyEvents.isOurs(cgEvent) { return }
        // A click is input, and nothing else (see the tap).
        if [.leftMouseDown, .rightMouseDown, .otherMouseDown].contains(event.type) {
            inputCount &+= 1
            return
        }

        let kind: HotkeyMatcher.EventKind
        // `isARepeat` raises for anything but a key event; read it only there.
        var isRepeat = false
        switch event.type {
        case .flagsChanged:
            let rawFlags = UInt64(event.modifierFlags.rawValue)
            handleDoubleCommandFlags(rawFlags: rawFlags)
            handleSingleModifierFlags(keyCode: event.keyCode, rawFlags: rawFlags)
            handleShiftTapFlags(keyCode: event.keyCode, rawFlags: rawFlags)
            kind = .flagsChanged
        case .keyUp:
            kind = .keyUp
        default:
            isRepeat = event.isARepeat
            handleSingleModifierKeyDown()
            handleShiftTapKeyDown(isRepeat: isRepeat)
            kind = .keyDown
        }

        let verdict = matcher.handle(
            kind, keyCode: event.keyCode, modifiers: event.modifierFlags, isRepeat: isRepeat,
            bindings: bindingsSnapshot)

        // Monitors cannot suppress events; deliver synchronously.
        deliver(verdict.fires, deferred: false)
        countKeyDown(kind, isRepeat: isRepeat, verdict: verdict)
    }

    /// Shared by both delivery paths: one more key the owner pressed, unless
    /// it repeated or was a hotkey's own key. The fallback cannot suppress a
    /// hotkey's key, but it is not counted there either, so both paths agree.
    private func countKeyDown(
        _ kind: HotkeyMatcher.EventKind, isRepeat: Bool, verdict: HotkeyMatcher.Verdict
    ) {
        guard kind == .keyDown, !isRepeat, !verdict.suppressKeyEvent else { return }
        inputCount &+= 1
    }

    /// Deliver matcher fires to their registrations, looking each one up at
    /// delivery time so an unregister that lands before a deferred delivery
    /// quietly drops the fire.
    private func deliver(_ fires: [HotkeyMatcher.Fire], deferred: Bool) {
        for fire in fires {
            if deferred {
                DispatchQueue.main.async { [weak self] in
                    self?.deliverOne(fire)
                }
            } else {
                deliverOne(fire)
            }
        }
    }

    private func deliverOne(_ fire: HotkeyMatcher.Fire) {
        guard let reg = registrations[fire.id] else { return }
        switch fire.direction {
        case .down: reg.onDown()
        case .up: reg.onUp?()
        }
    }

    // MARK: - Double-Command Chord

    /// Shared by both event paths (tap and NSEvent fallback): feed one
    /// `flagsChanged` word to the chord detector and, on fire, invoke every
    /// double-Command registration once. Always hops to the next main-queue
    /// turn so both paths deliver identically.
    private func handleDoubleCommandFlags(rawFlags: UInt64) {
        guard doubleCommandDetector.handleFlagsChanged(rawFlags: rawFlags) else { return }
        for (_, reg) in registrations where reg.combo.isDoubleCommand {
            DispatchQueue.main.async {
                reg.onDown()
            }
        }
    }

    // MARK: - One-Key Hotkeys

    /// Shared by both event paths: feed one `flagsChanged` event to every
    /// one-key detector and deliver what they report on the next main-queue
    /// turn, like every other fire.
    private func handleSingleModifierFlags(keyCode: UInt16, rawFlags: UInt64) {
        for id in singleModifierDetectors.keys.sorted() {
            guard
                let event = singleModifierDetectors[id]?.flagsChanged(
                    keyCode: keyCode, rawFlags: rawFlags)
            else { continue }
            deliverSingleModifier(event, id: id)
        }
    }

    /// Any key typed while a one-key hotkey is held spoils it.
    private func handleSingleModifierKeyDown() {
        for id in singleModifierDetectors.keys.sorted() {
            guard let event = singleModifierDetectors[id]?.keyDown() else { continue }
            deliverSingleModifier(event, id: id)
        }
    }

    private func deliverSingleModifier(_ event: ModifierKeyDetector.Event, id: String) {
        DispatchQueue.main.async { [weak self] in
            guard let reg = self?.registrations[id] else { return }
            switch event {
            case .down: reg.onDown()
            case .up: reg.onUp?()
            case .cancel: reg.onCancel?()
            }
        }
    }

    // MARK: - Shift Tap

    /// Shared by both delivery paths: feed one `flagsChanged` event to the
    /// Shift-tap detector and, on a tap, deliver it on the next main-queue
    /// turn to every registration that wants taps and is held right now
    /// (before this event reaches the matcher, so a release in the same
    /// event still counts the tap; its `onUp` is queued after the tap).
    private func handleShiftTapFlags(keyCode: UInt16, rawFlags: UInt64) {
        guard !shiftTapIDs.isEmpty,
            shiftTapDetector.flagsChanged(keyCode: keyCode, rawFlags: rawFlags)
        else { return }
        for id in shiftTapIDs where matcher.pressed.contains(id) {
            DispatchQueue.main.async { [weak self] in
                self?.registrations[id]?.onShiftTap?()
            }
        }
    }

    /// A key pressed while Shift is down makes it a shifted key, not a tap.
    private func handleShiftTapKeyDown(isRepeat: Bool) {
        guard !shiftTapIDs.isEmpty else { return }
        shiftTapDetector.keyDown(isRepeat: isRepeat)
    }

    // MARK: - Permission Handling

    func refreshForAccessibilityPermission() {
        guard isListening else { return }

        if !isUsingEventTap && AXIsProcessTrusted() {
            stopNSEventMonitors()
            startEventTap()
        }
    }

    // MARK: - Hotkey Recording

    /// Record the next hotkey: a key with its modifiers, or a modifier
    /// pressed and released on its own (a one-key hotkey such as Right ⌥).
    /// Escape, the recorder's Cancel (`cancelRecording()`) and a 10-second
    /// timeout end it with nil.
    func recordHotkey() async -> KeyCombo? {
        return await withCheckedContinuation { continuation in
            var hasResumed = false
            var monitor: Any?
            let shouldResumeListening = isListening
            // A one-key candidate: a lone modifier went down and nothing
            // else has been pressed since.
            var loneModifier: ModifierKeyDetector?

            if shouldResumeListening {
                stopListening()
            }

            // Every path here runs on the main thread: the local monitor, the
            // timeout task, and the recorder's Cancel.
            func finish(_ result: KeyCombo?) {
                guard !hasResumed else { return }
                hasResumed = true
                MainActor.assumeIsolated { self.cancelPendingRecording = nil }
                if let monitor {
                    NSEvent.removeMonitor(monitor)
                }
                if shouldResumeListening {
                    Task { @MainActor in
                        self.startListening()
                    }
                }
                continuation.resume(returning: result)
            }
            MainActor.assumeIsolated { self.cancelPendingRecording = { finish(nil) } }

            monitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .flagsChanged]) {
                event in
                let keyCode = event.keyCode
                if event.type == .flagsChanged {
                    let rawFlags = UInt64(event.modifierFlags.rawValue)
                    if loneModifier == nil,
                        let detector = ModifierKeyDetector(combo: KeyCombo(keyCode: keyCode))
                    {
                        loneModifier = detector
                    }
                    switch loneModifier?.flagsChanged(keyCode: keyCode, rawFlags: rawFlags) {
                    case .up:
                        finish(KeyCombo(keyCode: keyCode))
                    case .cancel:
                        loneModifier = nil
                    case .down, nil:
                        break
                    }
                    return event
                }
                loneModifier = nil
                let modifiers = event.modifierFlags.intersection([
                    .command, .option, .control, .shift, .function,
                ])

                // Escape cancels recording
                if keyCode == UInt16(kVK_Escape) {
                    finish(nil)
                    return nil
                }

                finish(KeyCombo(keyCode: keyCode, modifiers: modifiers))
                return nil
            }

            // Timeout after 10 seconds
            Task { @MainActor in
                try? await Task.sleep(for: .seconds(10))
                finish(nil)
            }
        }
    }

    /// End a recording in progress with no new hotkey (the recorder's
    /// Cancel): listening resumes at once instead of after the timeout.
    func cancelRecording() {
        cancelPendingRecording?()
    }

}
