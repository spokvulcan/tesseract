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
}

@MainActor
final class HotkeyManager: ObservableObject {
    @Published var currentHotkey: KeyCombo = .optionSpace {
        didSet {
            // Keep backward compat: updating currentHotkey updates the "dictation" registration
            if var reg = registrations["dictation"] {
                reg = HotkeyRegistration(
                    id: "dictation", combo: currentHotkey, onDown: reg.onDown, onUp: reg.onUp)
                registrations["dictation"] = reg
            }
        }
    }
    @Published private(set) var isListening = false
    @Published private(set) var isUsingEventTap = false

    var onHotkeyDown: (() -> Void)? {
        didSet {
            // Backward compat: sync to dictation registration
            syncDictationRegistration()
        }
    }
    var onHotkeyUp: (() -> Void)? {
        didSet {
            syncDictationRegistration()
        }
    }

    private var registrations: [String: HotkeyRegistration] = [:]

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
        id: String, combo: KeyCombo, onDown: @escaping () -> Void, onUp: (() -> Void)? = nil
    ) {
        registrations[id] = HotkeyRegistration(id: id, combo: combo, onDown: onDown, onUp: onUp)
    }

    func unregisterHotkey(id: String) {
        registrations.removeValue(forKey: id)
        matcher.forget(id: id)
    }

    func updateRegisteredHotkey(id: String, combo: KeyCombo) {
        guard var reg = registrations[id] else { return }
        reg = HotkeyRegistration(id: id, combo: combo, onDown: reg.onDown, onUp: reg.onUp)
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

                let keyCode = UInt16(event.getIntegerValueField(.keyboardEventKeycode))
                let flags = event.flags

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
                    kind = .flagsChanged
                case .keyUp:
                    kind = .keyUp
                default:
                    kind = .keyDown
                }

                let verdict = manager.matcher.handle(
                    kind, keyCode: keyCode, modifiers: modifiers,
                    bindings: manager.registrations.mapValues(\.combo))

                // Deliver on the next main-queue turn so the tap callback
                // stays fast; the matcher state is already settled.
                manager.deliver(verdict.fires, deferred: true)

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

        runLoopSource = CFMachPortCreateRunLoopSource(nil, eventTap, 0)
        CFRunLoopAddSource(CFRunLoopGetMain(), runLoopSource, .commonModes)
        CGEvent.tapEnable(tap: eventTap, enable: true)

        isUsingEventTap = true
    }

    private func stopEventTap() {
        if let eventTap = eventTap {
            CGEvent.tapEnable(tap: eventTap, enable: false)
            CFMachPortInvalidate(eventTap)
            self.eventTap = nil
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
            matching: [.keyDown, .keyUp, .flagsChanged]
        ) { [weak self] event in
            Task { @MainActor in
                self?.handleKeyEvent(event)
            }
        }

        // Local monitor for when app is focused
        localMonitor = NSEvent.addLocalMonitorForEvents(
            matching: [.keyDown, .keyUp, .flagsChanged]
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
        let kind: HotkeyMatcher.EventKind
        switch event.type {
        case .flagsChanged:
            handleDoubleCommandFlags(rawFlags: UInt64(event.modifierFlags.rawValue))
            kind = .flagsChanged
        case .keyUp:
            kind = .keyUp
        default:
            kind = .keyDown
        }

        let verdict = matcher.handle(
            kind, keyCode: event.keyCode, modifiers: event.modifierFlags,
            bindings: registrations.mapValues(\.combo))

        // Monitors cannot suppress events; deliver synchronously.
        deliver(verdict.fires, deferred: false)
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

    // MARK: - Permission Handling

    func refreshForAccessibilityPermission() {
        guard isListening else { return }

        if !isUsingEventTap && AXIsProcessTrusted() {
            stopNSEventMonitors()
            startEventTap()
        }
    }

    // MARK: - Hotkey Recording

    func recordHotkey() async -> KeyCombo? {
        return await withCheckedContinuation { continuation in
            var hasResumed = false
            var monitor: Any?
            let shouldResumeListening = isListening

            if shouldResumeListening {
                stopListening()
            }

            func finish(_ result: KeyCombo?) {
                guard !hasResumed else { return }
                hasResumed = true
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

            monitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown]) { event in
                let keyCode = event.keyCode
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

    func updateHotkey(_ combo: KeyCombo) {
        currentHotkey = combo
        matcher.reset()
    }

    // MARK: - Private

    private func syncDictationRegistration() {
        if let onDown = onHotkeyDown {
            registrations["dictation"] = HotkeyRegistration(
                id: "dictation",
                combo: currentHotkey,
                onDown: onDown,
                onUp: onHotkeyUp
            )
        }
    }
}
