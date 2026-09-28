//
//  DictationLabAccess.swift
//  tesseract
//
//  PROTOTYPE — THROWAWAY (Dictation page redesign, 2026-09-28). Never merge.
//
//  Reaching the text in other apps: Accessibility reads of the focused
//  field, putting a fix back where the dictated words were, and reading a
//  selection. Accessibility first; keystrokes only when nothing was typed
//  since the insertion, so a fix can never delete the wrong text.
//

import AppKit
import ApplicationServices
import Carbon.HIToolbox

@MainActor
enum LabAX {
    static func focusedElement() -> AXUIElement? {
        let system = AXUIElementCreateSystemWide()
        AXUIElementSetMessagingTimeout(system, 0.25)
        var value: CFTypeRef?
        guard
            AXUIElementCopyAttributeValue(system, kAXFocusedUIElementAttribute as CFString, &value)
                == .success, let value, CFGetTypeID(value) == AXUIElementGetTypeID()
        else { return nil }
        let element = unsafeBitCast(value, to: AXUIElement.self)
        AXUIElementSetMessagingTimeout(element, 0.25)
        return element
    }

    static func string(_ element: AXUIElement, _ attribute: String) -> String? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, attribute as CFString, &value) == .success
        else { return nil }
        return value as? String
    }

    static func value(of element: AXUIElement) -> String? {
        string(element, kAXValueAttribute)
    }

    static func selectedText(of element: AXUIElement) -> String? {
        string(element, kAXSelectedTextAttribute)
    }

    /// An editable text field: its value can be set. Terminals expose their
    /// text read-only, and a paste there types at the cursor instead of
    /// replacing a selection.
    static func isEditableText(_ element: AXUIElement) -> Bool {
        var settable: DarwinBoolean = false
        guard
            AXUIElementIsAttributeSettable(element, kAXValueAttribute as CFString, &settable)
                == .success
        else { return false }
        return settable.boolValue
    }

    static func isSecure(_ element: AXUIElement) -> Bool {
        let role = string(element, kAXRoleAttribute) ?? ""
        let subrole = string(element, kAXSubroleAttribute) ?? ""
        return role == "AXSecureTextField" || subrole == "AXSecureTextField"
    }

    static func selectedRange(of element: AXUIElement) -> CFRange? {
        var value: CFTypeRef?
        guard
            AXUIElementCopyAttributeValue(
                element, kAXSelectedTextRangeAttribute as CFString, &value) == .success,
            let value, CFGetTypeID(value) == AXValueGetTypeID()
        else { return nil }
        var range = CFRange()
        guard AXValueGetValue(unsafeBitCast(value, to: AXValue.self), .cfRange, &range) else {
            return nil
        }
        return range
    }

    @discardableResult
    static func setSelectedRange(_ range: CFRange, on element: AXUIElement) -> Bool {
        var range = range
        guard let value = AXValueCreate(.cfRange, &range) else { return false }
        return AXUIElementSetAttributeValue(
            element, kAXSelectedTextRangeAttribute as CFString, value) == .success
    }

    /// Screen rect of a text range, in AppKit coordinates (origin bottom-left).
    static func bounds(of range: CFRange, in element: AXUIElement) -> CGRect? {
        var range = range
        guard let param = AXValueCreate(.cfRange, &range) else { return nil }
        var value: CFTypeRef?
        guard
            AXUIElementCopyParameterizedAttributeValue(
                element, kAXBoundsForRangeParameterizedAttribute as CFString, param, &value)
                == .success,
            let value, CFGetTypeID(value) == AXValueGetTypeID()
        else { return nil }
        var rect = CGRect.zero
        guard AXValueGetValue(unsafeBitCast(value, to: AXValue.self), .cgRect, &rect),
            rect.width >= 0
        else {
            return nil
        }
        // AX rects are top-left based on the primary screen.
        let primaryHeight = NSScreen.screens.first?.frame.height ?? 0
        return CGRect(
            x: rect.minX, y: primaryHeight - rect.maxY, width: rect.width, height: rect.height)
    }

    /// Chromium and Electron build their accessibility tree only on request.
    static func requestManualAccessibility(pid: pid_t) {
        let app = AXUIElementCreateApplication(pid)
        AXUIElementSetAttributeValue(app, "AXManualAccessibility" as CFString, kCFBooleanTrue)
    }

    static func pid(of element: AXUIElement) -> pid_t? {
        var pid: pid_t = 0
        return AXUIElementGetPid(element, &pid) == .success ? pid : nil
    }
}

/// Where a fix could be put back, and how.
nonisolated enum LabReplaceOutcome: Equatable, Sendable {
    /// Selected through Accessibility and pasted over.
    case replacedInPlace
    /// Backspaced over the unchanged insertion and retyped.
    case retyped
    /// The target moved on (another app, or typing since); the fix was
    /// learned and the corrected text left on the clipboard.
    case copied(reason: String)
}

@MainActor
final class LabTextReplacer {
    private let injector: any TextInjecting
    private let keyDownCount: @MainActor () -> Int

    init(injector: any TextInjecting, keyDownCount: @escaping @MainActor () -> Int) {
        self.injector = injector
        self.keyDownCount = keyDownCount
    }

    /// Puts `corrected` where `take`'s words were inserted. `allowedKeys` is
    /// the number of key presses the fix itself cost (the shortcut).
    func replace(_ take: LabTake, with corrected: String, allowedKeys: Int) async
        -> LabReplaceOutcome
    {
        let front = NSWorkspace.shared.frontmostApplication
        guard front?.processIdentifier == take.pid else {
            return copy(corrected, reason: "\(take.appName) is no longer in front")
        }
        let old = take.insertedText
        let new = corrected + " "

        // 1. Accessibility: find the inserted words right before the caret.
        if let element = LabAX.focusedElement(), !LabAX.isSecure(element),
            LabAX.isEditableText(element),
            let value = LabAX.value(of: element), let caret = LabAX.selectedRange(of: element)
        {
            let utf16 = Array(value.utf16)
            let target = Array(old.utf16)
            let trimmedTarget = Array(old.trimmingCharacters(in: .whitespaces).utf16)
            var start: Int?
            var length = 0
            for candidate in [target, trimmedTarget] where !candidate.isEmpty {
                let end = caret.location
                if end >= candidate.count, end <= utf16.count,
                    Array(utf16[(end - candidate.count)..<end]) == candidate
                {
                    start = end - candidate.count
                    length = candidate.count
                    break
                }
            }
            if start == nil, let r = Self.lastRange(of: trimmedTarget, in: utf16) {
                start = r.lowerBound
                length = r.count
            }
            if let start {
                LabAX.setSelectedRange(CFRange(location: start, length: length), on: element)
                try? await Task.sleep(for: .milliseconds(40))
                let replacement = length == target.count ? new : corrected
                if (try? await injector.inject(replacement)) != nil {
                    return .replacedInPlace
                }
            }
        }

        // 2. Keystrokes: only if nothing was typed since the insertion.
        let typedSince = keyDownCount() - take.keyCountAtInsertion
        guard typedSince <= allowedKeys else {
            return copy(corrected, reason: "you typed since the dictation")
        }
        let common = Self.commonPrefixCount(old, new)
        let toDelete = old.count - common
        let toType = String(new.dropFirst(common))
        Self.postBackspaces(toDelete)
        try? await Task.sleep(for: .milliseconds(60))
        if !toType.isEmpty {
            guard (try? await injector.inject(toType)) != nil else {
                return copy(corrected, reason: "the paste was refused")
            }
        }
        return .retyped
    }

    func restoreClipboard(_ restore: Bool) {
        injector.restoreClipboard = restore
    }

    /// Pastes over the current selection in the frontmost app.
    func replaceSelection(with text: String) async -> Bool {
        (try? await injector.inject(text)) != nil
    }

    /// The selected text in the frontmost app, with where it sits on screen
    /// when Accessibility knows. Falls back to a copy.
    func readSelection() async -> (text: String, bounds: CGRect?)? {
        if let element = LabAX.focusedElement(), !LabAX.isSecure(element),
            let text = LabAX.selectedText(of: element), !text.isEmpty
        {
            let bounds = LabAX.selectedRange(of: element).flatMap {
                LabAX.bounds(of: $0, in: element)
            }
            return (text, bounds)
        }
        if let text = try? await TextExtractor().extractSelectedText(), !text.isEmpty {
            return (text, nil)
        }
        return nil
    }

    private func copy(_ text: String, reason: String) -> LabReplaceOutcome {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        return .copied(reason: reason)
    }

    private static func commonPrefixCount(_ a: String, _ b: String) -> Int {
        var count = 0
        for (x, y) in zip(a, b) {
            guard x == y else { break }
            count += 1
        }
        return count
    }

    private static func lastRange(of needle: [UInt16], in haystack: [UInt16]) -> Range<Int>? {
        guard !needle.isEmpty, haystack.count >= needle.count else { return nil }
        var i = haystack.count - needle.count
        while i >= 0 {
            if Array(haystack[i..<(i + needle.count)]) == needle { return i..<(i + needle.count) }
            i -= 1
        }
        return nil
    }

    static func postBackspaces(_ count: Int) {
        guard count > 0 else { return }
        let source = CGEventSource(stateID: .hidSystemState)
        for _ in 0..<count {
            CGEvent(keyboardEventSource: source, virtualKey: CGKeyCode(kVK_Delete), keyDown: true)?
                .post(tap: .cghidEventTap)
            CGEvent(keyboardEventSource: source, virtualKey: CGKeyCode(kVK_Delete), keyDown: false)?
                .post(tap: .cghidEventTap)
        }
    }
}
