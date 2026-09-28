//
//  DictationLabWatcher.swift
//  tesseract
//
//  PROTOTYPE — THROWAWAY (Dictation page redesign, 2026-09-28). Never merge.
//
//  Variant D's zero-action teacher: after a take lands, read the field it
//  landed in through Accessibility, keep finding the dictated words between
//  the text around them, and when the owner edits them in place, learn the
//  change once the edit settles (focus leaves, the text is sent, the next
//  take starts, or 60 s pass). The same windows VoiceInk's Auto Learn uses
//  (correction UX note §2). Apps whose text Accessibility can't see are
//  reported as blind, and the fix shortcut covers them.
//

import AppKit
import ApplicationServices
import Observation

@Observable @MainActor
final class LabEditWatcher {
    enum State: Equatable {
        case idle
        case watching(app: String, until: Date)
        case blind(app: String)
    }

    private(set) var state: State = .idle
    /// Apps seen this session, and whether their text could be watched.
    private(set) var apps: [String: Bool] = [:]

    @ObservationIgnored private weak var lab: DictationLab?
    @ObservationIgnored private var task: Task<Void, Never>?

    static let window: Duration = .seconds(60)

    init(lab: DictationLab) {
        self.lab = lab
    }

    func stop() {
        task?.cancel()
        task = nil
        if case .watching = state { state = .idle }
    }

    func watch(_ take: LabTake) {
        stop()
        task = Task { [weak self] in
            await self?.run(take)
        }
    }

    /// Terminals expose their whole scrollback as one read-only string and
    /// post no edit notifications; watching them costs the terminal's main
    /// thread and sees nothing useful (AX research note §2).
    static let blindBundleIDs: Set<String> = [
        "com.apple.Terminal", "com.googlecode.iterm2", "com.mitchellh.ghostty",
        "dev.warp.Warp-Stable",
        "net.kovidgoyal.kitty", "org.alacritty", "com.github.wez.wezterm", "dev.zed.Zed",
    ]

    private func run(_ take: LabTake) async {
        // Accessibility is never switched on in apps that keep it off
        // (Electron's switch puts the whole app, VS Code included, into
        // screen-reader mode), and terminals are never read.
        if let bundle = take.bundleID, Self.blindBundleIDs.contains(bundle) {
            markBlind(take)
            return
        }
        // Give the paste a moment to land.
        try? await Task.sleep(for: .milliseconds(200))
        guard !Task.isCancelled else { return }
        let words = take.text.trimmingCharacters(in: .whitespaces)
        guard let element = LabAX.focusedElement(), LabAX.pid(of: element) == take.pid,
            !LabAX.isSecure(element), let value = LabAX.value(of: element),
            let found = value.range(of: words, options: .backwards)
        else {
            markBlind(take)
            return
        }
        apps[take.appName] = true
        let before = String(value[value.startIndex..<found.lowerBound].suffix(24))
        let after = String(value[found.upperBound...].prefix(24))
        let deadline = Date().addingTimeInterval(60)
        state = .watching(app: take.appName, until: deadline)
        var lastRegion = words

        while !Task.isCancelled, Date() < deadline {
            try? await Task.sleep(for: .milliseconds(400))
            guard !Task.isCancelled else { return }
            // A new take ends this one's window.
            if lab?.lastTake?.id != take.id { break }
            // Focus left the field: the edit is done.
            guard let focused = LabAX.focusedElement(), CFEqual(focused, element) else { break }
            guard let current = LabAX.value(of: element) else { break }
            guard let region = Self.region(in: current, before: before, after: after) else {
                // The text is gone (sent, or cleared): keep what was seen last.
                break
            }
            lastRegion = region.trimmingCharacters(in: .whitespaces)
        }
        state = .idle
        guard !Task.isCancelled, lastRegion != words, !lastRegion.isEmpty else { return }
        lab?.learn(before: words, after: lastRegion, source: .watched, appName: take.appName)
    }

    private func markBlind(_ take: LabTake) {
        state = .blind(app: take.appName)
        apps[take.appName] = false
        lab?.showToast(
            LabToast(
                title: "Can't see the text in \(take.appName)",
                detail: "Fix it with ⌃⌥Space instead", isWarning: true))
    }

    /// The text between two anchors, or nil when either is gone.
    static func region(in text: String, before: String, after: String) -> String? {
        let start: String.Index
        if before.isEmpty {
            start = text.startIndex
        } else {
            guard let r = text.range(of: before, options: .backwards) else { return nil }
            start = r.upperBound
        }
        let end: String.Index
        if after.isEmpty {
            end = text.endIndex
        } else {
            guard let r = text.range(of: after, range: start..<text.endIndex) else { return nil }
            end = r.lowerBound
        }
        guard start <= end else { return nil }
        return String(text[start..<end])
    }
}
