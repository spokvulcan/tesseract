import AppKit
import SwiftUI
import WebKit

// MARK: - AgentBrowserWindowPresenter

/// Production presenter: renders each Browser Session's active tab in a real,
/// visible `NSWindow` (ADR-0026 — always-visible browsing). One window per
/// session, retitled and re-pointed as the agent navigates or switches tabs.
/// The user can watch, scroll, and intervene; closing the window just drops the
/// reference, and the next navigation reopens it.
@MainActor
final class AgentBrowserWindowPresenter: NSObject, AgentBrowserPresenting {

    private final class Entry {
        let window: NSWindow
        var shownPageID: ObjectIdentifier?
        init(window: NSWindow) { self.window = window }
    }

    private var entries: [String: Entry] = [:]
    /// Reverse lookup so the window-close delegate can find its session.
    private var windowToSession: [ObjectIdentifier: String] = [:]
    private var cascadePoint = NSPoint(x: 140, y: 140)

    func present(sessionID: String, page: WebPage, title: String) {
        let entry = entries[sessionID] ?? makeEntry(for: sessionID)

        let pageID = ObjectIdentifier(page)
        if entry.shownPageID != pageID {
            entry.window.contentView = NSHostingView(rootView: WebView(page))
            entry.shownPageID = pageID
        }
        entry.window.title = title.isEmpty ? "Agent Browser" : title

        if !entry.window.isVisible {
            entry.window.makeKeyAndOrderFront(nil)
        } else {
            entry.window.orderFront(nil)
        }
    }

    func close(sessionID: String) {
        guard let entry = entries.removeValue(forKey: sessionID) else { return }
        windowToSession[ObjectIdentifier(entry.window)] = nil
        entry.window.close()
    }

    // MARK: - Private

    private func makeEntry(for sessionID: String) -> Entry {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1000, height: 760),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.isReleasedWhenClosed = false
        window.title = "Agent Browser"
        window.tabbingMode = .disallowed
        cascadePoint = window.cascadeTopLeft(from: cascadePoint)
        window.delegate = self

        let entry = Entry(window: window)
        entries[sessionID] = entry
        windowToSession[ObjectIdentifier(window)] = sessionID
        return entry
    }
}

// MARK: - NSWindowDelegate

extension AgentBrowserWindowPresenter: NSWindowDelegate {
    /// When the user closes a session window, forget it (without tearing the
    /// session down) so the next `present` reopens a fresh window.
    func windowWillClose(_ notification: Notification) {
        guard let window = notification.object as? NSWindow,
            let sessionID = windowToSession.removeValue(forKey: ObjectIdentifier(window))
        else { return }
        entries[sessionID] = nil
    }
}
