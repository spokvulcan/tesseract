import Foundation
import WebKit

// MARK: - AgentBrowserPresenting

/// The window seam for the **Agent Browser**. ADR-0026 makes browsing
/// *always visible* — every Browser Session's active tab renders in a real
/// window so the user can watch and intervene. Keeping presentation behind a
/// port lets the core (navigation, extraction, tools) run windowless in tests
/// via ``NoOpBrowserPresenter`` while production shows real windows.
///
/// Calls are idempotent per session: `present` both creates the window and
/// reflects later navigations/tab-switches (updated page + title).
@MainActor
protocol AgentBrowserPresenting: AnyObject {
    /// Show (or update) the window for `sessionID`, displaying `page`.
    func present(sessionID: String, page: WebPage, title: String)
    /// Close and forget the window for `sessionID`.
    func close(sessionID: String)
}

// MARK: - NoOpBrowserPresenter

/// A presenter that shows nothing — the test double, and the safe default when
/// no window UI is wired. Lets the browser core be driven headlessly.
@MainActor
final class NoOpBrowserPresenter: AgentBrowserPresenting {
    /// Records the last presented state per session, purely so tests can assert
    /// the core *asked* for presentation without any window appearing.
    private(set) var presented: [String: (title: String, url: String?)] = [:]

    func present(sessionID: String, page: WebPage, title: String) {
        presented[sessionID] = (title, page.url?.absoluteString)
    }

    func close(sessionID: String) {
        presented[sessionID] = nil
    }
}
