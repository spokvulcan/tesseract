import Foundation
import Observation
import WebKit

// MARK: - AgentBrowser

/// The **Agent Browser**: the app-owned WebKit browser and the one owner of the
/// **Agent Profile**. Hands out a private **Browser Session** per MCP client,
/// all sharing the profile's persistent logins (ADR-0026/0027). The single
/// arbiter of the browser across every client — Tesseract's own agent connects
/// as just another client through its MCP client (PRD #190).
///
/// `@Observable` so the Settings surface can reach it through the SwiftUI
/// environment to open the user login window.
@MainActor
@Observable
final class AgentBrowser {

    let profile: AgentProfile
    private let presenter: AgentBrowserPresenting?
    private var sessions: [String: BrowserSession] = [:]

    init(profile: AgentProfile = AgentProfile(), presenter: AgentBrowserPresenting? = nil) {
        self.profile = profile
        self.presenter = presenter
    }

    /// Get or create the session for `id` (an MCP session id).
    func session(id: String) -> BrowserSession {
        if let existing = sessions[id] { return existing }
        let session = BrowserSession(id: id, profile: profile, presenter: presenter)
        sessions[id] = session
        return session
    }

    /// Look up an existing session without creating one.
    func existingSession(id: String) -> BrowserSession? {
        sessions[id]
    }

    /// Close one client's session and its window.
    func closeSession(id: String) {
        sessions.removeValue(forKey: id)?.teardown()
    }

    /// Close every session (server stop / app termination).
    func closeAll() {
        for session in sessions.values { session.teardown() }
        sessions.removeAll()
    }

    var sessionCount: Int { sessions.count }

    // MARK: - User login window

    /// The single user-facing page (User Story #1), retained so reopening the
    /// window returns to where the user left off.
    private var userPage: WebPage?

    /// Open — or resurface — the user's own Agent Browser window on the shared
    /// profile: the deliberate surface for logging into the sites they want to
    /// grant agents. Cookies set here persist in the profile, so every Browser
    /// Session is authenticated afterwards.
    func openUserBrowserWindow() {
        let page: WebPage
        if let existing = userPage {
            page = existing
        } else {
            let fresh = WebPage(configuration: profile.makePageConfiguration())
            fresh.customUserAgent = AgentProfile.userAgent
            userPage = fresh
            page = fresh
        }
        presenter?.presentUserBrowser(page: page)
    }
}
