import Foundation
import WebKit

// MARK: - AsyncSerialGate

/// Minimal MainActor mutual-exclusion gate. Serializes a Browser Session's own
/// tool calls so a single client pipelining requests can't interleave the
/// multi-await steps of two operations (e.g. `navigate` mid-load while
/// `read_page` grabs the DOM). Cross-session concurrency is unaffected —
/// different sessions drive different tabs.
@MainActor
final class AsyncSerialGate {
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func acquire() async {
        if !busy {
            busy = true
            return
        }
        await withCheckedContinuation { waiters.append($0) }
    }

    func release() {
        if waiters.isEmpty {
            busy = false
        } else {
            waiters.removeFirst().resume()
        }
    }
}

// MARK: - TabSummary

/// A row of the `tabs` tool listing.
nonisolated struct TabSummary: Sendable, Equatable {
    let id: String
    let url: String
    let title: String
    let active: Bool
}

// MARK: - BrowserSession

/// One MCP client's private set of tabs over the shared **Agent Profile**
/// (ADR-0027). Sessions never see each other's tabs; login state is common to
/// all because every tab is configured from the same persistent profile store.
///
/// The session owns tab lifecycle and drives the window presenter whenever the
/// active tab or its title/URL changes.
@MainActor
final class BrowserSession {

    let id: String
    private let profile: AgentProfile
    private weak var presenter: AgentBrowserPresenting?
    private let gate = AsyncSerialGate()

    private var tabs: [BrowserTab] = []
    private var activeTabID: String?

    init(id: String, profile: AgentProfile, presenter: AgentBrowserPresenting?) {
        self.id = id
        self.profile = profile
        self.presenter = presenter
    }

    // MARK: Serialization

    /// Run `body` with the session's other tool calls excluded.
    func serialized<T: Sendable>(_ body: @MainActor () async throws -> T) async rethrows -> T {
        await gate.acquire()
        defer { gate.release() }
        return try await body()
    }

    // MARK: Tabs

    var activeTab: BrowserTab? {
        guard let activeTabID else { return tabs.first }
        return tabs.first { $0.id == activeTabID } ?? tabs.first
    }

    /// The active tab, creating a first blank tab on demand. Navigation and
    /// reads flow through here so a fresh session is immediately usable.
    @discardableResult
    func requireActiveTab() -> BrowserTab {
        if let tab = activeTab { return tab }
        return openTab()
    }

    @discardableResult
    func openTab() -> BrowserTab {
        let tab = BrowserTab(configuration: profile.makePageConfiguration())
        tabs.append(tab)
        activeTabID = tab.id
        reflect()
        return tab
    }

    func selectTab(id: String) -> Bool {
        guard tabs.contains(where: { $0.id == id }) else { return false }
        activeTabID = id
        reflect()
        return true
    }

    func closeTab(id: String) -> Bool {
        guard let index = tabs.firstIndex(where: { $0.id == id }) else { return false }
        tabs.remove(at: index)
        if activeTabID == id {
            activeTabID = tabs.last?.id
        }
        if tabs.isEmpty {
            presenter?.close(sessionID: self.id)
        } else {
            reflect()
        }
        return true
    }

    func tabSummaries() -> [TabSummary] {
        tabs.map { tab in
            TabSummary(
                id: tab.id,
                url: tab.status.url,
                title: tab.status.title,
                active: tab.id == (activeTabID ?? tabs.first?.id)
            )
        }
    }

    /// Push the current active-tab state to the window presenter. Call after any
    /// navigation/interaction so the visible window tracks the agent.
    func reflect() {
        guard let tab = activeTab else { return }
        let status = tab.status
        let title = status.title.isEmpty ? status.url : status.title
        presenter?.present(sessionID: id, page: tab.page, title: title)
    }

    /// Tear the session down: forget tabs and close its window.
    func teardown() {
        tabs.removeAll()
        activeTabID = nil
        presenter?.close(sessionID: id)
    }
}
