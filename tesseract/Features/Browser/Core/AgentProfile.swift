import Foundation
import WebKit

// MARK: - AgentProfile

/// The **Agent Profile** — the persistent credential silo the **Agent Browser**
/// drives. A single dedicated `WKWebsiteDataStore`, identified by a stable UUID
/// so cookies and local storage survive relaunches, and kept distinct from
/// `WKWebsiteDataStore.default()` (and therefore from anything else in the app
/// that might touch WebKit). Nothing is ever imported from Safari or Chrome:
/// the user logs into sites deliberately, inside the Agent Browser, and *that*
/// curation is the security boundary (ADR-0026).
@MainActor
final class AgentProfile {

    /// Stable identity for the profile's on-disk data store. Hard-coded so the
    /// same store is reopened on every launch. Never reuse this UUID for any
    /// other store.
    static let dataStoreIdentifier = UUID(uuidString: "7E55EC1A-0B17-4E2A-9F3C-A9D0B7C61F42")!

    /// User-agent string presented to sites by the Agent Browser. A stock
    /// desktop Safari UA — sites gate features and layouts on it, and a custom
    /// token trips bot heuristics on exactly the authenticated sites this is
    /// meant to reach.
    static let userAgent =
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        + "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Safari/605.1.15"

    let dataStore: WKWebsiteDataStore

    /// Designated init — inject the store. Tests pass `.nonPersistent()` so they
    /// never touch (or contend on) the real on-disk profile.
    init(dataStore: WKWebsiteDataStore) {
        self.dataStore = dataStore
    }

    /// Production profile: the persistent, stable-identity store.
    convenience init() {
        self.init(dataStore: WKWebsiteDataStore(forIdentifier: Self.dataStoreIdentifier))
    }

    /// A `WebPage.Configuration` bound to the persistent profile store — every
    /// tab in every **Browser Session** shares this data store, so a login in
    /// one session is a login for all (concurrent sessions share auth, never
    /// tabs).
    func makePageConfiguration() -> WebPage.Configuration {
        var config = WebPage.Configuration()
        config.websiteDataStore = dataStore
        // The UA is applied per-page via `WebPage.customUserAgent` (which fully
        // replaces the string) in BrowserTab / the login window — not here:
        // `applicationNameForUserAgent` only *appends* to the default UA, so
        // setting it to the full Safari string would malform the UA if it ever
        // took effect. One source of truth, applied correctly.
        return config
    }
}
