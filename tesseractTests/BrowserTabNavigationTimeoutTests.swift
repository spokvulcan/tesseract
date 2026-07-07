import Foundation
import Testing
import WebKit

@testable import Tesseract_Agent

// MARK: - BrowserTabNavigationTimeoutTests

/// Regression cover for the browser-use freeze: a navigation whose WebKit event
/// stream never yields `.finished` (and never ends or errors) — the shape a
/// back-forward-cache restore can take — must raise ``BrowserTabError/timeout``
/// in bounded time, not hang forever.
///
/// Before the fix, `runNavigation` created an orphaned timeout `Task` that never
/// cancelled the event loop, so this exact scenario hung until the user aborted
/// (observed: `browser.back` ran 2m17s against a 30s configured timeout).
@MainActor
struct BrowserTabNavigationTimeoutTests {

    /// A navigation event stream that never emits and never finishes.
    private func stalledEvents() -> AsyncThrowingStream<WebPage.NavigationEvent, any Error> {
        AsyncThrowingStream { _ in /* hold the continuation open forever */ }
    }

    @Test func navigationTimesOutInsteadOfHangingForever() async {
        let tab = BrowserTab(
            configuration: WebPage.Configuration(),
            navigationTimeout: .milliseconds(150))

        let start = ContinuousClock.now
        var caught: Error?
        do {
            try await tab.runNavigation(stalledEvents())
        } catch {
            caught = error
        }
        let elapsed = ContinuousClock.now - start

        // The timeout fired: `.timeout`, and nowhere near a real hang.
        guard case .timeout? = caught as? BrowserTabError else {
            Issue.record("expected BrowserTabError.timeout, got \(String(describing: caught))")
            return
        }
        #expect(elapsed < .seconds(5))
    }

    @Test func withTimeoutReturnsFastResultUnchanged() async throws {
        let value = try await BrowserTab.withTimeout(.seconds(5)) { 42 }
        #expect(value == 42)
    }

    @Test func withTimeoutRaisesTimeoutOnAStuckOperation() async {
        let start = ContinuousClock.now
        var caught: Error?
        do {
            _ = try await BrowserTab.withTimeout(.milliseconds(150)) {
                try await Task.sleep(for: .seconds(60))  // never completes in budget
                return 0
            }
        } catch {
            caught = error
        }
        let elapsed = ContinuousClock.now - start

        guard case .timeout? = caught as? BrowserTabError else {
            Issue.record("expected BrowserTabError.timeout, got \(String(describing: caught))")
            return
        }
        #expect(elapsed < .seconds(5))
    }
}
