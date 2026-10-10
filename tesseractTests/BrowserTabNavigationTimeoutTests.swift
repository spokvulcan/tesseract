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

    /// A wait that ignores cancellation, like a WebKit callback that never
    /// comes. Only the backstop ends it, long after any budget here, so a
    /// `withTimeout` that waits for its operation fails these tests instead
    /// of hanging the run.
    private static func wedged() async {
        await withCheckedContinuation { continuation in
            Task {
                try? await Task.sleep(for: waitBackstop)
                continuation.resume()
            }
        }
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

    /// `runNavigation` maps a `WebPage.NavigationError` the operation throws,
    /// so the race must hand an operation's own error back as it was thrown.
    @Test func withTimeoutRethrowsFastErrorUnchanged() async {
        var caught: Error?
        do {
            try await BrowserTab.withTimeout(.seconds(5)) {
                throw BrowserTabError.noHistory
            }
        } catch {
            caught = error
        }

        guard case .noHistory? = caught as? BrowserTabError else {
            Issue.record("expected BrowserTabError.noHistory, got \(String(describing: caught))")
            return
        }
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

    /// The operation is cancelled at the timeout but never checks for it, the
    /// way a WebKit await that never resumes doesn't: the caller still gets
    /// `.timeout` on time, and the operation is left to finish on its own.
    @Test func withTimeoutAbandonsAnOperationThatIgnoresCancellation() async {
        let start = ContinuousClock.now
        var caught: Error?
        do {
            _ = try await BrowserTab.withTimeout(.milliseconds(150)) {
                await Self.wedged()
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

    /// A cancelled caller is released at once with `CancellationError`, even
    /// while the operation ignores the cancellation passed on to it.
    @Test func withTimeoutReleasesCancelledCallerFromAWedgedOperation() async {
        let call = Task {
            try await BrowserTab.withTimeout(waitBackstop) {
                await Self.wedged()
                return 0
            }
        }
        let start = ContinuousClock.now
        call.cancel()
        let result = await call.result
        let elapsed = ContinuousClock.now - start

        guard case .failure(let error) = result, error is CancellationError else {
            Issue.record("expected CancellationError, got \(result)")
            return
        }
        #expect(elapsed < .seconds(5))
    }
}
