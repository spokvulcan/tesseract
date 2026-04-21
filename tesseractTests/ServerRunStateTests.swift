import SwiftUI
import Testing
@testable import Tesseract_Agent

struct ServerRunStateTests {

    @Test
    func enabledServerUsesStartingOnlyWhileListenerIsPending() {
        let state = ServerRunState(
            enabled: true,
            isRunning: false,
            isStarting: true,
            lastStartError: nil
        )

        #expect(state == .starting)
        #expect(state.displayLabel == "Starting…")
        #expect(state.dotColor == Color.orange)
    }

    @Test
    func enabledServerWithoutPendingStartDoesNotMasqueradeAsStarting() {
        let state = ServerRunState(
            enabled: true,
            isRunning: false,
            isStarting: false,
            lastStartError: nil
        )

        #expect(state == .stopped)
        #expect(state.displayLabel == "Stopped")
    }

    @Test
    func enabledServerSurfacesFailureAheadOfStoppedState() {
        let state = ServerRunState(
            enabled: true,
            isRunning: false,
            isStarting: false,
            lastStartError: "Bind failed"
        )

        #expect(state == .failed("Bind failed"))
        #expect(state.failureMessage == "Bind failed")
    }
}
