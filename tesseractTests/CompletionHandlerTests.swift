import Foundation
import Testing
@testable import Tesseract_Agent

struct CompletionHandlerTests {

    // MARK: - LeaseAcquiredSignal

    @Test func signalStartsFalse() {
        let signal = LeaseAcquiredSignal()
        #expect(!signal.isSet)
    }

    @Test func signalBecomesTrue() {
        let signal = LeaseAcquiredSignal()
        signal.set()
        #expect(signal.isSet)
    }

    @Test func signalSetIsIdempotent() {
        let signal = LeaseAcquiredSignal()
        signal.set()
        signal.set()
        #expect(signal.isSet)
    }

    // MARK: - withAcquisitionTimeout

    @Test func timeoutThrowsWhenBodyNeverSignals() async {
        do {
            try await CompletionHandler.withAcquisitionTimeout(
                timeoutNanoseconds: 50_000_000 // 50ms
            ) { _ in
                // Simulate waiting in arbiter queue — never signal
                try await Task.sleep(nanoseconds: 5_000_000_000)
            }
            Issue.record("Expected LeaseTimeoutError")
        } catch is LeaseTimeoutError {
            // Expected
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test func longBodyNotCancelledAfterSignal() async throws {
        let completed = LeaseAcquiredSignal()

        try await CompletionHandler.withAcquisitionTimeout(
            timeoutNanoseconds: 100_000_000 // 100ms
        ) { signal in
            signal.set()
            // Simulate generation that takes longer than timeout
            try await Task.sleep(nanoseconds: 300_000_000) // 300ms > 100ms
            completed.set()
        }

        // Body must complete — timeout should not have cancelled it
        #expect(completed.isSet)
    }

    @Test func fastBodyCompletesBeforeTimeout() async throws {
        let completed = LeaseAcquiredSignal()

        try await CompletionHandler.withAcquisitionTimeout(
            timeoutNanoseconds: 1_000_000_000 // 1s
        ) { signal in
            signal.set()
            completed.set()
        }

        #expect(completed.isSet)
    }

    @Test func bodyErrorPropagatesNotTimeout() async {
        struct BodyError: Error {}

        do {
            try await CompletionHandler.withAcquisitionTimeout(
                timeoutNanoseconds: 1_000_000_000
            ) { signal in
                signal.set()
                throw BodyError()
            }
            Issue.record("Expected BodyError")
        } catch is BodyError {
            // Body error propagates, not LeaseTimeoutError
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    @Test func bodyErrorBeforeSignalPropagates() async {
        struct EarlyError: Error {}

        do {
            try await CompletionHandler.withAcquisitionTimeout(
                timeoutNanoseconds: 1_000_000_000
            ) { _ in
                // Error before signaling (e.g. arbiter load failure)
                throw EarlyError()
            }
            Issue.record("Expected EarlyError")
        } catch is EarlyError {
            // Expected
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }
}
