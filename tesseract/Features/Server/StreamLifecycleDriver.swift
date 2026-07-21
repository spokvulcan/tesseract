import Foundation

/// Owns the transport-lifecycle race of one streaming HTTP completion: the
/// disconnect watch, the idle keepalive prober, and the drive, run together as
/// first-finisher-wins.
///
/// `CompletionHandler.runStreamingCompletion` builds the SSE envelope and then
/// hands three injected transport closures plus the drive to `run`; the driver
/// constructs the three-task group, returns the first outcome, and cancels the
/// losers. Pulling the race out of the handler gives it a name and a locality —
/// and makes the correctness property that matters under real clients (a client
/// abort must cancel a long prefill promptly) a table of driver tests rather
/// than an emergent behavior of a task group nobody constructs.
///
/// This sits *below* the ADR-0015 dispatcher seam: no new entry points, and the
/// handler keeps HTTP framing (SSE writing, chunk building). The driver knows
/// only the injected closures and the `CompletionHandler.StreamingOutcome`
/// currency the handler switches on.
///
/// `nonisolated` so it composes with the handler's off-actor streaming path with
/// zero isolation change; every closure it holds is `@Sendable`.
nonisolated enum StreamLifecycleDriver {

    /// The transport probes the driver races against the drive. In production
    /// these wrap the request's `HTTPResponseWriter` and `SSEWriter`; in tests
    /// they are scripted, so the race runs without a socket.
    struct Transport: Sendable {
        /// Suspends until the client connection drops (production:
        /// `HTTPResponseWriter.waitForDisconnect`). Must be cancellation-aware
        /// so it drains when another task wins the race.
        var waitForDisconnect: @Sendable () async -> Void
        /// True when the stream has been quiet for at least the given duration —
        /// the gate before a keepalive is written (production: `SSEWriter.idleFor`).
        var idleFor: @Sendable (Duration) async -> Bool
        /// Emit one keepalive probe; `false` ⇒ the write failed, i.e. the client
        /// is gone (production: `SSEWriter.keepalive`).
        var sendKeepalive: @Sendable () async -> Bool
    }

    /// Race the disconnect watch, the idle keepalive prober, and `drive` as
    /// first-finisher-wins, then cancel the losers.
    ///
    /// Semantics preserved verbatim from the handler's former task group:
    /// - The disconnect watch bridges the transport drop to `onTransportCancel`
    ///   (so a long prefill stops) and reports `.disconnected(.connectionState)`;
    ///   if it was cancelled first it yields `.cancelled` instead.
    /// - The keepalive prober wakes every `keepaliveCadence`, and only when the
    ///   stream is idle for the cadence it writes a keepalive; a failed write
    ///   bridges the cancel and reports `.disconnected(.keepaliveWrite)`. Its
    ///   own cancellation yields `.cancelled`; any other sleep error `.failed`.
    /// - `drive`'s outcome (completed / failed / cancelled / disconnected) passes
    ///   through unchanged.
    ///
    /// - Parameters:
    ///   - transport: the three scripted/real probe closures.
    ///   - keepaliveCadence: the prober's wake interval and idle threshold —
    ///     today's production value by default; tests shorten it to stay fast.
    ///   - onTransportCancel: the cancel bridge into generation (production:
    ///     `StartedGeneration.cancel`), invoked when the transport drops or a
    ///     keepalive write fails.
    ///   - drive: the generation stream pump (production:
    ///     `CompletionHandler.streamGenerationEvents`).
    static func run(
        transport: Transport,
        keepaliveCadence: Duration = .milliseconds(250),
        onTransportCancel: @escaping @Sendable () -> Void,
        drive: @escaping @Sendable () async -> CompletionHandler.StreamingOutcome
    ) async -> CompletionHandler.StreamingOutcome {
        await withTaskGroup(of: CompletionHandler.StreamingOutcome.self) { group in
            group.addTask {
                await transport.waitForDisconnect()
                guard !Task.isCancelled else { return .cancelled }
                onTransportCancel()
                return .disconnected(.connectionState)
            }

            group.addTask {
                // Keepalive: while the stream is idle, probe the transport
                // frequently so client aborts cancel long prefill promptly.
                while true {
                    do {
                        try await Task.sleep(for: keepaliveCadence)
                        try Task.checkCancellation()
                    } catch is CancellationError {
                        return .cancelled
                    } catch {
                        return .failed(error.localizedDescription)
                    }

                    guard await transport.idleFor(keepaliveCadence) else {
                        continue
                    }

                    guard await transport.sendKeepalive() else {
                        onTransportCancel()
                        return .disconnected(.keepaliveWrite)
                    }
                }
            }

            group.addTask {
                await drive()
            }

            let first = await group.next() ?? .cancelled
            group.cancelAll()
            return first
        }
    }
}
