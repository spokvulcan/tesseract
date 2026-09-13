import Foundation

/// Owns the transport-lifecycle race of one streaming HTTP completion: the
/// disconnect watch, the idle keepalive prober, and the drive, run together as
/// first-finisher-wins.
///
/// `CompletionDelivery.deliver` hands the sink's transport probes plus the
/// pump to `run`; the driver
/// constructs the three-task group, returns the first outcome, and cancels the
/// losers. Pulling the race out of the handler gives it a name and a locality —
/// and makes the correctness property that matters under real clients (a client
/// abort must cancel a long prefill promptly) a table of driver tests rather
/// than an emergent behavior of a task group nobody constructs.
///
/// This sits *below* the ADR-0015 dispatcher seam: no new entry points, and the
/// handler keeps HTTP framing (SSE writing, chunk building). The driver knows
/// only the injected closures and the `CompletionDelivery.StreamingOutcome`
/// currency the handler switches on.
///
/// `nonisolated` so it composes with the handler's off-actor streaming path with
/// zero isolation change; every closure it holds is `@Sendable`.
nonisolated enum StreamLifecycleDriver {

    /// Observe disconnects while tokenization, restore and prefill are still
    /// building the generation handle, before delivery can open its response.
    static func startGeneration(
        waitForDisconnect: @escaping @Sendable () async -> Void,
        start: @escaping @Sendable () async -> Result<CompletionDelivery.Generation, Error>
    ) async -> Result<CompletionDelivery.Generation, Error> {
        await withTaskGroup(of: Result<CompletionDelivery.Generation, Error>?.self) { group in
            group.addTask {
                await waitForDisconnect()
                return nil
            }
            group.addTask { await start() }
            let first = (await group.next()).flatMap { $0 }
            group.cancelAll()

            // A start may finish just as the connection drops, or return a
            // handle despite cancellation. Drain that handle before the
            // handler can release its GPU lease; never abandon a live owner.
            var started = first
            for await remaining in group {
                if let remaining { started = remaining }
            }
            if let first, !Task.isCancelled { return first }
            if case .success(let generation) = started {
                generation.cancel()
                await generation.waitForCompletion()
            }
            return .failure(CancellationError())
        }
    }

    /// The transport probes the driver races against the drive. In production
    /// these wrap the request's `HTTPResponseWriter` and `SSEWriter`; in tests
    /// they are scripted, so the race runs without a socket.
    struct Transport: Sendable {
        /// Suspends until the client connection drops (production:
        /// `HTTPResponseWriter.waitForDisconnect`). Must be cancellation-aware
        /// so it drains when another task wins the race.
        var waitForDisconnect: @Sendable () async -> Void
        /// The keepalive channel, or `nil` for a transport that cannot carry a
        /// probe mid-response (a single JSON body): the prober is then never
        /// started and only the disconnect watch races the drive.
        var keepalive: Keepalive?
    }

    /// The two probes the idle keepalive prober needs.
    struct Keepalive: Sendable {
        /// True when the stream has been quiet for at least the given duration —
        /// the gate before a keepalive is written (production: `SSEWriter.idleFor`).
        var idleFor: @Sendable (Duration) async -> Bool
        /// Emit one keepalive probe; `false` ⇒ the write failed, i.e. the client
        /// is gone (production: `SSEWriter.keepalive`).
        var send: @Sendable () async -> Bool
    }

    /// Race the disconnect watch, the idle keepalive prober, and `drive` as
    /// first-finisher-wins, then cancel the losers.
    ///
    /// Semantics preserved verbatim from the handler's former task group:
    /// - The disconnect watch bridges the transport drop to `onTransportCancel`
    ///   (so a long prefill stops) and reports `.disconnected(.connectionState)`;
    ///   if it was cancelled first it yields `.cancelled` instead.
    /// - The keepalive prober (only when the transport has a keepalive channel)
    ///   wakes every `keepaliveCadence`, and only when the
    ///   stream is idle for the cadence it writes a keepalive; a failed write
    ///   bridges the cancel and reports `.disconnected(.keepaliveWrite)`. Its
    ///   own cancellation yields `.cancelled`; any other sleep error `.failed`.
    /// - `drive`'s outcome (completed / failed / cancelled / disconnected) passes
    ///   through unchanged.
    ///
    /// - Parameters:
    ///   - transport: the scripted/real disconnect watch plus optional keepalive.
    ///   - keepaliveCadence: the prober's wake interval and idle threshold —
    ///     today's production value by default; tests shorten it to stay fast.
    ///   - onTransportCancel: the cancel bridge into generation (production:
    ///     `CompletionDelivery.Generation.cancel`), invoked when the transport drops or a
    ///     keepalive write fails.
    ///   - drive: the generation stream pump (production:
    ///     `CompletionDelivery.pump`).
    static func run(
        transport: Transport,
        keepaliveCadence: Duration = .milliseconds(250),
        onTransportCancel: @escaping @Sendable () -> Void,
        drive: @escaping @Sendable () async -> CompletionDelivery.StreamingOutcome
    ) async -> CompletionDelivery.StreamingOutcome {
        await withTaskGroup(of: CompletionDelivery.StreamingOutcome.self) { group in
            group.addTask {
                await transport.waitForDisconnect()
                guard !Task.isCancelled else { return .cancelled }
                onTransportCancel()
                return .disconnected(.connectionState)
            }

            if let keepalive = transport.keepalive {
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

                        guard await keepalive.idleFor(keepaliveCadence) else {
                            continue
                        }

                        guard await keepalive.send() else {
                            onTransportCancel()
                            return .disconnected(.keepaliveWrite)
                        }
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
