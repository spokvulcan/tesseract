// TesseractSpeech — the demand-paced event channel behind an Utterance.
//
// Why not AsyncThrowingStream: it buffers without exposing consumer demand,
// and demand IS the pacing signal (ADR-0038: the engine suspends, lease-free,
// until pulled). This channel parks the producer at segment boundaries when
// the undelivered-segment count reaches the lookahead bound, and parks the
// consumer when the buffer is empty. Single consumer by contract.

import Foundation

actor UtteranceChannel {
    private var buffer: [SpeechEvent] = []
    private var terminal: Result<Void, Error>?
    private var consumerWaiter: CheckedContinuation<SpeechEvent?, Error>?
    private var producerWaiter: CheckedContinuation<Void, Never>?

    /// Segment-granularity demand accounting.
    private var segmentsProduced = 0
    private var segmentsDelivered = 0

    /// Invoked once if the consumer side goes away (task cancel while waiting,
    /// or utterance dropped) — wired by the engine to cancel the driver task.
    private var onConsumerGone: (@Sendable () -> Void)?
    private var consumerGoneFired = false

    func setOnConsumerGone(_ handler: @escaping @Sendable () -> Void) {
        onConsumerGone = handler
    }

    // MARK: - Producer side (the utterance driver)

    /// Never parks: chunk-level flow inside a segment is bounded by segment
    /// size; pacing happens at segment boundaries via `waitForDemand`.
    func send(_ event: SpeechEvent) {
        guard terminal == nil else { return }
        if case .segmentDone = event { segmentsProduced += 1 }
        if let waiter = consumerWaiter {
            consumerWaiter = nil
            deliverOut(event, to: waiter)
        } else {
            buffer.append(event)
        }
    }

    /// Park until undelivered completed segments < limit. Called lease-free.
    func waitForDemand(limit: Int) async {
        while terminal == nil, segmentsProduced - segmentsDelivered >= limit, !Task.isCancelled {
            await withTaskCancellationHandler {
                await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
                    if terminal == nil, segmentsProduced - segmentsDelivered >= limit, !Task.isCancelled {
                        producerWaiter = cont
                    } else {
                        cont.resume()
                    }
                }
            } onCancel: {
                Task { await self.resumeProducer() }
            }
        }
    }

    /// Terminal. `nil` = full render; an error (incl. CancellationError)
    /// propagates to the consumer untranslated.
    func finish(throwing error: Error?) {
        guard terminal == nil else { return }
        terminal = error.map(Result.failure) ?? .success(())
        if let waiter = consumerWaiter {
            consumerWaiter = nil
            if buffer.isEmpty {
                finishOut(to: waiter)
            } else {
                let event = buffer.removeFirst()
                deliverOut(event, to: waiter)
            }
        }
        resumeProducer()
    }

    private func resumeProducer() {
        if let waiter = producerWaiter {
            producerWaiter = nil
            waiter.resume()
        }
    }

    // MARK: - Consumer side (the Utterance iterator)

    func next() async throws -> SpeechEvent? {
        precondition(consumerWaiter == nil, "Utterance streams are single-consumer")
        if !buffer.isEmpty {
            let event = buffer.removeFirst()
            noteDelivered(event)
            return event
        }
        if let terminal {
            return try terminalValue(terminal)
        }
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<SpeechEvent?, Error>) in
                if !buffer.isEmpty {
                    let event = buffer.removeFirst()
                    deliverOut(event, to: cont)
                } else if let terminal {
                    finishOut(to: cont, terminal: terminal)
                } else if Task.isCancelled {
                    cont.resume(throwing: CancellationError())
                    consumerGone()
                } else {
                    consumerWaiter = cont
                }
            }
        } onCancel: {
            Task { await self.cancelWaitingConsumer() }
        }
    }

    /// The utterance value was dropped without being drained.
    func consumerDropped() {
        consumerGone()
    }

    private func cancelWaitingConsumer() {
        if let waiter = consumerWaiter {
            consumerWaiter = nil
            waiter.resume(throwing: CancellationError())
        }
        consumerGone()
    }

    private func consumerGone() {
        guard !consumerGoneFired else { return }
        consumerGoneFired = true
        onConsumerGone?()
        resumeProducer()
    }

    // MARK: - Helpers

    private func deliverOut(_ event: SpeechEvent, to cont: CheckedContinuation<SpeechEvent?, Error>) {
        noteDelivered(event)
        cont.resume(returning: event)
    }

    private func finishOut(to cont: CheckedContinuation<SpeechEvent?, Error>, terminal: Result<Void, Error>? = nil) {
        let result = terminal ?? self.terminal ?? .success(())
        switch result {
        case .success: cont.resume(returning: nil)
        case .failure(let error): cont.resume(throwing: error)
        }
    }

    private func terminalValue(_ result: Result<Void, Error>) throws -> SpeechEvent? {
        switch result {
        case .success: return nil
        case .failure(let error): throw error
        }
    }

    private func noteDelivered(_ event: SpeechEvent) {
        if case .segmentDone = event {
            segmentsDelivered += 1
            resumeProducer()
        }
    }

    /// The driver finished or was cancelled and the channel is fully drained
    /// or abandoned — used by tests and the engine's supersession await.
    var isFinished: Bool { terminal != nil }
}

/// Keeps drop-stops-generation honest: the last copy of an Utterance (or its
/// iterator) dying fires `onDrop`, which the engine wires to driver cancel.
final class DropToken: @unchecked Sendable {
    private let onDrop: @Sendable () -> Void
    init(onDrop: @escaping @Sendable () -> Void) { self.onDrop = onDrop }
    deinit { onDrop() }
}

// MARK: - Utterance

/// One admitted text→speech run (ADR-0038): admission-time facts plus the one
/// event sequence that is the audio transport, the timing channel, the pacing
/// signal, and the cancellation token.
public struct Utterance: Sendable {
    public let sampleRate: Int
    public let framesPerSecond: Double
    public let segmentCount: Int

    let channel: UtteranceChannel
    let dropToken: DropToken

    /// The event sequence. Single-consumer; consume `events` OR `audio`.
    public var events: Events { Events(channel: channel, token: dropToken) }

    /// Projection for callers that don't highlight: just the audio chunks.
    public var audio: AudioEvents { AudioEvents(base: events) }

    public struct Events: AsyncSequence, Sendable {
        public typealias Element = SpeechEvent
        let channel: UtteranceChannel
        let token: DropToken

        public func makeAsyncIterator() -> Iterator {
            Iterator(channel: channel, token: token)
        }

        public struct Iterator: AsyncIteratorProtocol {
            let channel: UtteranceChannel
            let token: DropToken

            public mutating func next() async throws -> SpeechEvent? {
                try await channel.next()
            }
        }
    }

    public struct AudioEvents: AsyncSequence, Sendable {
        public typealias Element = AudioChunk
        let base: Events

        public func makeAsyncIterator() -> Iterator {
            Iterator(base: base.makeAsyncIterator())
        }

        public struct Iterator: AsyncIteratorProtocol {
            var base: Events.Iterator

            public mutating func next() async throws -> AudioChunk? {
                while let event = try await base.next() {
                    if case .audio(let chunk) = event { return chunk }
                }
                return nil
            }
        }
    }
}
