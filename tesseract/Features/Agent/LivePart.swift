import Foundation
import Observation

/// The **Live Part** (ADR-0024): the single observable box that exists while
/// one Content Part streams. A token delta appends to `raw` (deliberately not
/// observable) and republishes `displayText` at most once per `throttle`, so
/// exactly one `Text` view invalidates — at a bounded rate — no matter how
/// fast tokens arrive. On part end the Chat Session commits the part into the
/// value rows and drops this box.
///
/// Tool calls never stream through a Live Part — they arrive fully parsed and
/// commit atomically — so the kind is text or thinking only.
@Observable @MainActor
final class LivePart: Identifiable {

    enum Kind: Equatable, Sendable {
        case text
        case thinking
    }

    /// Stable identity: the streaming message's id + this part's index.
    nonisolated let messageID: UUID
    nonisolated let partIndex: Int
    nonisolated let kind: Kind

    /// What the view renders — `raw`, republished at most once per `throttle`.
    private(set) var displayText: String

    /// The full accumulated content. Not observable by design: appending a
    /// delta must not invalidate the view; only `displayText` publishes.
    @ObservationIgnored private(set) var raw: String

    @ObservationIgnored private let throttle: Duration
    @ObservationIgnored private var lastPublish: ContinuousClock.Instant
    @ObservationIgnored private var trailingFlush: Task<Void, Never>?

    init(messageID: UUID, partIndex: Int, kind: Kind, initial: String = "", throttle: Duration) {
        self.messageID = messageID
        self.partIndex = partIndex
        self.kind = kind
        self.raw = initial
        self.displayText = initial
        self.throttle = throttle
        self.lastPublish = .now
    }

    /// Append a streamed delta. Publishes immediately when the throttle window
    /// has elapsed; otherwise schedules one trailing flush so the final tokens
    /// of a burst never sit unpublished.
    func append(_ delta: String) {
        raw += delta
        let now = ContinuousClock.now
        if now - lastPublish >= throttle {
            publish(at: now)
        } else if trailingFlush == nil {
            let deadline = lastPublish + throttle
            trailingFlush = Task { [weak self] in
                try? await Task.sleep(until: deadline, clock: .continuous)
                guard let self, !Task.isCancelled else { return }
                self.publish(at: .now)
            }
        }
    }

    /// Publish `raw` immediately (part end, resync).
    func flush() {
        publish(at: .now)
    }

    private func publish(at instant: ContinuousClock.Instant) {
        trailingFlush?.cancel()
        trailingFlush = nil
        lastPublish = instant
        if displayText != raw { displayText = raw }
    }
}
