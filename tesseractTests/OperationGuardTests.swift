import Testing

@testable import Tesseract_Agent

/// Tests for the **Operation Guard** — the monotonic-epoch *stale-result* protocol
/// shared by the capture→transcribe→commit coordinators. Driven directly through
/// the `invalidate` / `capture` / `isCurrent` interface, with zero dependencies:
/// no audio, no `Transcribing`, no `Task`.
@MainActor
struct OperationGuardTests {

    @Test
    func freshTicketIsCurrent() {
        let guardian = OperationGuard()

        let ticket = guardian.capture()

        #expect(ticket.isCurrent)
    }

    /// The load-bearing regression: an `invalidate()` after the ticket was captured —
    /// the cancel-or-restart that, without this protocol, lets a late transcription
    /// inject stale text — must make the ticket report stale. One line instead of a
    /// full capture→transcribe→cancel→restart integration dance.
    @Test
    func invalidateAfterCaptureMakesTicketStale() {
        let guardian = OperationGuard()
        let ticket = guardian.capture()

        guardian.invalidate()

        #expect(!ticket.isCurrent)
    }

    /// Two tickets captured across an `invalidate()` behave independently and
    /// monotonically: the earlier one is stale, the later one current. This is the
    /// overlapping-operation case — a new operation supersedes the prior one without
    /// the prior's still-running work being able to commit.
    @Test
    func laterTicketSupersedesEarlierAcrossInvalidate() {
        let guardian = OperationGuard()
        let earlier = guardian.capture()

        guardian.invalidate()
        let later = guardian.capture()

        #expect(!earlier.isCurrent)
        #expect(later.isCurrent)
    }

    /// `capture()` is a pure read: it snapshots the epoch without advancing it, so two
    /// tickets taken with no `invalidate()` between them are both current. The dual of the
    /// staleness tests — it pins the invariant the docs warn against losing: a naive
    /// `begin()` that bumped-and-snapshotted would make this fail.
    @Test
    func captureDoesNotAdvanceEpoch() {
        let guardian = OperationGuard()

        let first = guardian.capture()
        let second = guardian.capture()

        #expect(first.isCurrent)
        #expect(second.isCurrent)
    }
}
