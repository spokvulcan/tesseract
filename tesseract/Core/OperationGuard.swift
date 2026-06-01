//
//  OperationGuard.swift
//  tesseract
//
//  The **Operation Guard** — the single home for the monotonic-epoch *stale-result*
//  protocol shared by the capture→transcribe→commit coordinators (today
//  `DictationCoordinator` and **Voice Input** / `AgentVoiceInputController`).
//
//  It exists because the underlying recognizer may *ignore `Task` cancellation and
//  return success anyway*, so cancelling the task is not enough to stop a stale
//  transcription from committing — only a post-`await` epoch comparison can. Callers:
//
//    - call `invalidate()` to advance the epoch — at **two** distinct moments: at
//      `cancel()` (drop the in-flight operation) **and** at *operation start*
//      (`startRecording`/`start`, to supersede an *overlapping previous* operation,
//      since neither caller cancels the prior task on restart);
//    - call `capture()` once at the entry of async work to snapshot the epoch into an
//      `OperationTicket`;
//    - read `ticket.isCurrent` after **every** `await` resume to decide whether the
//      still-running work may commit.
//
//  It owns **only** the epoch protocol. `Task` cancellation, `audioCapture.stopCapture()`,
//  and `transcriptionEngine.cancelTranscription()` stay caller-side — they are the
//  complementary, cancellation-*aware* mechanism (see CONTEXT.md → Operation staleness).
//
//  Distinct from Swift's `guard` statement (which is, confusingly, how `isCurrent` is
//  read at the call sites) and from `Task` cancellation. No `Sendable`: the guard is a
//  stored property of an `@MainActor` coordinator and every ticket is captured inside a
//  coordinator-owned non-detached `Task`, so the guard outlives every ticket it vends.
//

/// Owns one monotonic `epoch` counter and vends inert `OperationTicket` snapshots of it.
/// See the file header and CONTEXT.md → "Operation staleness".
@MainActor
final class OperationGuard {
    /// `fileprivate` getter so the co-located `OperationTicket` can compare against the
    /// live epoch; `private` setter so only `invalidate()` advances it.
    fileprivate private(set) var epoch = 0

    /// Advance the epoch, marking every previously-captured ticket stale. Called at a
    /// `cancel()` and at *operation start*; the two *meanings* live at the call sites.
    func invalidate() {
        epoch += 1
    }

    /// Snapshot the current epoch at the entry of async work. Does **not** advance it.
    func capture() -> OperationTicket {
        OperationTicket(owner: self, epoch: epoch)
    }
}

/// The inert value vended by `OperationGuard.capture()`: an `unowned` reference to its
/// owning guard plus the snapshot epoch. Its sole interface is `isCurrent`, read on the
/// MainActor after each `await` resume. Carries no behaviour beyond the comparison and is
/// never persisted past the operation that captured it.
struct OperationTicket {
    unowned let owner: OperationGuard
    let epoch: Int

    /// `true` until the owning guard is `invalidate()`d — i.e. while this ticket's
    /// operation is still the current one.
    var isCurrent: Bool { owner.epoch == epoch }
}
