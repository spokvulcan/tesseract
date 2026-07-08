//
//  SkillClusterController.swift
//  tesseract
//
//  The **Skill Cluster** interaction state machine (ADR-0030): collapsed ⇄
//  expanded ⇄ pinned. Hover opens after a short delay, leaving collapses after
//  a grace window, a click pins until click-away/Esc, and firing a pill
//  collapses. A publisher-agnostic leaf in the Agent-coordinator-leaves
//  pattern — the SwiftUI cluster view is a dumb rendering of `phase`; skill
//  execution stays on the coordinator spine.
//

import Foundation
import Observation

@Observable @MainActor
final class SkillClusterController {

    /// The cluster's three interaction states. `expanded` is hover-held and
    /// collapses when the pointer leaves; `pinned` survives pointer exit.
    enum Phase: Equatable, Sendable {
        case collapsed
        case expanded
        case pinned
    }

    // MARK: - Observable State

    private(set) var phase: Phase = .collapsed

    /// Whether the fanned pills are showing (either hover-held or pinned).
    var isOpen: Bool { phase != .collapsed }

    /// External gate — true while a run is generating or the slash popup is
    /// open. A suppressed cluster ignores hover and clicks; becoming
    /// suppressed collapses it and cancels any pending transition.
    var isSuppressed: Bool = false {
        didSet {
            guard isSuppressed != oldValue else { return }
            if isSuppressed {
                cancelPendingTransition()
                phase = .collapsed
            } else {
                // A draft written mid-suppression still deserves the
                // suggestion once the run/popup is out of the way.
                autoOpenIfArmed()
            }
        }
    }

    // MARK: - Draft auto-open

    /// Whether the composer draft currently has content (text or images) —
    /// mirrored in by the view.
    @ObservationIgnored private var draftHasContent = false

    /// Armed on the draft's empty→non-empty edge; disarmed by any manual
    /// close (bubble toggle, Esc, click-away, fire) or by the draft emptying.
    /// While armed, an auto-open fires as soon as the cluster is collapsed
    /// and unsuppressed — so a close is always the user's last word.
    @ObservationIgnored private var autoOpenArmed = false

    /// The composer draft gained or lost content. Gaining content opens the
    /// cluster pinned — suggesting the skills for what was just typed or
    /// attached — and losing it (send, fire, clear) retires the cluster.
    func draftContentChanged(hasContent: Bool) {
        guard hasContent != draftHasContent else { return }
        draftHasContent = hasContent
        if hasContent {
            autoOpenArmed = true
            autoOpenIfArmed()
        } else {
            autoOpenArmed = false
            cancelPendingTransition()
            phase = .collapsed
        }
    }

    private func autoOpenIfArmed() {
        guard autoOpenArmed, !isSuppressed, phase == .collapsed else { return }
        cancelPendingTransition()
        phase = .pinned
    }

    // MARK: - Timing

    /// Hover-open delay — long enough that a pointer crossing the bubble on
    /// its way elsewhere never flashes the cluster open.
    private let openDelay: Duration
    /// Exit grace — long enough that a small pointer slip off the cluster's
    /// union doesn't collapse it mid-reach. (A 2s grace was tried and
    /// reverted — it made the cluster feel stuck open.)
    private let exitGrace: Duration

    /// The one in-flight delayed transition (pending open or pending
    /// collapse). Any newer event cancels and replaces it.
    @ObservationIgnored private var pendingTransition: Task<Void, Never>?

    // MARK: - Init

    init(
        openDelay: Duration = .milliseconds(150),
        exitGrace: Duration = .milliseconds(250)
    ) {
        self.openDelay = openDelay
        self.exitGrace = exitGrace
    }

    // MARK: - Pointer events

    /// Pointer entered the cluster's union (bubble or fanned pills). Opens
    /// after `openDelay` when collapsed; rescinds a pending collapse when
    /// already open.
    func pointerEntered() {
        guard !isSuppressed else { return }
        cancelPendingTransition()
        guard phase == .collapsed else { return }
        pendingTransition = Task { [weak self] in
            try? await Task.sleep(for: self?.openDelay ?? .zero)
            guard !Task.isCancelled, let self, self.phase == .collapsed, !self.isSuppressed
            else { return }
            self.phase = .expanded
        }
    }

    /// Pointer left the cluster's union. Cancels a pending open; collapses a
    /// hover-held cluster after `exitGrace`. A pinned cluster ignores it.
    func pointerExited() {
        cancelPendingTransition()
        guard phase == .expanded else { return }
        pendingTransition = Task { [weak self] in
            try? await Task.sleep(for: self?.exitGrace ?? .zero)
            guard !Task.isCancelled, let self, self.phase == .expanded else { return }
            self.phase = .collapsed
        }
    }

    // MARK: - Click events

    /// The ✦ bubble was clicked: pin the cluster open (from collapsed or
    /// hover-held), or collapse it when already pinned. Synchronous — a
    /// deliberate click never waits out the hover delay.
    func buttonClicked() {
        guard !isSuppressed else { return }
        cancelPendingTransition()
        if phase == .pinned {
            autoOpenArmed = false
            phase = .collapsed
        } else {
            phase = .pinned
        }
    }

    /// A click landed outside the cluster while pinned — collapse.
    func clickedAway() {
        cancelPendingTransition()
        autoOpenArmed = false
        phase = .collapsed
    }

    /// Esc pressed. Collapses an open cluster and reports whether the event
    /// was consumed, so callers can chain to the next Esc responder.
    @discardableResult
    func escapePressed() -> Bool {
        guard isOpen else { return false }
        cancelPendingTransition()
        autoOpenArmed = false
        phase = .collapsed
        return true
    }

    /// A pill fired — the invocation is on its way, collapse immediately.
    func pillFired() {
        cancelPendingTransition()
        autoOpenArmed = false
        phase = .collapsed
    }

    // MARK: - Helpers

    private func cancelPendingTransition() {
        pendingTransition?.cancel()
        pendingTransition = nil
    }

    /// Await the in-flight delayed transition, if any — lets tests settle the
    /// state machine deterministically with zero durations.
    func settle() async {
        await pendingTransition?.value
    }
}
