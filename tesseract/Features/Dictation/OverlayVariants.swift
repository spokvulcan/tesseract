//
//  OverlayVariants.swift
//  tesseract
//

import SwiftUI

/// The variant-agnostic overlay action surface (ticket #289) — the callbacks
/// a variant's lingering-beat affordances invoke. The mirror of the Overlay
/// Feed: the feed carries signals *to* every variant, this carries the few
/// sanctioned clicks *back*, so variants never see the coordinator.
@MainActor
struct OverlayActions {
    /// One-click "that was wrong" on the last take — marks its Correction
    /// Pair gold. No window, no focus steal.
    let flagLastTakeWrong: @MainActor () -> Void
    /// Opens the dictation history at the last take's entry (full editing
    /// lives there — the overlay stays keyboard-free).
    let editLastTake: @MainActor () -> Void
    /// Injects the raw text of a rejected take anyway (and flags its pair —
    /// using it *is* "the pass was wrong").
    let insertRawAnyway: @MainActor () -> Void

    /// Inert actions for previews and tests.
    static let none = OverlayActions(
        flagLastTakeWrong: {}, editLastTake: {}, insertRawAnyway: {})
}

/// One **Overlay Variant** (map #283): a live overlay exploration — a hosted
/// view over the shared **Overlay Feed** plus the placement of the fixed
/// panel canvas it draws in. Variants differ in everything visual; the feed
/// and the panel are common ground.
@MainActor
struct OverlayVariant: Identifiable {
    let id: String
    let displayName: String
    let placement: OverlayPlacement
    let makeView: @MainActor (DictationFeed, OverlayActions) -> AnyView
}

/// The variant registry the overlay-variant Setting selects from. Exploration
/// scaffolding: the registry and the Setting are deleted when the redesign
/// program prunes to one winner.
@MainActor
enum OverlayVariants {
    static let classic = OverlayVariant(
        id: "classic",
        displayName: "Classic Pill",
        placement: .pill
    ) { feed, actions in
        AnyView(GlobalOverlayHUD(feed: feed, actions: actions))
    }

    static let all: [OverlayVariant] = [classic]

    /// Unknown ids (a removed exploration surviving in defaults) fall back to
    /// the classic pill.
    static func variant(for id: String) -> OverlayVariant {
        all.first { $0.id == id } ?? classic
    }
}
