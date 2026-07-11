//
//  OverlayVariants.swift
//  tesseract
//

import SwiftUI

/// One **Overlay Variant** (map #283): a live overlay exploration — a hosted
/// view over the shared **Overlay Feed** plus the placement of the fixed
/// panel canvas it draws in. Variants differ in everything visual; the feed
/// and the panel are common ground.
@MainActor
struct OverlayVariant: Identifiable {
    let id: String
    let displayName: String
    let placement: OverlayPlacement
    let makeView: @MainActor (DictationFeed) -> AnyView
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
    ) { feed in
        AnyView(GlobalOverlayHUD(feed: feed))
    }

    static let all: [OverlayVariant] = [classic]

    /// Unknown ids (a removed exploration surviving in defaults) fall back to
    /// the classic pill.
    static func variant(for id: String) -> OverlayVariant {
        all.first { $0.id == id } ?? classic
    }
}
