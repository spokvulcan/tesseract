//
//  MemoryWindowSupport.swift
//  tesseract
//
//  Shared presentation vocabulary for the Memory window (ADR-0035 §9):
//  layout constants, and how the domain's enums read on screen.
//
//  Provenance is the one that matters. STATED versus INFERRED is a safety
//  field, not a nicety (ADR-0035 §2) — the window renders it visually
//  unmissable everywhere a memory appears: a colored badge on every row, a
//  colored banner on every detail.
//

import SwiftUI

/// One spacing rhythm, one readable column (design language §2). The window
/// keeps to the system body size throughout — hierarchy comes from weight
/// and color, never from size.
enum MemoryWindowLayout {
    /// The single vertical rhythm of the surface.
    static let rhythm: CGFloat = 12
    /// Readable cap on prose-like content in the detail and journal columns.
    static let columnMaxWidth: CGFloat = 640
}

// MARK: - Provenance

extension Provenance {
    var label: String {
        switch self {
        case .stated: "Stated"
        case .inferred: "Inferred"
        }
    }

    /// Semantic, not decorative: the owner's own words and the assistant's
    /// conclusions must be tell-apart-able at a glance.
    var color: Color {
        switch self {
        case .stated: .blue
        case .inferred: .purple
        }
    }

    var symbol: String {
        switch self {
        case .stated: "quote.opening"
        case .inferred: "sparkles"
        }
    }

    /// The detail banner, in the assistant's own voice.
    var bannerText: String {
        switch self {
        case .stated: "You told me this."
        case .inferred: "I concluded this myself — you never said it outright."
        }
    }
}

/// The one badge shape of the window: a tinted capsule whose color *is* the
/// meaning. Provenance and status both wear it, so they can never drift apart
/// visually.
struct TintedCapsule: View {
    let label: String
    let color: Color

    var body: some View {
        Text(label)
            .fontWeight(.medium)
            .foregroundStyle(color)
            .padding(.horizontal, 8)
            .padding(.vertical, 1)
            .background(color.opacity(0.14), in: Capsule())
    }
}

/// The row-level provenance marker: a tinted capsule leading every memory
/// row, so no belief is ever read without its origin.
struct ProvenanceBadge: View {
    let provenance: Provenance

    var body: some View {
        TintedCapsule(label: provenance.label, color: provenance.color)
    }
}

// MARK: - Status

extension MemoryStatus {
    /// nil for `.live` — a live memory carries no status chrome.
    var badgeLabel: String? {
        switch self {
        case .live: nil
        case .contested: "Contested"
        case .superseded: "Superseded"
        }
    }

    var color: Color {
        switch self {
        case .live: .primary
        case .contested: .red
        case .superseded: .secondary
        }
    }
}

struct MemoryStatusBadge: View {
    let status: MemoryStatus

    var body: some View {
        if let label = status.badgeLabel {
            TintedCapsule(label: label, color: status.color)
        }
    }
}

// MARK: - Tier

extension MemoryTier {
    var label: String {
        switch self {
        case .core: "Core"
        case .hot: "Hot"
        case .warm: "Warm"
        case .cold: "Cold"
        }
    }

    /// What the tier concretely means for retrieval — shown in the detail
    /// facts so the lifecycle is legible, not mystical.
    var meaning: String {
        switch self {
        case .core: "part of every conversation"
        case .hot: "in the default retrieval pool"
        case .warm: "retrieved when strongly relevant"
        case .cold: "retired from default retrieval — never deleted"
        }
    }
}

// MARK: - Kind & source

extension MemoryKind {
    var label: String {
        switch self {
        case .belief: "Belief"
        case .event: "Event"
        case .pattern: "Pattern"
        case .directive: "Directive"
        }
    }
}

extension MemorySource {
    var label: String {
        switch self {
        case .chat: "Chat"
        case .companion: "Companion"
        case .dictation: "Dictation"
        case .backfill: "Backfill"
        }
    }

    var symbol: String {
        switch self {
        case .chat: "bubble.left"
        case .companion: "heart.text.square"
        case .dictation: "mic"
        case .backfill: "clock.arrow.circlepath"
        }
    }
}

// MARK: - Journal mutations

extension MemoryMutation {
    var label: String {
        switch self {
        case .added: "Added"
        case .confirmed: "Confirmed"
        case .contested: "Contested"
        case .superseded: "Superseded"
        case .promoted: "Promoted"
        case .demoted: "Demoted"
        case .graded: "Useful"
        case .deletedByOwner: "Deleted by you"
        }
    }

    var symbol: String {
        switch self {
        case .added: "plus.circle"
        case .confirmed: "checkmark.circle"
        case .contested: "exclamationmark.triangle"
        case .superseded: "arrow.right.circle"
        case .promoted: "arrow.up.circle"
        case .demoted: "arrow.down.circle"
        case .graded: "checkmark.seal"
        case .deletedByOwner: "trash"
        }
    }

    /// Semantic color per rule 4: growth green, dispute red, movement blue,
    /// the quiet rest secondary.
    var color: Color {
        switch self {
        case .added, .confirmed, .graded: .green
        case .contested, .deletedByOwner: .red
        case .promoted: .blue
        case .superseded, .demoted: .secondary
        }
    }
}

// MARK: - Dates

extension Date {
    /// Day-resolution, for born/last-used facts.
    var memoryDay: String {
        formatted(date: .abbreviated, time: .omitted)
    }

    /// Minute-resolution, for episodes and journal entries.
    var memoryMoment: String {
        formatted(date: .abbreviated, time: .shortened)
    }
}
