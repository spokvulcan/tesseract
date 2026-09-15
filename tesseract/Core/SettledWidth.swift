//
//  SettledWidth.swift
//  tesseract
//
//  Settled Width (CONTEXT.md → App composition): the width a page lays out
//  against while the detail column is in motion. When the sidebar slides,
//  AppKit animates the column ~30 frames and SwiftUI proposes a new width to
//  the whole page on each, so every markdown row in a long transcript
//  re-measures per frame (`ResolvedStyledText.StringDrawing.sizeThatFits`,
//  main thread, no size cache in Textual). Pinning the page to its
//  pre-slide width for the duration makes each frame a no-op — SwiftUI
//  skips subtrees whose proposal did not change — and the page reflows once
//  when the column settles. Measured in the design prototype: 23 full
//  measures per open → 1.
//
//  Trigger: the column's window-space origin moving *together with* its
//  width. That is the sidebar (and the split divider); a window resize moves
//  the width alone and stays live, as the platform expects. The visibility
//  binding cannot be the trigger — SwiftUI flips it only after the AppKit
//  animation ends — but that makes it the ideal *release*, landing ~13ms
//  after the last frame. A short settle timer backs it up for divider drags,
//  which never flip the binding.
//
//  Frozen content overflows the column on open and underfills it on close,
//  anchored top-leading either way (the sidebar visibly pushes the page
//  aside); the hosting view's bounds clip the overflow. The policy is a pure
//  value so the sequencing is unit-tested without a window.
//

import SwiftUI

// MARK: - Policy

/// The Settled Width decision as a pure reducer: column geometry samples and
/// the two release signals in, the pinned width and a timer instruction out.
nonisolated struct SettledWidthPolicy: Equatable {
    /// The horizontal extent of the detail column in window space — the two
    /// numbers the sidebar slide moves together.
    struct ColumnSpan: Equatable {
        var minX: CGFloat
        var width: CGFloat
    }

    enum Event: Equatable {
        case columnSpanChanged(ColumnSpan)
        /// The split view's column-visibility binding changed value.
        case columnVisibilityChanged
        case settleTimerFired
    }

    enum Effect: Equatable {
        case none
        /// (Re)start the settle timer; it fires `.settleTimerFired` once the
        /// column has held still for the settle delay.
        case armSettleTimer
        case disarmSettleTimer
    }

    /// The width the page lays out against while the column is in motion;
    /// `nil` when the page follows the column live.
    private(set) var pinnedWidth: CGFloat?
    private var lastSpan: ColumnSpan?

    /// How long the column must hold still before a pin is released without
    /// a visibility flip (divider drags). Sidebar slides never wait this
    /// long: their binding flip releases within a frame of the last move.
    static let settleDelay: Duration = .milliseconds(120)

    mutating func apply(_ event: Event) -> Effect {
        switch event {
        case .columnSpanChanged(let span):
            defer { lastSpan = span }
            guard let previous = lastSpan else { return .none }
            let originMoved = span.minX != previous.minX
            let widthMoved = span.width != previous.width
            if pinnedWidth == nil {
                guard originMoved, widthMoved, previous.width > 0 else { return .none }
                pinnedWidth = previous.width
                return .armSettleTimer
            }
            return (originMoved || widthMoved) ? .armSettleTimer : .none

        case .columnVisibilityChanged:
            guard pinnedWidth != nil else { return .none }
            pinnedWidth = nil
            return .disarmSettleTimer

        case .settleTimerFired:
            pinnedWidth = nil
            return .none
        }
    }
}

// MARK: - Modifier

/// Lays the wrapped page out against its Settled Width. Apply once at the
/// detail root of the split view; every page inherits the behavior.
///
/// `release` is the split view's column visibility (any `Equatable` works):
/// its change is the primary pin release.
struct SettledWidthModifier<Release: Equatable>: ViewModifier {
    let release: Release

    @State private var policy = SettledWidthPolicy()
    @State private var settleTimer: Task<Void, Never>?

    func body(content: Content) -> some View {
        // The reader is sized by the column alone — a flexible frame would
        // grow to the pinned child and report the page's width, not the
        // column's. It also places the child top-leading, which is the
        // anchoring the overflow needs.
        GeometryReader { _ in
            content
                .frame(width: policy.pinnedWidth, alignment: .leading)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .onGeometryChange(for: SettledWidthPolicy.ColumnSpan.self) { proxy in
            let frame = proxy.frame(in: .global)
            return .init(minX: frame.minX, width: frame.width)
        } action: { span in
            perform(policy.apply(.columnSpanChanged(span)))
        }
        .onChange(of: release) { _, _ in
            perform(policy.apply(.columnVisibilityChanged))
        }
        .onDisappear { settleTimer?.cancel() }
    }

    private func perform(_ effect: SettledWidthPolicy.Effect) {
        switch effect {
        case .none:
            break
        case .disarmSettleTimer:
            settleTimer?.cancel()
            settleTimer = nil
        case .armSettleTimer:
            settleTimer?.cancel()
            settleTimer = Task { @MainActor in
                try? await Task.sleep(for: SettledWidthPolicy.settleDelay)
                guard !Task.isCancelled else { return }
                perform(policy.apply(.settleTimerFired))
            }
        }
    }
}

extension View {
    /// Pins the page to its **Settled Width** while the split view's detail
    /// column animates, releasing when `release` changes value or the column
    /// holds still. See `SettledWidth.swift`.
    func settledWidth(releasingOn release: some Equatable) -> some View {
        modifier(SettledWidthModifier(release: release))
    }
}
