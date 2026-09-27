//
//  PrototypeSwitcher.swift
//  tesseract
//
//  PROTOTYPE — THROWAWAY. The floating variant switcher for in-app UI
//  prototypes (the /prototype skill's bottom bar): ‹ label ›, and the ← →
//  keys cycle whenever no text field has focus. It only ever shows in a
//  local development build (one run from Xcode's DerivedData), never in an
//  app installed from a release. Never merge.
//

import AppKit
import SwiftUI

enum PrototypeGate {
    /// A build launched from DerivedData is a local development build; a
    /// shipped app runs from /Applications. Prototypes stay invisible there.
    static let isDevelopmentBuild: Bool = Bundle.main.bundlePath.contains("/DerivedData/")
}

/// A high-contrast pill that is obviously not part of the design under test.
struct PrototypeSwitcher: View {
    let labels: [String]
    @Binding var index: Int
    @State private var sizeIndex = -1

    var body: some View {
        HStack(spacing: 2) {
            arrow("chevron.left", step: -1)
            VStack(spacing: 0) {
                Text(labels[clamped])
                    .font(.system(size: 12, weight: .semibold))
                Text("\(clamped + 1) of \(labels.count) · ← →")
                    .font(.system(size: 9, weight: .medium))
                    .foregroundStyle(.white.opacity(0.55))
            }
            .frame(minWidth: 150)
            arrow("chevron.right", step: 1)
            Button {
                sizeIndex = (sizeIndex + 1) % Self.windowSizes.count
                resizeWindow(to: Self.windowSizes[sizeIndex].size)
            } label: {
                Label("Resize window", systemImage: "arrow.up.left.and.arrow.down.right")
                    .labelStyle(.iconOnly)
                    .font(.system(size: 10, weight: .bold))
                    .frame(width: 24, height: 28)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help(
                "Cycle the window through common sizes: \(Self.windowSizes.map(\.label).joined(separator: ", "))"
            )
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 4)
        .padding(.vertical, 4)
        .background(Capsule().fill(Color.black.opacity(0.88)))
        .overlay(Capsule().strokeBorder(Color.yellow.opacity(0.85), lineWidth: 1))
        .shadow(color: .black.opacity(0.35), radius: 10, y: 3)
        .background(ArrowKeyMonitor(onStep: step))
        .environment(\.colorScheme, .dark)
        .help("Prototype switcher — not part of the design")
    }

    private var clamped: Int { min(max(index, 0), labels.count - 1) }

    private static let windowSizes: [(label: String, size: CGSize)] = [
        ("Large · 1280 × 860", CGSize(width: 1280, height: 860)),
        ("Typical · 1100 × 830", CGSize(width: 1100, height: 830)),
        ("Compact · 860 × 700", CGSize(width: 860, height: 700)),
        ("Narrow · 640 × 660", CGSize(width: 640, height: 660)),
    ]

    /// Keeps the window's top-left corner and fits it on its screen.
    private func resizeWindow(to size: CGSize) {
        guard
            let window = NSApp.keyWindow ?? NSApp.mainWindow
                ?? NSApp.windows.first(where: {
                    $0.isVisible && !($0 is NSPanel)
                })
        else { return }
        let visible = window.screen?.visibleFrame ?? NSScreen.main?.visibleFrame ?? .zero
        let width = min(size.width, visible.width)
        let height = min(size.height, visible.height)
        var origin = NSPoint(x: window.frame.minX, y: window.frame.maxY - height)
        origin.x = min(max(origin.x, visible.minX), visible.maxX - width)
        origin.y = min(max(origin.y, visible.minY), visible.maxY - height)
        // Not animated: an animated programmatic resize leaves NSSplitView's
        // side columns overflowing the window instead of shrinking the detail.
        window.setFrame(
            NSRect(origin: origin, size: CGSize(width: width, height: height)), display: true,
            animate: false)
        window.contentView?.layoutSubtreeIfNeeded()
    }

    private func arrow(_ symbol: String, step delta: Int) -> some View {
        Button {
            step(delta)
        } label: {
            Label(delta < 0 ? "Previous variant" : "Next variant", systemImage: symbol)
                .labelStyle(.iconOnly)
                .font(.system(size: 12, weight: .bold))
                .frame(width: 28, height: 28)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private func step(_ delta: Int) {
        guard !labels.isEmpty else { return }
        index = (clamped + delta + labels.count) % labels.count
    }
}

/// Cycles on bare ← / → key presses in this window, unless a text view or
/// field editor holds focus (the arrows belong to the caret then).
private struct ArrowKeyMonitor: NSViewRepresentable {
    let onStep: (Int) -> Void

    func makeNSView(context: Context) -> MonitorView {
        let view = MonitorView()
        view.onStep = onStep
        return view
    }

    func updateNSView(_ nsView: MonitorView, context: Context) {
        nsView.onStep = onStep
    }

    final class MonitorView: NSView {
        var onStep: ((Int) -> Void)?
        private var monitor: Any?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            if let monitor { NSEvent.removeMonitor(monitor) }
            monitor = nil
            guard window != nil else { return }
            monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
                guard let self, let window = self.window, event.window === window else {
                    return event
                }
                let modifiers = event.modifierFlags.intersection([
                    .command, .option, .control, .shift,
                ])
                guard modifiers.isEmpty else { return event }
                if window.firstResponder is NSText { return event }
                switch event.keyCode {
                case 123:
                    self.onStep?(-1)
                    return nil
                case 124:
                    self.onStep?(1)
                    return nil
                default:
                    return event
                }
            }
        }

        // Leaving the window (window == nil above) removes the monitor.
        override func hitTest(_ point: NSPoint) -> NSView? { nil }
    }
}
