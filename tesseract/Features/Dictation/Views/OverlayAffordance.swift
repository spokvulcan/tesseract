//
//  OverlayAffordance.swift
//  tesseract
//

import SwiftUI

extension View {
    /// The overlay's click-only control treatment: plain style (the glass is
    /// the chrome) and never focusable.
    ///
    /// The "never focusable" half is load-bearing: on macOS 27.0 a focusable
    /// control that appears in an overlay panel's hosting view after its
    /// first layout sends SwiftUI's key-view-loop rebuild into an endless
    /// loop on the main thread (the 2026-09-15 freeze on the first committed
    /// take, reproduced in `tools/overlay-focus-hang-lab`). Every Lens button
    /// goes through this modifier so a new control cannot reintroduce it; the
    /// Lens's one focusable control, its text field, exists from the first
    /// layout.
    func overlayAffordance() -> some View {
        buttonStyle(.plain).focusable(false)
    }
}
