//
//  ImageInputAvailability.swift
//  tesseract
//

import Foundation

/// The **image-input-availability projection** (ADR-0013, PRD #112): a pure
/// decision over *(is the selected model vision-capable?, is the global "Use
/// vision models when available" setting on?)* → whether the composer should
/// surface image affordances (the `+` picker, paste, drop).
///
/// Kept `nonisolated` and free of any model-loading or capability-probing
/// dependency precisely so it is a plain boolean function: the view supplies the
/// two inputs (capability via `ModelDownloadManager.isVisionCapable(_:)`, the flag
/// via `SettingsManager`), and this returns the verdict. The probing/loading lives
/// at the impure edges; only this verdict is unit-tested over the full matrix.
nonisolated enum ImageInputAvailability {

    /// Show image affordances only when the selected model can actually serve
    /// images *and* the user hasn't opted out of vision. Either input false →
    /// the chat will run the text-only container, so attaching would silently
    /// drop — hide the affordance instead.
    static func showImageAffordance(
        isVisionCapable: Bool,
        useVisionWhenAvailable: Bool
    ) -> Bool {
        isVisionCapable && useVisionWhenAvailable
    }
}
