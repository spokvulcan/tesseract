//
//  FixInLensAction.swift
//  tesseract
//
//  Opens a take from the Dictation page in the **Lens** (PRD #612) to fix a
//  word in it. The app injects it with the dictation dependencies; the
//  default does nothing, so a preview or a test needs no Lens.
//

import SwiftUI

nonisolated struct FixInLensAction: Sendable {
    private let open: @MainActor @Sendable (DictatedTake) -> Bool

    init(_ open: @escaping @MainActor @Sendable (DictatedTake) -> Bool) {
        self.open = open
    }

    /// Opens `take` for fixing; false while a take is being recorded or a
    /// fix is being put back in an app.
    @MainActor
    @discardableResult
    func callAsFunction(_ take: DictatedTake) -> Bool {
        open(take)
    }
}

extension EnvironmentValues {
    @Entry var fixInLens = FixInLensAction { _ in false }
}
