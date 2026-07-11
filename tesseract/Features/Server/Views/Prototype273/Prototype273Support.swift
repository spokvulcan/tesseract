//
//  Prototype273Support.swift
//  tesseract
//
//  PROTOTYPE (wayfinder #273) — THROWAWAY. Layout constants of the Activity
//  prototype; goes away when the Activity cutover (#276) replaces the
//  Dashboard. The chart palette/hover/tooltip pieces that used to live here
//  were promoted to production in the Cache cutover (#277) —
//  `Views/ChartSupport.swift`.
//

import SwiftUI

enum Proto273Layout {
    /// Below this content width the Activity page stacks single-column
    /// (status → transcript → recent) — the ~400 pt bar with margin.
    static let activityWideBreakpoint: CGFloat = 700
    /// The recent rail's fixed width on wide layouts.
    static let railWidth: CGFloat = 264
    /// The rail's height when stacked below the transcript at narrow widths.
    static let stackedRailHeight: CGFloat = 188
}
