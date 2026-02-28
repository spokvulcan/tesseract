//
//  DesignTokens.swift
//  tesseract
//

import SwiftUI

/// Centralized design tokens for consistent UI across the app.
enum Theme {
    enum Radius {
        static let small: CGFloat = 8
        static let medium: CGFloat = 12
        static let large: CGFloat = 16
        static let pill: CGFloat = 20
    }

    enum Spacing {
        static let xs: CGFloat = 4
        static let sm: CGFloat = 8
        static let md: CGFloat = 12
        static let lg: CGFloat = 16
        static let xl: CGFloat = 20
        static let xxl: CGFloat = 24
    }

    enum Layout {
        static let contentMaxWidth: CGFloat = 820
        static let sidebarMinWidth: CGFloat = 180
        static let sidebarWidth: CGFloat = 260
    }
}
