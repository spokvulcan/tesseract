//
//  GlowTheme.swift
//  whisper-on-device
//

import SwiftUI

/// Theme options for the full-screen glow effect
enum GlowTheme: String, CaseIterable, Identifiable {
    case appleIntelligence
    case matrix
    case ocean
    case fire
    case aurora
    case monochrome

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .appleIntelligence: "Apple Intelligence"
        case .matrix: "Matrix"
        case .ocean: "Ocean"
        case .fire: "Fire"
        case .aurora: "Aurora"
        case .monochrome: "Monochrome"
        }
    }

    var colors: [Color] {
        switch self {
        case .appleIntelligence:
            return [
                Color(hex: "BC82F3"),  // Purple
                Color(hex: "F5B9EA"),  // Pink
                Color(hex: "8D9FFF"),  // Blue
                Color(hex: "FF6778"),  // Coral
                Color(hex: "FFBA71"),  // Orange
                Color(hex: "C686FF"),  // Violet
            ]

        case .matrix:
            return [
                Color(hex: "00FF00"),  // Bright green
                Color(hex: "00DD00"),  // Green
                Color(hex: "00AA00"),  // Medium green
                Color(hex: "33FF33"),  // Light green
                Color(hex: "00FF66"),  // Cyan-green
                Color(hex: "66FF00"),  // Yellow-green
            ]

        case .ocean:
            return [
                Color(hex: "00CED1"),  // Dark turquoise
                Color(hex: "20B2AA"),  // Light sea green
                Color(hex: "5F9EA0"),  // Cadet blue
                Color(hex: "4169E1"),  // Royal blue
                Color(hex: "00BFFF"),  // Deep sky blue
                Color(hex: "7FFFD4"),  // Aquamarine
            ]

        case .fire:
            return [
                Color(hex: "FF4500"),  // Orange red
                Color(hex: "FF6600"),  // Orange
                Color(hex: "FF8C00"),  // Dark orange
                Color(hex: "FFD700"),  // Gold
                Color(hex: "FF0000"),  // Red
                Color(hex: "FF3300"),  // Red-orange
            ]

        case .aurora:
            return [
                Color(hex: "00FF87"),  // Spring green
                Color(hex: "60EFFF"),  // Cyan
                Color(hex: "B967FF"),  // Purple
                Color(hex: "01CDFE"),  // Sky blue
                Color(hex: "05FFA1"),  // Mint
                Color(hex: "FF71CE"),  // Pink
            ]

        case .monochrome:
            return [
                Color(hex: "FFFFFF"),  // White
                Color(hex: "E0E0E0"),  // Light gray
                Color(hex: "C0C0C0"),  // Silver
                Color(hex: "F5F5F5"),  // White smoke
                Color(hex: "DCDCDC"),  // Gainsboro
                Color(hex: "D3D3D3"),  // Light gray
            ]
        }
    }
}
