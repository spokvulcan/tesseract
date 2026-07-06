//
//  ChatMarkdownStyle.swift
//  tesseract
//
//  The Prose Accent Palette: assistant markdown syntax rendered in color so
//  the document structure reads at a glance — the one sanctioned exception to
//  the monochrome content layer. Roles and values are adapted from OpenCode's
//  default theme (packages/tui/src/theme/assets/opencode.json), dark and
//  light variants alike; block layout stays GitHub's.
//

import SwiftUI
import Textual

/// A color from a 24-bit RGB hex value, e.g. `0x9D7CD8`.
private func rgb(_ hex: UInt32) -> Color {
    Color(
        red: Double((hex >> 16) & 0xFF) / 255,
        green: Double((hex >> 8) & 0xFF) / 255,
        blue: Double(hex & 0xFF) / 255
    )
}

extension DynamicColor {
    /// Headings — OpenCode `markdownHeading`.
    static let proseHeading = DynamicColor(light: rgb(0xD68C27), dark: rgb(0x9D7CD8))
    /// Bold spans — OpenCode `markdownStrong`.
    static let proseStrong = DynamicColor(light: rgb(0xD68C27), dark: rgb(0xF5A742))
    /// Italic spans — OpenCode `markdownEmph`.
    static let proseEmphasis = DynamicColor(light: rgb(0xB0851F), dark: rgb(0xE5C07B))
    /// Inline code — OpenCode `markdownCode`.
    static let proseCode = DynamicColor(light: rgb(0x3D9A57), dark: rgb(0x7FD88F))
    /// Links — OpenCode `markdownLink`.
    static let proseLink = DynamicColor(light: rgb(0x3B7DD8), dark: rgb(0xFAB283))
    /// List bullets — OpenCode `markdownListItem`.
    static let proseListMarker = DynamicColor(light: rgb(0x3B7DD8), dark: rgb(0xFAB283))
    /// Ordered-list ordinals — OpenCode `markdownListEnumeration`. A cool hue
    /// on purpose: bold, italic, and bullets are all warm, and numbered lists
    /// of bold terms otherwise read as a wall of orange.
    static let proseListEnumeration = DynamicColor(light: rgb(0x318795), dark: rgb(0x56B6C2))
    /// Failed tool rows — OpenCode `error`. Softer than the system red, which
    /// overpowers the muted transcript.
    static let chatError = DynamicColor(light: rgb(0xD1383D), dark: rgb(0xE06C75))
}

extension InlineStyle {
    /// GitHub's inline metrics with the Prose Accent Palette colors. Inline
    /// code drops GitHub's gray chip — the accent color alone marks it, as in
    /// OpenCode.
    static var proseAccents: InlineStyle {
        InlineStyle()
            .code(.monospaced, .fontScale(0.85), .foregroundColor(.proseCode))
            .strong(.fontWeight(.semibold), .foregroundColor(.proseStrong))
            .emphasis(.italic, .foregroundColor(.proseEmphasis))
            .link(.foregroundColor(.proseLink), .underlineStyle(.single))
    }
}

/// GitHub's heading scale and H1/H2 divider, tinted with the heading accent.
/// No text underline on H1 (OpenCode's terminal underline maps to the divider
/// here).
struct AccentHeadingStyle: StructuredText.HeadingStyle {
    func makeBody(configuration: Configuration) -> some View {
        StructuredText.GitHubHeadingStyle.gitHub
            .makeBody(configuration: configuration)
            .foregroundStyle(DynamicColor.proseHeading)
    }
}

/// GitHub's hierarchical bullets, tinted with the list-marker accent.
struct AccentUnorderedListMarker: StructuredText.UnorderedListMarker {
    func makeBody(configuration: Configuration) -> some View {
        StructuredText.HierarchicalSymbolListMarker
            .hierarchical(.disc, .circle, .square)
            .makeBody(configuration: configuration)
            .foregroundStyle(DynamicColor.proseListMarker)
    }
}

/// GitHub's decimal ordinals, tinted with the list-enumeration accent.
struct AccentOrderedListMarker: StructuredText.OrderedListMarker {
    func makeBody(configuration: Configuration) -> some View {
        StructuredText.DecimalListMarker.decimal
            .makeBody(configuration: configuration)
            .foregroundStyle(DynamicColor.proseListEnumeration)
    }
}

/// The chat's markdown style bundle: GitHub block layout with the Prose
/// Accent Palette applied to headings, inline spans, and list markers. The
/// code block style here is nominal — `AssistantProseView` overrides it with
/// `CopyableCodeBlockStyle`.
struct ChatMarkdownStyle: StructuredText.Style {
    let inlineStyle: InlineStyle = .proseAccents
    let headingStyle = AccentHeadingStyle()
    let paragraphStyle: StructuredText.GitHubParagraphStyle = .gitHub
    let blockQuoteStyle: StructuredText.GitHubBlockQuoteStyle = .gitHub
    let codeBlockStyle: StructuredText.GitHubCodeBlockStyle = .gitHub
    let listItemStyle: StructuredText.DefaultListItemStyle = .default
    let unorderedListMarker = AccentUnorderedListMarker()
    let orderedListMarker = AccentOrderedListMarker()
    let tableStyle: StructuredText.GitHubTableStyle = .gitHub
    let tableCellStyle: StructuredText.GitHubTableCellStyle = .gitHub
    let thematicBreakStyle: StructuredText.GitHubThematicBreakStyle = .gitHub
}
