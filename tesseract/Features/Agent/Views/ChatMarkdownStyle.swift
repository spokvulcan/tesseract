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

// MARK: - Code Accent Palette (prose code blocks)

extension StructuredText.HighlighterTheme {
    /// Prose code blocks in the Code Accent Palette (PRD #200): the same
    /// system-semantic colors the Tool Panels use, so code looks identical
    /// everywhere in the transcript. Textual tokenizes with its bundled
    /// Prism.js — its token names map onto the palette's roles here; the
    /// panels' Rust engine (`tesseract-highlight`) cannot be injected into
    /// Textual's renderer without forking it, so the *palette* is the shared
    /// contract while the tokenizers differ.
    static let codeAccents: Self = {
        // Prism token name(s) → palette role. Built imperatively — a single
        // dictionary literal of this size stalls the type checker.
        let roles: [(tokens: [TokenType], role: CodeTokenRole)] = [
            ([.keyword, .literal], .keyword),
            ([.boolean, .constant], .constant),
            ([.string, .char, .regex], .string),
            ([.number], .number),
            ([.className, .builtin], .type),
            ([.function, .functionName], .function),
            ([.attribute, .attributeName, .directive, .preprocessor], .attribute),
            ([.comment, .blockComment, .docComment], .comment),
        ]
        var properties: [TokenType: AnyTextProperty] = [:]
        for (tokens, role) in roles {
            let property = AnyTextProperty(.foregroundColor(DynamicColor.codeAccent(role)))
            for token in tokens {
                properties[token] = property
            }
        }
        return Self(
            foregroundColor: .codePanelPlain,
            backgroundColor: .codePanelBackground,
            tokenProperties: properties
        )
    }()
}

extension DynamicColor {
    /// Base text of prose code blocks — near-label, both appearances.
    fileprivate static let codePanelPlain = DynamicColor(
        light: Color(red: 0, green: 0, blue: 0, opacity: 0.85),
        dark: Color(red: 1, green: 1, blue: 1, opacity: 0.85)
    )
    /// Block background, matching Textual's GitHub block surface.
    fileprivate static let codePanelBackground = DynamicColor(
        light: Color(red: 0.960784, green: 0.960784, blue: 0.968627),
        dark: Color(red: 0.120543, green: 0.122844, blue: 0.141312)
    )
}

extension DynamicColor {
    /// A Code Accent Palette role as a `DynamicColor` — the system semantic
    /// colors already adapt to light/dark, so both variants are the same.
    fileprivate static func codeAccent(_ role: CodeTokenRole) -> DynamicColor {
        let color = Color.codeAccent(role)
        return DynamicColor(light: color, dark: color)
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
