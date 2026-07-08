//
//  ToolPanelViews.swift
//  tesseract
//
//  Thin renderers over the Tool Panel model (PRD #200): every decision —
//  panel kind, rows, tints, line numbers, cap state — is made by
//  `ToolPanelBuilder`; these views only draw. The Code Accent Palette is the
//  second sanctioned exception to the monochrome content layer (beside the
//  Prose Accent Palette): syntax token roles plus the semantic diff tints,
//  all derived from system semantic colors so light/dark follow the system.
//  No nested scroll views, ever — the Panel Cap expands content inline and
//  the transcript remains the only scroller.
//

import SwiftUI
import Textual

// MARK: - Code Accent Palette

extension Color {
    /// Syntax role → color, derived from system semantic colors (adaptive).
    static func codeAccent(_ role: CodeTokenRole) -> Color {
        switch role {
        case .plain, .variable: .primary
        case .keyword: Color(nsColor: .systemPink)
        case .string: Color(nsColor: .systemOrange)
        case .number, .constant: Color(nsColor: .systemPurple)
        case .comment: .secondary
        case .type: Color(nsColor: .systemTeal)
        case .function: Color(nsColor: .systemBlue)
        case .attribute: Color(nsColor: .systemIndigo)
        }
    }

    /// Semantic diff row tints — like error red, these are meaning, never
    /// decoration. Backgrounds, so the syntax colors stay legible on top.
    static let diffAddedRow = Color(nsColor: .systemGreen).opacity(0.13)
    static let diffRemovedRow = Color(nsColor: .systemRed).opacity(0.12)
    /// Stronger tint on the word-level changed range inside a modified line.
    static let diffAddedEmphasis = Color(nsColor: .systemGreen).opacity(0.32)
    static let diffRemovedEmphasis = Color(nsColor: .systemRed).opacity(0.30)
}

// MARK: - ToolPanelView

/// Dispatches a committed tool result to its specialized panel.
struct ToolPanelView: View {
    let panel: ToolPanel
    /// Pretty-printed arguments, for the generic panel only.
    let argsFormatted: String

    var body: some View {
        switch panel {
        case .diff(_, let rows):
            CodeRowsPanel(rows: rows, showsDiffGutter: true, footnote: nil)
        case .code(_, let rows, let footnote):
            CodeRowsPanel(rows: rows, showsDiffGutter: false, footnote: footnote)
        case .text(let text):
            MonospacedTextPanel(text: text, isError: false)
        case .search(let results, let fallbackText):
            SearchResultsPanel(results: results, fallbackText: fallbackText)
        case .page(let title, let url, let body, let raw):
            PagePanel(title: title, url: url, markdown: body, raw: raw)
        case .status(let lines):
            StatusPanel(lines: lines)
        case .error(let message):
            MonospacedTextPanel(text: message, isError: true)
        case .generic(let resultText):
            GenericPanel(argsFormatted: argsFormatted, resultText: resultText)
        }
    }
}

// MARK: - Panel chrome

/// The one panel surface: full-width muted box, matching the transcript's
/// existing quinary boxes.
private struct PanelBox<Content: View>: View {
    @ViewBuilder var content: Content

    var body: some View {
        content
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.quinary, in: RoundedRectangle(cornerRadius: 6))
    }
}

/// The Panel Cap's expander row: quiet, inline, no scroll hijacking.
private struct ShowMoreRow: View {
    let hidden: Int
    let expand: () -> Void

    var body: some View {
        Button {
            expand()
        } label: {
            Text("Show \(hidden) more lines")
                .font(.system(size: chatBodyFontSize))
                .foregroundStyle(.tertiary)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help("Expand the remaining lines inline")
    }
}

// MARK: - Code / diff rows

/// Line-numbered highlighted rows — the shared body of the diff panel
/// (± tints, dual gutter) and the read/write panels (single gutter).
private struct CodeRowsPanel: View {
    let rows: [PanelCodeRow]
    let showsDiffGutter: Bool
    let footnote: String?

    @State private var capExpanded = false

    var body: some View {
        let (visible, hidden) = PanelCap.split(rows, expanded: capExpanded)
        let gutterWidth = gutterWidth

        VStack(alignment: .leading, spacing: 2) {
            PanelBox {
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(visible) { row in
                        CodeRowView(
                            row: row, showsDiffGutter: showsDiffGutter, gutterWidth: gutterWidth)
                    }
                    if hidden > 0 {
                        ShowMoreRow(hidden: hidden) {
                            withAnimation(.easeOut(duration: 0.15)) { capExpanded = true }
                        }
                        .padding(.top, 4)
                    }
                }
                .textSelection(.enabled)
            }
            if let footnote {
                Text(footnote)
                    .font(.system(size: chatBodyFontSize))
                    .foregroundStyle(.tertiary)
            }
        }
    }

    /// Width of one gutter column, sized to the largest line number shown.
    private var gutterWidth: CGFloat {
        let maxNumber = rows.reduce(1) { max($0, $1.oldLine ?? 0, $1.newLine ?? 0) }
        let digits = max(2, String(maxNumber).count)
        // Monospaced digit advance at the transcript size is ~0.6 × point size.
        return CGFloat(digits) * chatBodyFontSize * 0.62 + 4
    }
}

private struct CodeRowView: View {
    let row: PanelCodeRow
    let showsDiffGutter: Bool
    let gutterWidth: CGFloat

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 0) {
            if showsDiffGutter {
                gutterText(row.oldLine)
                gutterText(row.newLine)
                    .padding(.trailing, 8)
            } else {
                gutterText(row.newLine)
                    .padding(.trailing, 8)
            }
            spansText
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.vertical, 1)
        .background(rowBackground)
    }

    private func gutterText(_ number: Int?) -> some View {
        Text(number.map(String.init) ?? "")
            .font(.system(size: chatBodyFontSize, design: .monospaced))
            .monospacedDigit()
            .foregroundStyle(.tertiary)
            .frame(width: gutterWidth, alignment: .trailing)
    }

    /// The row's spans as one attributed line, so selection reads as a single
    /// line of code and emphasized runs carry their own background tint.
    private var spansText: Text {
        var attributed = AttributedString()
        for span in row.spans {
            var piece = AttributedString(span.text)
            piece.foregroundColor = Color.codeAccent(span.role)
            if span.emphasized {
                piece.backgroundColor =
                    row.kind == .removed ? .diffRemovedEmphasis : .diffAddedEmphasis
            }
            attributed += piece
        }
        return Text(attributed)
            .font(.system(size: chatBodyFontSize, design: .monospaced))
    }

    @ViewBuilder
    private var rowBackground: some View {
        switch row.kind {
        case .context:
            EmptyView()
        case .added:
            Color.diffAddedRow
        case .removed:
            Color.diffRemovedRow
        }
    }
}

// MARK: - Monospaced text panel

/// Plain monospaced content (ls listings, page maps, skill text) and the
/// semantic-error rendering — the direct descendant of the old raw boxes,
/// with the Panel Cap applied.
private struct MonospacedTextPanel: View {
    let text: String
    let isError: Bool

    @State private var capExpanded = false

    var body: some View {
        let lines = text.trimmingCharacters(in: .newlines)
            .split(separator: "\n", omittingEmptySubsequences: false)
        let (visible, hidden) = PanelCap.split(Array(lines), expanded: capExpanded)

        PanelBox {
            VStack(alignment: .leading, spacing: 0) {
                Text(visible.joined(separator: "\n"))
                    .font(.system(size: chatBodyFontSize, design: .monospaced))
                    .foregroundStyle(
                        isError
                            ? AnyShapeStyle(DynamicColor.chatError)
                            : AnyShapeStyle(.secondary)
                    )
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                if hidden > 0 {
                    ShowMoreRow(hidden: hidden) {
                        withAnimation(.easeOut(duration: 0.15)) { capExpanded = true }
                    }
                    .padding(.top, 4)
                }
            }
        }
    }
}

// MARK: - Search results panel

private struct SearchResultsPanel: View {
    let results: [PanelSearchHit]
    let fallbackText: String?

    var body: some View {
        if let fallbackText {
            MonospacedTextPanel(text: fallbackText, isError: false)
        } else {
            PanelBox {
                VStack(alignment: .leading, spacing: 10) {
                    ForEach(results) { hit in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(hit.title)
                                .font(.system(size: chatBodyFontSize, weight: .medium))
                                .foregroundStyle(.primary)
                            Text(hit.url)
                                .font(.system(size: chatBodyFontSize))
                                .foregroundStyle(.tertiary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                            if !hit.snippet.isEmpty {
                                Text(hit.snippet)
                                    .font(.system(size: chatBodyFontSize))
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }
                .textSelection(.enabled)
            }
        }
    }
}

// MARK: - Page panel (fetch / read_page)

/// Fetched pages render as transcript markdown by default; the quiet
/// `Rendered · Raw` header switch drops to the exact returned text. The
/// choice is a global reading preference (PRD #200).
private struct PagePanel: View {
    let title: String?
    let url: String?
    let markdown: String
    let raw: String

    @AppStorage("toolPanelPageShowsRaw") private var showsRaw = false
    @State private var capExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            header
            if showsRaw {
                MonospacedTextPanel(text: raw, isError: false)
            } else {
                renderedBody
            }
        }
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            if let title, !title.isEmpty {
                Text(title)
                    .font(.system(size: chatBodyFontSize, weight: .medium))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            if let url, !url.isEmpty {
                Text(url)
                    .font(.system(size: chatBodyFontSize))
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            Spacer(minLength: 8)
            modeSwitch
        }
    }

    /// The quiet text switch: the active mode reads secondary, the inactive
    /// one tertiary — no control chrome in the content layer.
    private var modeSwitch: some View {
        HStack(spacing: 4) {
            modeButton("Rendered", isActive: !showsRaw) { showsRaw = false }
            Text("·").foregroundStyle(.quaternary)
            modeButton("Raw", isActive: showsRaw) { showsRaw = true }
        }
        .font(.system(size: chatBodyFontSize))
    }

    private func modeButton(_ label: String, isActive: Bool, action: @escaping () -> Void)
        -> some View
    {
        Button(action: action) {
            Text(label)
                .foregroundStyle(isActive ? AnyShapeStyle(.secondary) : AnyShapeStyle(.tertiary))
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(label == "Raw" ? "Show the exact returned text" : "Render the page as markdown")
    }

    private var renderedBody: some View {
        let lines = markdown.split(separator: "\n", omittingEmptySubsequences: false)
        let (visible, hidden) = PanelCap.splitMarkdown(Array(lines), expanded: capExpanded)

        return PanelBox {
            VStack(alignment: .leading, spacing: 8) {
                StructuredText(markdown: visible.joined(separator: "\n"))
                    .textual.structuredTextStyle(ChatMarkdownStyle())
                    .textual.codeBlockStyle(CopyableCodeBlockStyle())
                    .textual.highlighterTheme(.codeAccents)
                    .textual.textSelection(.enabled)
                    .font(.system(size: chatBodyFontSize))
                if hidden > 0 {
                    ShowMoreRow(hidden: hidden) {
                        withAnimation(.easeOut(duration: 0.15)) { capExpanded = true }
                    }
                }
            }
        }
    }
}

// MARK: - Status panel

/// Compact outcome lines for navigation-style calls — no box, no JSON.
private struct StatusPanel: View {
    let lines: [String]

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            ForEach(Array(lines.enumerated()), id: \.offset) { _, line in
                Text(line)
                    .font(.system(size: chatBodyFontSize))
                    .foregroundStyle(.secondary)
            }
        }
        .textSelection(.enabled)
    }
}

// MARK: - Generic panel

/// Unknown/external tools and detail-less legacy messages: the full
/// transparency form — pretty-printed arguments plus result text.
private struct GenericPanel: View {
    let argsFormatted: String
    let resultText: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            labeledSection("Arguments") {
                MonospacedTextPanel(text: argsFormatted, isError: false)
            }
            if let resultText, !resultText.isEmpty {
                labeledSection("Result") {
                    MonospacedTextPanel(text: resultText, isError: false)
                }
            }
        }
    }

    private func labeledSection(_ label: String, @ViewBuilder content: () -> some View)
        -> some View
    {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.system(size: chatBodyFontSize, weight: .medium))
                .foregroundStyle(.tertiary)
            content()
        }
    }
}
