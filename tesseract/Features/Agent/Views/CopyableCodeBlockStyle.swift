//
//  CopyableCodeBlockStyle.swift
//  tesseract
//
//  Every code block in the chat gets a hover copy button (PRD #174): skills
//  return paste-ready artifacts (corrected text, tweet variants, translations)
//  as fenced blocks — one block, one artifact — so one click copies the chosen
//  variant. Wraps Textual's GitHub code-block rendering and uses its
//  `CodeBlockProxy` pasteboard action.
//

import SwiftUI
import Textual

struct CopyableCodeBlockStyle: StructuredText.CodeBlockStyle {
    func makeBody(configuration: Configuration) -> some View {
        CopyableCodeBlock(configuration: configuration)
    }
}

/// Per-block wrapper so each block owns its hover/copied state.
private struct CopyableCodeBlock: View {
    let configuration: StructuredText.CodeBlockStyleConfiguration

    @State private var isHovering = false
    @State private var didCopy = false

    var body: some View {
        StructuredText.GitHubCodeBlockStyle()
            .makeBody(configuration: configuration)
            .overlay(alignment: .topTrailing) { copyButton }
            .onHover { isHovering = $0 }
    }

    private var copyButton: some View {
        Button {
            configuration.codeBlock.copyToPasteboard()
            didCopy = true
            Task {
                try? await Task.sleep(for: .seconds(1.5))
                didCopy = false
            }
        } label: {
            Image(systemName: didCopy ? "checkmark" : "doc.on.doc")
                .font(.system(size: 12))
                .foregroundStyle(didCopy ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary))
                .frame(width: 24, height: 24)
                .background(.quinary, in: RoundedRectangle(cornerRadius: 6))
        }
        .buttonStyle(.plain)
        .padding(6)
        .opacity(isHovering || didCopy ? 1 : 0)
        .animation(.easeInOut(duration: 0.15), value: isHovering)
        .help("Copy code block")
    }
}
