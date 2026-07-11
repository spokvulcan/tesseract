//
//  CopyableCodeBlockStyle.swift
//  tesseract
//
//  Every code block in the chat gets a hover copy button (PRD #174): skills
//  return paste-ready artifacts (corrected text, tweet variants, translations)
//  as fenced blocks — one block, one artifact — so one click copies the chosen
//  variant. Renders Textual's GitHub code-block geometry inline — minus
//  GitHub's 0.85 font scale, so code reads at the one transcript type size
//  like everything else — and uses its `CodeBlockProxy` pasteboard action.
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
    @State private var resetTask: Task<Void, Never>?

    var body: some View {
        // GitHub's block geometry (Overflow, 16pt padding, 6pt radius, same
        // line spacing) with two deliberate departures: no 0.85 font scale,
        // and the surface is the Code Accent Palette's panel background
        // (Textual's theme carries it too, but keeps the property internal).
        Overflow {
            configuration.label
                .textual.lineSpacing(.fontScaled(0.225))
                .fixedSize(horizontal: false, vertical: true)
                .monospaced()
                .padding(16)
        }
        .background(DynamicColor.codePanelBackground)
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .textual.blockSpacing(.init(top: 0, bottom: 16))
        .overlay(alignment: .topTrailing) { copyButton }
        .onHover { isHovering = $0 }
    }

    private var copyButton: some View {
        Button {
            configuration.codeBlock.copyToPasteboard()
            didCopy = true
            // Supersede any in-flight reset so rapid re-copies keep the
            // checkmark up for the full window.
            resetTask?.cancel()
            resetTask = Task {
                try? await Task.sleep(for: .seconds(1.5))
                guard !Task.isCancelled else { return }
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
