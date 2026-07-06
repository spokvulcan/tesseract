//
//  SkillInvocationRowView.swift
//  tesseract
//
//  The **Skill Invocation Row** (PRD #174): the compact, trailing-aligned
//  rendering of a fired skill — name, the user's argument text, attachment
//  thumbnails — expandable to the exact injected `<skill>` block (the same
//  transparency philosophy as the System Prompt Inspector). Replaces the
//  wall-of-text user block for every invocation surface (pill or slash).
//

import SwiftUI

struct SkillInvocationRowView: View {
    let block: SkillInvocationBlock
    let images: [ImageAttachment]
    let timestamp: Date

    @State private var isExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !images.isEmpty {
                HStack(spacing: 8) {
                    ForEach(images) { attachment in
                        AsyncImageAttachmentView(attachment: attachment)
                    }
                }
            }

            header

            if !block.argumentText.isEmpty {
                Text(block.argumentText)
                    .font(.system(size: chatBodyFontSize - 2))
                    .foregroundStyle(.primary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            if isExpanded {
                Text(block.injectedBlock)
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(.quinary, in: RoundedRectangle(cornerRadius: 6))
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.quinary, in: RoundedRectangle(cornerRadius: 12))
        .help(timestamp.formatted(date: .abbreviated, time: .shortened))
        .frame(maxWidth: .infinity, alignment: .trailing)
    }

    /// Skill badge + the expansion chevron. The whole line toggles the
    /// injected-block detail.
    private var header: some View {
        Button {
            withAnimation(.easeOut(duration: 0.15)) { isExpanded.toggle() }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "sparkles")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.secondary)
                Text(block.displayLabel)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.primary)
                Spacer(minLength: 12)
                Image(systemName: "chevron.right")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(.tertiary)
                    .rotationEffect(.degrees(isExpanded ? 90 : 0))
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(isExpanded ? "Hide the injected skill text" : "Show the injected skill text")
    }
}
