//
//  SkillInvocationRowView.swift
//  tesseract
//
//  The **Skill Invocation Row** (PRD #174): the compact, trailing-aligned
//  user-block rendering of a fired skill — name, the user's argument text,
//  attachment thumbnails — expandable (+/−) to the exact injected `<skill>`
//  block. Wears the same block dress as a typed user message (it *is* a user
//  turn — owner-approved as-is; do not restyle).
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
                    .font(.system(size: chatBodyFontSize))
                    .foregroundStyle(.primary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            if isExpanded {
                Text(block.injectedBlock)
                    .font(.system(size: chatBodyFontSize, design: .monospaced))
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

    /// Marker + skill name. The whole line toggles the injected-block detail.
    private var header: some View {
        Button {
            withAnimation(.easeOut(duration: 0.15)) { isExpanded.toggle() }
        } label: {
            HStack(spacing: 8) {
                CollapseMarker(isExpanded: isExpanded)
                Text(block.displayLabel)
                    .font(.system(size: chatBodyFontSize, weight: .medium))
                    .foregroundStyle(.primary)
                Spacer(minLength: 12)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(isExpanded ? "Hide the injected skill text" : "Show the injected skill text")
    }
}
