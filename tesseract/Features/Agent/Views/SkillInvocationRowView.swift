//
//  SkillInvocationRowView.swift
//  tesseract
//
//  The **Skill Invocation Row** (PRD #174): the compact, right-aligned
//  rendering of a fired skill — name, the user's argument text, attachment
//  thumbnails — expandable to the exact injected `<skill>` block (the same
//  transparency philosophy as the System Prompt Inspector). Replaces the
//  wall-of-text user bubble for every invocation surface (pill or slash).
//

import SwiftUI

struct SkillInvocationRowView: View, Equatable {
    let data: SkillInvocationRow
    let rowID: String

    @Environment(AgentCoordinator.self) private var coordinator

    // Equality compares only the row data — the environment-injected
    // coordinator is read lazily on user action, never during diffing.
    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.data == rhs.data && lhs.rowID == rhs.rowID
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !data.images.isEmpty {
                HStack(spacing: 8) {
                    ForEach(data.images) { attachment in
                        AsyncImageAttachmentView(attachment: attachment)
                    }
                }
            }

            header

            if !data.argumentText.isEmpty {
                Text(data.argumentText)
                    .font(.system(size: chatBodyFontSize - 2))
                    .foregroundStyle(.primary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            if data.isExpanded {
                Text(data.injectedBlock)
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(.quinary.opacity(0.5))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(.quinary)
        .clipShape(
            .rect(
                topLeadingRadius: 18,
                bottomLeadingRadius: 18,
                bottomTrailingRadius: 4,
                topTrailingRadius: 18
            )
        )
    }

    /// Skill badge + timestamp + the expansion chevron. The whole line toggles
    /// the injected-block detail through the shared expansion state, so the
    /// toggle survives transcript rebuilds.
    private var header: some View {
        Button {
            coordinator.toggleDetailExpanded(rowID)
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "sparkles")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.tint)
                Text(data.displayLabel)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.primary)
                Spacer(minLength: 12)
                Text(data.timestamp)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                Image(systemName: "chevron.right")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(.tertiary)
                    .rotationEffect(.degrees(data.isExpanded ? 90 : 0))
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(data.isExpanded ? "Hide the injected skill text" : "Show the injected skill text")
    }
}
