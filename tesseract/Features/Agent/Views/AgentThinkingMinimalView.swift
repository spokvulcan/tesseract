import SwiftUI

struct AgentThinkingMinimalView: View {
    let content: String
    @State private var isExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button(action: {
                withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                    isExpanded.toggle()
                }
            }) {
                HStack(spacing: 8) {
                    Text("Thinking")
                        .font(.system(size: 13))
                        .foregroundStyle(.secondary)

                    Spacer()

                    Image(systemName: "chevron.right")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(.tertiary)
                        .rotationEffect(.degrees(isExpanded ? 90 : 0))
                }
                .padding(.vertical, 0)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            thinkingContent
        }
    }

    @ViewBuilder
    private var thinkingContent: some View {
        if isExpanded {
            Text(content)
                .font(.system(size: 13))
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .padding(.trailing, 14)
                .padding(.bottom, 8)
                .padding(.top, 4)
                .frame(maxWidth: .infinity, alignment: .leading)
                .fixedSize(horizontal: false, vertical: true)
                .transition(.opacity.combined(with: .move(edge: .top)))
        } else {
            Text(content)
                .font(.system(size: 13))
                .foregroundStyle(.tertiary)
                .lineLimit(2)
                .truncationMode(.tail)
                .padding(.trailing, 14)
                .padding(.bottom, 8)
                .padding(.top, 4)
                .frame(maxWidth: .infinity, alignment: .leading)
                .allowsHitTesting(false)
        }
    }
}
