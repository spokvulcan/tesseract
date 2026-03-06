import SwiftUI
import MLXLMCommon

struct AgentToolResultBubbleView: View {
    let message: AgentChatMessage
    @State private var isExpanded = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button(action: {
                withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                    isExpanded.toggle()
                }
            }) {
                HStack(spacing: 8) {
                    Image(systemName: message.isError ? "xmark" : "checkmark")
                        .font(.system(size: 12))
                        .foregroundStyle(message.isError ? .red : .secondary)
                        .frame(width: 16)

                    Text(message.isError ? "Failed" : "Done")
                        .font(.system(size: 13))
                        .foregroundStyle(message.isError ? .red : .secondary)
                    
                    Spacer()
                    
                    Image(systemName: "chevron.right")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(.tertiary)
                        .rotationEffect(.degrees(isExpanded ? 90 : 0))
                }
                .padding(.vertical, 6)
                .padding(.horizontal, 14)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            
            if isExpanded {
                VStack(alignment: .leading, spacing: 8) {
                    Text(message.content)
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .padding(.leading, 38)
                        .padding(.trailing, 14)
                        .padding(.bottom, 6)
                }
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
    }
}
