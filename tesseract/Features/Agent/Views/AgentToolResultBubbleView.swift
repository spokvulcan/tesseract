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
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 14))
                        .foregroundStyle(.green)
                    
                    Text("Tool Execution Result")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundStyle(.primary)
                    
                    Spacer()
                    
                    Text(message.timestamp.formatted(date: .omitted, time: .shortened))
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                    
                    Image(systemName: "chevron.right")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundStyle(.secondary)
                        .rotationEffect(.degrees(isExpanded ? 90 : 0))
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            
            if isExpanded {
                Divider()
                    .padding(.horizontal, 14)
                
                VStack(alignment: .leading, spacing: 8) {
                    Text(message.content)
                        .font(.system(size: 13, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .padding(10)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.black.opacity(0.15))
                        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                }
                .padding(14)
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .background(Color(white: 0.12))
        .clipShape(
            .rect(
                topLeadingRadius: 18,
                bottomLeadingRadius: 18,
                bottomTrailingRadius: 18,
                topTrailingRadius: 18
            )
        )
        .overlay(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .strokeBorder(Color.white.opacity(0.08), lineWidth: 1)
        )
    }
}
