import SwiftUI
import MLXLMCommon

struct AgentToolCallMinimalView: View {
    let toolCall: ToolCall
    var toolResult: AgentChatMessage? = nil
    @State private var isExpanded = false

    private var normalizedArguments: [String: JSONValue] {
        ToolArgumentNormalizer.normalize(toolCall.function.arguments)
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button(action: {
                withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                    isExpanded.toggle()
                }
            }) {
                HStack(spacing: 8) {
                    Text(ToolDisplayHelpers.titleForTool(toolCall.function.name, arguments: normalizedArguments))
                        .font(.system(size: 13))
                        .foregroundStyle(toolResult?.isError == true ? .red : .secondary)
                    
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
            
            if isExpanded {
                VStack(alignment: .leading, spacing: 12) {
                    // Arguments
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Arguments")
                            .font(.system(size: 10, weight: .medium))
                            .foregroundStyle(.tertiary)
                            .textCase(.uppercase)
                        
                        Text(ToolDisplayHelpers.formatArguments(normalizedArguments))
                            .font(.system(size: 12, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                            .padding(8)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color(white: 0.1).opacity(0.5))
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                    }
                    
                    // Result
                    if let result = toolResult {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Result")
                                .font(.system(size: 10, weight: .medium))
                                .foregroundStyle(.tertiary)
                                .textCase(.uppercase)
                            
                            Text(result.content)
                                .font(.system(size: 12, design: .monospaced))
                                .foregroundStyle(result.isError ? .red : .primary)
                                .textSelection(.enabled)
                                .fixedSize(horizontal: false, vertical: true)
                                .padding(8)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(Color(white: 0.1).opacity(0.5))
                                .clipShape(RoundedRectangle(cornerRadius: 6))
                        }
                    }
                }
                .padding(.trailing, 14)
                .padding(.bottom, 8)
                .padding(.top, 6)
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
    }
}
