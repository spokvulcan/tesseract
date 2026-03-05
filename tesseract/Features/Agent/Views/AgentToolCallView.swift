import SwiftUI
import MLXLMCommon

struct AgentToolCallView: View {
    let toolCall: ToolCall
    @State private var isExpanded = false
    @State private var isHovering = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button(action: {
                withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                    isExpanded.toggle()
                }
            }) {
                HStack(spacing: 8) {
                    Image(systemName: iconForTool(toolCall.function.name))
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(.tint)
                        .frame(width: 24, height: 24)
                        .background(Color.accentColor.opacity(0.15), in: Circle())
                    
                    Text(titleForTool(toolCall.function.name))
                        .font(.system(size: 14, weight: .medium))
                        .foregroundStyle(.primary)
                    
                    Spacer()
                    
                    Image(systemName: "chevron.right")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundStyle(.secondary)
                        .rotationEffect(.degrees(isExpanded ? 90 : 0))
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            
            if isExpanded {
                Divider()
                    .padding(.horizontal, 12)
                
                VStack(alignment: .leading, spacing: 8) {
                    Text("Function Call")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                        .textCase(.uppercase)
                    
                    Text(formatToolCall(toolCall))
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(.primary)
                        .textSelection(.enabled)
                        .padding(10)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.black.opacity(0.2))
                        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                }
                .padding(12)
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .background(Color(white: 0.1))
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(Color.white.opacity(0.1), lineWidth: 1)
        )
        .padding(.vertical, 4)
    }
    
    private func iconForTool(_ name: String) -> String {
        switch name.lowercased() {
        case let n where n.contains("read"): return "doc.text.magnifyingglass"
        case let n where n.contains("write"): return "square.and.pencil"
        case let n where n.contains("edit"): return "pencil.line"
        case let n where n.contains("list"): return "list.bullet.rectangle"
        case let n where n.contains("memory"): return "brain"
        case let n where n.contains("task") || n.contains("goal"): return "checklist"
        case let n where n.contains("search"): return "magnifyingglass"
        default: return "wrench.adjustable"
        }
    }
    
    private func titleForTool(_ name: String) -> String {
        // Convert camelCase or snake_case to Title Case beautifully if needed
        return name
    }
    
    private func formatToolCall(_ toolCall: ToolCall) -> String {
        let name = toolCall.function.name
        
        // MLXLMCommon tool arguments are [String: JSONValue].
        // Let's try to convert it to a nice JSON string
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        
        do {
            let data = try encoder.encode(toolCall.function.arguments)
            if let jsonString = String(data: data, encoding: .utf8) {
                return "{\n  \"name\": \"\(name)\",\n  \"arguments\": \(jsonString)\n}"
            }
        } catch {
            return "name: \(name)\narguments: \(toolCall.function.arguments)"
        }
        
        return "name: \(name)"
    }
}
