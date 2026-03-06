import SwiftUI
import MLXLMCommon

struct AgentToolCallView: View {
    let toolCall: ToolCall
    var toolResult: AgentChatMessage? = nil
    var isListStyle: Bool = false
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
                    if let result = toolResult {
                        if result.isError {
                            Image(systemName: "xmark")
                                .font(.system(size: 12, weight: .semibold))
                                .foregroundStyle(.red)
                                .frame(width: 16)
                        } else {
                            Image(systemName: "checkmark")
                                .font(.system(size: 12, weight: .semibold))
                                .foregroundStyle(.green)
                                .frame(width: 16)
                        }
                    } else {
                        Image(systemName: iconForTool(toolCall.function.name))
                            .font(.system(size: 12))
                            .foregroundStyle(.secondary)
                            .frame(width: 16)
                    }
                    
                    Text(titleForTool(toolCall.function.name, arguments: normalizedArguments))
                        .font(.system(size: 13))
                        .foregroundStyle(toolResult != nil ? (toolResult?.isError == true ? .red : .primary) : .secondary)
                    
                    Spacer()
                    
                    if toolResult == nil {
                        // Showing an activity indicator or just time could go here if needed
                    }
                    
                    Image(systemName: "chevron.right")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(.tertiary)
                        .rotationEffect(.degrees(isExpanded ? 90 : 0))
                }
                .padding(.vertical, 6)
                .padding(.horizontal, isListStyle ? 14 : 0)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            
            if isExpanded {
                VStack(alignment: .leading, spacing: 12) {
                    // Arguments
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Arguments")
                            .font(.system(size: 10, weight: .medium))
                            .foregroundStyle(.tertiary)
                            .textCase(.uppercase)
                        Text(formatToolCall(toolCall))
                            .font(.system(size: 12, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                    
                    // Result
                    if let result = toolResult {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Result")
                                .font(.system(size: 10, weight: .medium))
                                .foregroundStyle(.tertiary)
                                .textCase(.uppercase)
                            Text(result.content)
                                .font(.system(size: 12, design: .monospaced))
                                .foregroundStyle(.primary)
                                .textSelection(.enabled)
                        }
                    }
                }
                .padding(.leading, isListStyle ? 38 : 24)
                .padding(.trailing, 14)
                .padding(.bottom, 8)
                .padding(.top, 2)
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
    }
    
    private func iconForTool(_ name: String) -> String {
        switch name.lowercased() {
        case let n where n.contains("read"): return "doc.text"
        case let n where n.contains("write"): return "square.and.pencil"
        case let n where n.contains("edit"): return "pencil.line"
        case "ls": return "folder"
        case let n where n.contains("list"): return "folder"
        case let n where n.contains("memory"): return "brain"
        case let n where n.contains("task") || n.contains("goal"): return "checklist"
        case let n where n.contains("search"): return "magnifyingglass"
        default: return "wrench.adjustable"
        }
    }
    
    private func titleForTool(_ name: String, arguments: [String: JSONValue]?) -> String {
        guard let args = arguments else { return name }
        
        switch name.lowercased() {
        case "read_file":
            if case .string(let path)? = args["path"] {
                let filename = (path as NSString).lastPathComponent
                return "Reading \(filename)"
            }
            return "Reading file"
        case "write_file":
            if case .string(let path)? = args["path"] {
                let filename = (path as NSString).lastPathComponent
                return "Writing to \(filename)"
            }
            return "Writing file"
        case "edit_file":
            if case .string(let path)? = args["path"] {
                let filename = (path as NSString).lastPathComponent
                return "Editing \(filename)"
            }
            return "Editing file"
        case "ls", "list", "list_files", "list_directory":
            if case .string(let path)? = args["path"] {
                let folder = (path as NSString).lastPathComponent
                return "Listing files in \(folder)"
            }
            return "Listing files"
        case "search_files":
            if case .string(let query)? = args["query"] {
                return "Searching for \"\(query)\""
            }
            return "Searching files"
        case "memory_save":
            return "Saving memory"
        case "task_create":
            return "Creating task"
        case "task_complete":
            return "Completing task"
        default:
            return name
        }
    }
    
    private func formatToolCall(_ toolCall: ToolCall) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

        do {
            let data = try encoder.encode(normalizedArguments)
            if let jsonString = String(data: data, encoding: .utf8) {
                return jsonString == "{}" ? "No arguments" : jsonString
            }
        } catch {
            return "\(normalizedArguments)"
        }

        return "No arguments"
    }
}
